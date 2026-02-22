"""Background route planner using Gemini vision."""

import base64
import json
import re
import threading
import time
from queue import Queue, Empty
from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.gemini.gemini_client import GeminiClient
from src.detection.object_tracker import TrackedObject
from .route_types import PlannedRoute, RouteWaypoint
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DestinationMatcher:
    """
    Matches spoken destination requests to detected objects using Gemini.

    When user says "take me to the chair", this matches it to the best
    detected object (by class name, position, etc).
    """

    def __init__(self, gemini_client: GeminiClient):
        """
        Initialize destination matcher.

        Args:
            gemini_client: Shared Gemini client for API calls.
        """
        self.gemini = gemini_client
        self._lock = threading.Lock()

    def match_destination(
        self,
        destination_text: str,
        tracks: List[TrackedObject],
        min_safe_distance_m: float = 2.0
    ) -> Optional[int]:
        """
        Match a spoken destination to a tracked object.

        Args:
            destination_text: User's spoken destination (e.g., "the chair", "table").
            tracks: List of currently tracked objects.
            min_safe_distance_m: Minimum distance for valid targets.

        Returns:
            Track ID of the best matching object, or None if no match.
        """
        with self._lock:
            # Filter to valid targets
            valid_tracks = [t for t in tracks
                          if t.get_current_depth() >= min_safe_distance_m]

            if not valid_tracks:
                logger.warning(f"[DEST] No valid targets for '{destination_text}'")
                return None

            # Build object list for Gemini
            obj_list = []
            for t in valid_tracks:
                obj_list.append(f"  ID {t.track_id}: {t.class_name} at {t.get_current_depth():.1f}m")

            prompt = f"""The user wants to navigate to: "{destination_text}"

Here are the objects I can see:
{chr(10).join(obj_list)}

Which object best matches what the user wants? Consider:
- Exact class name matches (e.g., "chair" matches "chair")
- Similar objects (e.g., "seat" could match "chair", "couch")
- If multiple match, prefer the closest one
- If nothing matches well, pick the most likely candidate

Reply with ONLY the ID number of the best matching object. Just the number, nothing else.
Example response: 5"""

            try:
                response = self.gemini.generate(prompt)
                if not response:
                    logger.warning("[DEST] Gemini returned no response")
                    return self._fallback_match(destination_text, valid_tracks)

                # Parse response - should just be a number
                response = response.strip()
                match = re.search(r'\d+', response)
                if match:
                    track_id = int(match.group())
                    # Verify this ID exists
                    if any(t.track_id == track_id for t in valid_tracks):
                        logger.info(f"[DEST] Matched '{destination_text}' -> track {track_id}")
                        return track_id

                logger.warning(f"[DEST] Could not parse response: {response}")
                return self._fallback_match(destination_text, valid_tracks)

            except Exception as e:
                logger.error(f"[DEST] Gemini error: {e}")
                return self._fallback_match(destination_text, valid_tracks)

    def _fallback_match(
        self,
        destination_text: str,
        tracks: List[TrackedObject]
    ) -> Optional[int]:
        """
        Simple fallback matching based on class name substring.

        Args:
            destination_text: User's spoken destination.
            tracks: List of valid tracked objects.

        Returns:
            Track ID of best match, or furthest object if no match.
        """
        dest_lower = destination_text.lower()

        # Try exact substring match
        for t in tracks:
            if t.class_name.lower() in dest_lower or dest_lower in t.class_name.lower():
                logger.info(f"[DEST] Fallback matched '{destination_text}' -> {t.class_name} (track {t.track_id})")
                return t.track_id

        # No match - return furthest object
        if tracks:
            furthest = max(tracks, key=lambda t: t.get_current_depth())
            logger.info(f"[DEST] No match for '{destination_text}', using furthest: {furthest.class_name}")
            return furthest.track_id

        return None


class RoutePlanner:
    """
    Background route planner using Gemini vision.

    Runs every ~1 second, analyzes bird's eye view, plans optimal route.
    """

    def __init__(
        self,
        api_key: str,
        planning_interval: float = 1.0,
        image_size: Tuple[int, int] = (200, 200),
        jpeg_quality: int = 60,
        min_safe_distance_m: float = 2.0,
        max_route_age_seconds: float = 3.0
    ):
        """
        Initialize route planner.

        Args:
            api_key: Gemini API key.
            planning_interval: Seconds between planning cycles.
            image_size: Size to resize bird's eye view for Gemini.
            jpeg_quality: JPEG compression quality (1-100).
            min_safe_distance_m: Objects closer than this are avoided.
            max_route_age_seconds: Discard route if older than this.
        """
        self.planning_interval = planning_interval
        self.image_size = image_size
        self.jpeg_quality = jpeg_quality
        self.min_safe_distance_m = min_safe_distance_m
        self.max_route_age_seconds = max_route_age_seconds

        # Gemini client
        self.gemini = GeminiClient(api_key=api_key, model="gemini-2.5-flash-lite")

        # Destination matcher for voice commands
        self.destination_matcher = DestinationMatcher(self.gemini)

        # Thread-safe state
        self._current_route: Optional[PlannedRoute] = None
        self._last_good_route: Optional[PlannedRoute] = None  # Fallback route
        self._route_lock = threading.Lock()
        self._target_destination: Optional[str] = None  # Voice-requested destination
        self._target_track_id: Optional[int] = None  # Resolved target track ID

        # Input queue: (birdseye_image, tracks, timestamp)
        self._input_queue: Queue = Queue(maxsize=1)

        # Threading
        self._thread: Optional[threading.Thread] = None
        self._running = False

        # Stats
        self._last_plan_time = 0.0
        self._consecutive_failures = 0

        # Smoothing parameters
        self._min_similarity_to_keep = 0.5  # Keep old route if >50% similar
        self._min_waypoints_for_update = 2  # Need at least 2 waypoints to update

    def set_destination(self, destination_text: str) -> None:
        """
        Set a voice-requested destination.

        The next planning cycle will try to match this to a detected object.

        Args:
            destination_text: User's spoken destination (e.g., "the chair").
        """
        with self._route_lock:
            self._target_destination = destination_text
            self._target_track_id = None  # Will be resolved on next planning cycle
            self._current_route = None  # Clear current route to force replanning
            self._last_good_route = None
        logger.info(f"[ROUTE] Destination set: '{destination_text}'")

    def clear_destination(self) -> None:
        """Clear voice-requested destination, return to furthest-object mode."""
        with self._route_lock:
            self._target_destination = None
            self._target_track_id = None
        logger.info("[ROUTE] Destination cleared, returning to furthest-object mode")

    def get_current_destination(self) -> Optional[str]:
        """Get the current voice-requested destination, if any."""
        with self._route_lock:
            return self._target_destination

    def start(self) -> None:
        """Start background planning thread."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._planning_loop, daemon=True)
        self._thread.start()
        logger.info("Route planner started")

    def stop(self) -> None:
        """Stop background planning thread."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        logger.info("Route planner stopped")

    def update(
        self,
        birdseye_image: np.ndarray,
        tracks: List[TrackedObject],
        timestamp: Optional[float] = None
    ) -> None:
        """
        Provide latest data to planner (called from main loop).

        Non-blocking: drops old data if planner is busy.
        """
        if timestamp is None:
            timestamp = time.monotonic()

        # Replace queue contents (drop old if present)
        try:
            self._input_queue.get_nowait()
        except Empty:
            pass

        # Deep copy tracks to avoid mutation issues
        tracks_copy = []
        for t in tracks:
            tracks_copy.append(TrackedObject(
                track_id=t.track_id,
                bbox=t.bbox,
                class_id=t.class_id,
                class_name=t.class_name,
                confidence=t.confidence,
                depth_history=t.depth_history.copy(),
                last_seen=t.last_seen,
                frames_tracked=t.frames_tracked,
                closing_speed=t.closing_speed
            ))

        try:
            self._input_queue.put_nowait((birdseye_image.copy(), tracks_copy, timestamp))
        except:
            pass  # Queue full, skip this frame

    def get_current_route(self, current_tracks: Optional[List[TrackedObject]] = None) -> Optional[PlannedRoute]:
        """
        Get current planned route (thread-safe).

        Matches waypoints to current tracks by POSITION (depth), not by ID,
        since track IDs change frequently but positions are stable.

        Args:
            current_tracks: Current visible tracks to match waypoints against.

        Returns None if no valid route.
        """
        with self._route_lock:
            route = self._current_route
            if route is None:
                route = self._last_good_route

            if route is None or not route.waypoints:
                return None

            # If no current tracks provided, return route as-is
            if current_tracks is None:
                return route

            # Match waypoints to current tracks by depth (position)
            # For each waypoint depth, find the closest current track
            visitable = [t for t in current_tracks
                        if t.get_current_depth() >= self.min_safe_distance_m]

            if not visitable:
                return None

            matched_waypoints = []
            used_track_ids = set()

            for wp in route.waypoints:
                # Find closest track by depth (within 1.5m tolerance)
                best_match = None
                best_diff = 1.5  # Max depth difference to consider a match

                for track in visitable:
                    if track.track_id in used_track_ids:
                        continue
                    depth_diff = abs(track.get_current_depth() - wp.depth_m)
                    if depth_diff < best_diff:
                        best_diff = depth_diff
                        best_match = track

                if best_match:
                    used_track_ids.add(best_match.track_id)
                    matched_waypoints.append(RouteWaypoint(
                        track_id=best_match.track_id,
                        class_name=best_match.class_name,
                        order=len(matched_waypoints) + 1,
                        depth_m=best_match.get_current_depth(),
                        reasoning=wp.reasoning
                    ))

            if not matched_waypoints:
                return None

            # Ensure we always end at the furthest visible object
            furthest = max(visitable, key=lambda t: t.get_current_depth())
            if furthest.track_id not in used_track_ids:
                matched_waypoints.append(RouteWaypoint(
                    track_id=furthest.track_id,
                    class_name=furthest.class_name,
                    order=len(matched_waypoints) + 1,
                    depth_m=furthest.get_current_depth(),
                    reasoning="furthest"
                ))

            # Sort by depth
            matched_waypoints.sort(key=lambda w: w.depth_m)
            for i, wp in enumerate(matched_waypoints):
                wp.order = i + 1

            return PlannedRoute(
                waypoints=matched_waypoints,
                avoid_ids=[],
                target_id=matched_waypoints[-1].track_id if matched_waypoints else None,
                timestamp=route.timestamp,
                is_valid=True
            )

    def _planning_loop(self) -> None:
        """Background planning loop."""
        logger.debug("Route planning thread started")

        while self._running:
            try:
                # Wait for input data
                try:
                    birdseye_image, tracks, timestamp = self._input_queue.get(timeout=0.5)
                except Empty:
                    continue

                # Rate limiting
                elapsed = time.monotonic() - self._last_plan_time
                if elapsed < self.planning_interval:
                    time.sleep(self.planning_interval - elapsed)

                # Check for voice-requested destination that needs matching
                with self._route_lock:
                    destination = self._target_destination
                    target_id = self._target_track_id

                if destination and target_id is None:
                    # Need to match destination to a track
                    matched_id = self.destination_matcher.match_destination(
                        destination, tracks, self.min_safe_distance_m
                    )
                    if matched_id is not None:
                        with self._route_lock:
                            self._target_track_id = matched_id
                        target_id = matched_id
                        logger.info(f"[ROUTE] Matched destination '{destination}' to track {matched_id}")

                # Plan route (will use target_id if set)
                new_route = self._plan_route(birdseye_image, tracks, target_track_id=target_id)

                if new_route and new_route.waypoints:
                    # Ensure route is in depth order (nearest to furthest)
                    if not new_route.is_monotonic_depth():
                        new_route = new_route.sort_by_depth()

                    # Apply smoothing - decide whether to update
                    should_update = self._should_update_route(new_route, tracks)

                    if should_update:
                        with self._route_lock:
                            self._current_route = new_route
                            self._last_good_route = new_route
                        self._consecutive_failures = 0

                        # Log route to terminal
                        waypoint_strs = [f"#{w.track_id} {w.class_name} ({w.depth_m:.1f}m)" for w in new_route.waypoints]
                        target_depth = new_route.waypoints[-1].depth_m if new_route.waypoints else 0
                        logger.warning(f"[ROUTE] NEW: {' -> '.join(waypoint_strs)} [target: {target_depth:.0f}m]")
                    else:
                        # Keep existing route but refresh timestamp
                        with self._route_lock:
                            if self._current_route:
                                self._current_route.timestamp = time.time()
                                logger.info(f"[ROUTE] KEEPING existing route (stable)")
                else:
                    self._consecutive_failures += 1
                    if self._consecutive_failures > 3:
                        logger.warning(f"Route planning failed {self._consecutive_failures} times")

                self._last_plan_time = time.monotonic()

            except Exception as e:
                logger.error(f"Error in planning loop: {e}")
                self._consecutive_failures += 1

        logger.debug("Route planning thread stopped")

    def _should_update_route(
        self,
        new_route: PlannedRoute,
        current_tracks: List[TrackedObject]
    ) -> bool:
        """
        Decide whether to update to a new route.

        Only update if new route reaches significantly further.
        Position matching in get_current_route handles track ID changes.
        """
        with self._route_lock:
            old_route = self._current_route

        # Always update if no existing route
        if old_route is None or not old_route.waypoints:
            return True

        # Get target depths
        old_target_depth = old_route.waypoints[-1].depth_m if old_route.waypoints else 0
        new_target_depth = new_route.waypoints[-1].depth_m if new_route.waypoints else 0

        # Only update if new route reaches >5m further
        if new_target_depth > old_target_depth + 5.0:
            return True

        # Keep old route - position matching will update the actual track IDs
        return False

    def _plan_route(
        self,
        birdseye_image: np.ndarray,
        tracks: List[TrackedObject],
        target_track_id: Optional[int] = None
    ) -> Optional[PlannedRoute]:
        """
        Plan route using Gemini vision, with fallback to simple depth-based route.

        Args:
            birdseye_image: Bird's eye view image.
            tracks: List of tracked objects.
            target_track_id: Specific target track ID (from voice command), or None for furthest.

        Returns:
            PlannedRoute on success, fallback route if Gemini fails.
        """
        # Filter tracks with valid depth
        valid_tracks = [t for t in tracks if t.get_current_depth() > 0]
        visitable = [t for t in valid_tracks if t.get_current_depth() >= self.min_safe_distance_m]

        if not visitable:
            return None

        # Determine target track
        target_track = None
        if target_track_id is not None:
            # Voice-requested specific target
            target_track = next((t for t in visitable if t.track_id == target_track_id), None)
            if target_track is None:
                # Target track no longer visible - clear destination
                logger.warning(f"[ROUTE] Target track {target_track_id} no longer visible")
                with self._route_lock:
                    self._target_destination = None
                    self._target_track_id = None

        if target_track is None:
            # Default to furthest object
            target_track = max(visitable, key=lambda t: t.get_current_depth())

        # Build prompt with track info (limits to 8 objects)
        prompt = self._build_prompt(valid_tracks, target_track)

        if prompt is None:
            return self._create_fallback_route(visitable, target_track)

        # Encode image
        image_b64 = self._encode_image(birdseye_image)

        # Call Gemini
        response = self.gemini.analyze_image(image_b64, prompt, "image/jpeg")

        if not response:
            logger.warning("[ROUTE] Gemini failed, using fallback")
            return self._create_fallback_route(visitable, target_track)

        # Parse response
        result = self._parse_response(response, valid_tracks)

        if result is None or not result.waypoints:
            logger.warning("[ROUTE] Parse failed, using fallback")
            return self._create_fallback_route(visitable, target_track)

        # Ensure target is the final waypoint
        result = self._ensure_target(result, visitable, target_track)

        return result

    def _create_fallback_route(
        self,
        visitable_tracks: List[TrackedObject],
        target_track: Optional[TrackedObject] = None
    ) -> PlannedRoute:
        """Create a simple depth-ordered route when Gemini fails."""
        if not visitable_tracks:
            return PlannedRoute(is_valid=False)

        # Determine target
        if target_track is None:
            target_track = max(visitable_tracks, key=lambda t: t.get_current_depth())

        # Get tracks up to and including the target
        target_depth = target_track.get_current_depth()
        tracks_to_target = [t for t in visitable_tracks if t.get_current_depth() <= target_depth]

        # Sort by depth (nearest first)
        sorted_tracks = sorted(tracks_to_target, key=lambda t: t.get_current_depth())

        # Take up to 4 waypoints: nearest, middle points, and target
        if len(sorted_tracks) <= 4:
            selected = sorted_tracks
        else:
            # Pick nearest, 2 evenly spaced in between, and target
            selected = [
                sorted_tracks[0],
                sorted_tracks[len(sorted_tracks) // 3],
                sorted_tracks[2 * len(sorted_tracks) // 3],
            ]
            # Ensure target is included
            if target_track not in selected:
                selected.append(target_track)

        # Ensure target is last
        selected = [t for t in selected if t.track_id != target_track.track_id]
        selected.append(target_track)

        waypoints = []
        for i, t in enumerate(selected):
            waypoints.append(RouteWaypoint(
                track_id=t.track_id,
                class_name=t.class_name,
                order=i + 1,
                depth_m=t.get_current_depth(),
                reasoning="fallback"
            ))

        return PlannedRoute(
            waypoints=waypoints,
            target_id=target_track.track_id,
            is_valid=True
        )

    def _ensure_target(
        self,
        route: PlannedRoute,
        visitable_tracks: List[TrackedObject],
        target_track: TrackedObject
    ) -> PlannedRoute:
        """Ensure the route ends at the specified target object."""
        if not route.waypoints or not visitable_tracks:
            return route

        target_id = target_track.track_id

        # Check if target is already the last waypoint
        if route.waypoints[-1].track_id == target_id:
            return route

        # Check if target is already in route
        route_ids = {w.track_id for w in route.waypoints}

        if target_id in route_ids:
            # Move it to the end
            new_waypoints = [w for w in route.waypoints if w.track_id != target_id]
            target_wp = next(w for w in route.waypoints if w.track_id == target_id)
            target_wp.order = len(new_waypoints) + 1
            new_waypoints.append(target_wp)
        else:
            # Add target to the end
            new_waypoints = list(route.waypoints)
            new_waypoints.append(RouteWaypoint(
                track_id=target_id,
                class_name=target_track.class_name,
                order=len(new_waypoints) + 1,
                depth_m=target_track.get_current_depth(),
                reasoning="target"
            ))

        # Re-sort by depth to maintain order
        new_waypoints.sort(key=lambda w: w.depth_m)
        for i, wp in enumerate(new_waypoints):
            wp.order = i + 1

        # Remove waypoints that are beyond the target depth
        target_depth = target_track.get_current_depth()
        new_waypoints = [w for w in new_waypoints if w.depth_m <= target_depth]

        # Ensure target is still the last one
        if not new_waypoints or new_waypoints[-1].track_id != target_id:
            # Re-add target
            new_waypoints = [w for w in new_waypoints if w.track_id != target_id]
            new_waypoints.append(RouteWaypoint(
                track_id=target_id,
                class_name=target_track.class_name,
                order=len(new_waypoints) + 1,
                depth_m=target_track.get_current_depth(),
                reasoning="target"
            ))

        # Re-sort and renumber
        new_waypoints.sort(key=lambda w: w.depth_m)
        for i, wp in enumerate(new_waypoints):
            wp.order = i + 1

        return PlannedRoute(
            waypoints=new_waypoints,
            target_id=target_id,
            avoid_ids=route.avoid_ids,
            planning_notes=route.planning_notes,
            is_valid=True
        )

    def _encode_image(self, image: np.ndarray) -> str:
        """Resize and encode image to base64 JPEG."""
        resized = cv2.resize(image, self.image_size)
        _, buffer = cv2.imencode(
            '.jpg',
            resized,
            [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        )
        return base64.b64encode(buffer).decode('utf-8')

    def _build_prompt(self, tracks: List[TrackedObject], target_track: TrackedObject) -> str:
        """Build Gemini prompt with track information."""
        # Limit to max 8 objects to avoid overwhelming Gemini
        # Only include objects up to and including the target
        target_depth = target_track.get_current_depth()
        visitable = [t for t in tracks
                    if t.get_current_depth() >= self.min_safe_distance_m
                    and t.get_current_depth() <= target_depth]
        too_close = [t for t in tracks if t.get_current_depth() < self.min_safe_distance_m]

        if not visitable:
            # No visitable objects
            return None

        # Sort by depth and take up to 7 nearest + always include target
        visitable_sorted = sorted(visitable, key=lambda t: t.get_current_depth())

        # Take nearest 7 (or fewer) plus ensure target is included
        selected = visitable_sorted[:7]
        if target_track not in selected:
            selected.append(target_track)

        # Build simple object list - just ID and distance
        obj_list = []
        for t in selected:
            marker = " [TARGET]" if t.track_id == target_track.track_id else ""
            obj_list.append(f"  {t.track_id}: {t.class_name} at {t.get_current_depth():.1f}m{marker}")

        avoid_list = [str(t.track_id) for t in too_close[:3]]  # Limit avoid list too

        return f"""Plan a safe route for a visually impaired person's navigation robot. Visit waypoints in order, ending at the TARGET.

OBJECTS (id: type at distance):
{chr(10).join(obj_list)}

AVOID (too close): {', '.join(avoid_list) if avoid_list else 'none'}

CRITICAL DANGER - NEVER route towards these:
- ROADS, STREETS, HIGHWAYS - absolute top priority to avoid! Never cross or approach roads.
- Moving vehicles: cars, trucks, buses, motorcycles, bikes

SAFETY RULES - Skip these as waypoints (but you can pass by them):
- Moving objects: people, animals (unpredictable movement)
- Trip hazards: backpacks, bags, luggage on the ground
- Unstable objects: bottles, cups, small items that could be knocked over
- Sharp/dangerous: knives, scissors, glass
- Hot surfaces: ovens, stoves, microwaves (if recently used)
- Obstacles at leg height: chairs, stools, ottomans, low tables

PREFER as waypoints:
- Fixed/stable furniture: tables, desks, couches, beds, TVs
- Large stable objects: refrigerators, doors, plants, shelves
- Stationary landmarks: traffic lights, signs, benches
- Stay on sidewalks/paths, NEVER on roads

Reply with ONLY this JSON (no markdown, no explanation):
{{"waypoints":[{{"id":{selected[0].track_id}}},{{"id":{target_track.track_id}}}]}}

Replace the example with your actual route using the exact IDs above. Include 2-4 waypoints total, ending with {target_track.track_id}. Only include safe, stable waypoints. NEVER include roads as waypoints."""

    def _parse_response(
        self,
        response: str,
        tracks: List[TrackedObject]
    ) -> Optional[PlannedRoute]:
        """Parse Gemini JSON response into PlannedRoute."""
        try:
            # Extract JSON from response (may have markdown code blocks)
            json_match = re.search(r'\{[\s\S]*\}', response)
            if not json_match:
                logger.warning(f"[ROUTE] No JSON found")
                return None

            json_str = json_match.group()
            data = json.loads(json_str)

            # Build track lookup
            track_map = {t.track_id: t for t in tracks}

            # Parse waypoints - handle both {"id": X} and {"track_id": X}
            waypoints = []
            for i, wp_data in enumerate(data.get("waypoints", [])):
                # Support both "id" and "track_id" keys
                track_id = wp_data.get("id") or wp_data.get("track_id")

                if track_id is None:
                    continue

                if track_id not in track_map:
                    continue  # Track no longer exists

                track = track_map[track_id]
                depth = track.get_current_depth()

                # Skip if too close
                if depth < self.min_safe_distance_m:
                    continue

                waypoints.append(RouteWaypoint(
                    track_id=track_id,
                    class_name=track.class_name,
                    order=i + 1,
                    depth_m=depth,
                    reasoning=""
                ))

            # Target is the last waypoint
            target_id = waypoints[-1].track_id if waypoints else None

            return PlannedRoute(
                waypoints=waypoints,
                avoid_ids=[],
                target_id=target_id,
                planning_notes="",
                is_valid=True
            )

        except json.JSONDecodeError as e:
            logger.warning(f"[ROUTE] JSON error: {e}")
            return None
        except Exception as e:
            logger.error(f"[ROUTE] Parse error: {e}")
            return None
