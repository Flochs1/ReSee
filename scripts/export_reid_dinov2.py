#!/usr/bin/env python3
"""
Export DINOv2-Small to CoreML format for ReID embeddings.

DINOv2 is a self-supervised vision model from Meta that excels at
instance-level discrimination without task-specific training.

Run once:
    python scripts/export_reid_dinov2.py

Creates models/reid_dinov2.mlpackage for PyTorch-free inference.

Requirements (one-time export only):
    pip install torch torchvision coremltools
"""

from pathlib import Path

import torch
import torch.nn as nn
import coremltools as ct


class DINOv2Embedder(nn.Module):
    """
    DINOv2-Small backbone for ReID embeddings.

    Uses the [CLS] token output and projects to 256-dim embedding.
    DINOv2-Small has 21M parameters and outputs 384-dim features.
    """

    def __init__(self, embedding_dim: int = 256):
        super().__init__()

        # Load DINOv2-Small from torch hub
        self.backbone = torch.hub.load(
            'facebookresearch/dinov2',
            'dinov2_vits14',  # Small variant, 14x14 patches
            pretrained=True
        )

        # DINOv2-Small outputs 384-dim features
        self.projection = nn.Linear(384, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get [CLS] token features
        features = self.backbone(x)  # (B, 384)

        # Project to embedding space
        x = self.projection(features)

        # L2 normalize
        x = x / x.norm(dim=1, keepdim=True).clamp(min=1e-6)

        return x


def main():
    print("=" * 60)
    print("Exporting DINOv2-Small ReID to CoreML")
    print("=" * 60)

    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    output_path = models_dir / "reid_dinov2.mlpackage"

    # Create model
    print("\nLoading DINOv2-Small from torch hub...")
    print("(This may take a moment on first run)")
    model = DINOv2Embedder(embedding_dim=256)
    model.eval()

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,} ({total_params / 1e6:.1f}M)")

    # DINOv2 expects 224x224 or 518x518 (multiple of 14 for patch size)
    # Use 224x224 for speed, can use 140x140 (10x10 patches) for faster inference
    input_size = 224
    example_input = torch.randn(1, 3, input_size, input_size)

    # Test forward pass
    print(f"\nTesting forward pass with input shape: {example_input.shape}")
    with torch.no_grad():
        output = model(example_input)
    print(f"Output shape: {output.shape}")
    print(f"Output L2 norm: {output.norm().item():.4f} (should be ~1.0)")

    # Trace the model
    print("\nTracing model...")
    traced_model = torch.jit.trace(model, example_input)

    # Convert to CoreML
    print("Converting to CoreML (this may take a minute)...")
    mlmodel = ct.convert(
        traced_model,
        inputs=[
            ct.ImageType(
                name="image",
                shape=(1, 3, input_size, input_size),
                scale=1/255.0,
                bias=[0, 0, 0],
                color_layout=ct.colorlayout.RGB
            )
        ],
        outputs=[
            ct.TensorType(name="embedding")
        ],
        minimum_deployment_target=ct.target.macOS13,
    )

    # Add metadata
    mlmodel.author = "ReSee"
    mlmodel.short_description = "DINOv2-Small ReID embedder for object tracking"
    mlmodel.version = "1.0"

    # Save model
    import shutil
    if output_path.exists():
        shutil.rmtree(output_path)

    mlmodel.save(str(output_path))

    print(f"\n{'=' * 60}")
    print(f"Exported to: {output_path}")
    print(f"{'=' * 60}")

    # Verify the exported model
    print("\nVerifying exported model...")
    loaded_model = ct.models.MLModel(str(output_path))
    spec = loaded_model.get_spec()

    print(f"Input: {spec.description.input[0].name}")
    print(f"Output: {spec.description.output[0].name}")

    # Test inference with PIL image
    from PIL import Image
    import numpy as np

    test_img = Image.fromarray(
        np.random.randint(0, 255, (input_size, input_size, 3), dtype=np.uint8)
    )
    predictions = loaded_model.predict({"image": test_img})

    embedding = predictions["embedding"].flatten()
    print(f"\nTest embedding shape: {embedding.shape}")
    print(f"Test embedding L2 norm: {np.linalg.norm(embedding):.4f}")

    print("\nDINOv2 ReID model export complete!")
    print("Update config/detection_config.yaml to use: models/reid_dinov2.mlpackage")

    return 0


if __name__ == "__main__":
    exit(main())
