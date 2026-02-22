#!/usr/bin/env python3
"""
Export MobileNetV3-Small to CoreML format for ReID embeddings.

Run once:
    python scripts/export_reid_coreml.py

Creates models/reid_mobilenet.mlpackage for PyTorch-free inference.

Requirements (one-time export only):
    pip install torch torchvision coremltools
"""

from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
import coremltools as ct


class ReIDEmbedder(nn.Module):
    """
    MobileNetV3-Small backbone with embedding projection head.

    Extracts 256-dim L2-normalized embeddings from image crops.
    """

    def __init__(self, embedding_dim: int = 256):
        super().__init__()

        # Load pretrained MobileNetV3-Small
        backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)

        # Keep only the feature extractor (remove classifier)
        self.features = backbone.features

        # Global average pooling
        self.pool = nn.AdaptiveAvgPool2d(1)

        # MobileNetV3-Small outputs 576 channels after last conv
        self.projection = nn.Linear(576, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract features
        x = self.features(x)

        # Global average pool -> flatten
        x = self.pool(x).flatten(1)

        # Project to embedding space
        x = self.projection(x)

        # L2 normalize
        x = x / x.norm(dim=1, keepdim=True).clamp(min=1e-6)

        return x


def main():
    print("=" * 50)
    print("Exporting MobileNetV3-Small ReID to CoreML")
    print("=" * 50)

    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    output_path = models_dir / "reid_mobilenet.mlpackage"

    # Create model
    print("\nCreating ReID model...")
    model = ReIDEmbedder(embedding_dim=256)
    model.eval()

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,} ({total_params / 1e6:.2f}M)")

    # Create example input (128x128 RGB image)
    input_size = 128
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
    print("Converting to CoreML...")
    mlmodel = ct.convert(
        traced_model,
        inputs=[
            ct.ImageType(
                name="image",
                shape=(1, 3, input_size, input_size),
                scale=1/255.0,  # Normalize to [0, 1]
                bias=[0, 0, 0],  # No bias (ImageNet normalization happens in MobileNet)
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
    mlmodel.short_description = "MobileNetV3-Small ReID embedder for object tracking"
    mlmodel.version = "1.0"

    # Save model
    import shutil
    if output_path.exists():
        shutil.rmtree(output_path)

    mlmodel.save(str(output_path))

    print(f"\n{'=' * 50}")
    print(f"Exported to: {output_path}")
    print(f"{'=' * 50}")

    # Verify the exported model
    print("\nVerifying exported model...")
    loaded_model = ct.models.MLModel(str(output_path))
    spec = loaded_model.get_spec()

    print(f"Input: {spec.description.input[0].name}")
    print(f"Output: {spec.description.output[0].name}")

    # Test inference with PIL image
    from PIL import Image
    import numpy as np

    test_img = Image.fromarray(np.random.randint(0, 255, (input_size, input_size, 3), dtype=np.uint8))
    predictions = loaded_model.predict({"image": test_img})

    embedding = predictions["embedding"].flatten()
    print(f"\nTest embedding shape: {embedding.shape}")
    print(f"Test embedding L2 norm: {np.linalg.norm(embedding):.4f}")

    print("\nReID model export complete!")
    print("Now you can run tracking without PyTorch!")

    return 0


if __name__ == "__main__":
    exit(main())
