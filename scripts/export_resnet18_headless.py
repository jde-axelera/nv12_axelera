#!/usr/bin/env python3
"""
Export ResNet18 without classification head for embedding feature extraction.

Removes the final FC (1000-class) layer and keeps everything up to global
average pooling.  Output shape: [1, 512].

Usage:
    python scripts/export_resnet18_headless.py [output.onnx]
"""
import sys
import torch
import torch.nn as nn

try:
    import torchvision.models as models
except ImportError:
    sys.exit("ERROR: torchvision not found.  pip install torchvision")

out_path = sys.argv[1] if len(sys.argv) > 1 else "resnet18_embedding.onnx"

# Load pretrained weights, replace FC with Identity → output is [B, 512]
m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
m.fc = nn.Identity()
m.eval()

dummy = torch.zeros(1, 3, 224, 224)
with torch.no_grad():
    out = m(dummy)
assert out.shape == (1, 512), f"Unexpected output shape: {out.shape}"

torch.onnx.export(
    m,
    dummy,
    out_path,
    input_names=["input"],
    output_names=["embedding"],
    dynamic_axes=None,
    opset_version=13,
    do_constant_folding=True,
)
print(f"Exported: {out_path}")
print(f"  Input:  [1, 3, 224, 224]  (ImageNet RGB float32)")
print(f"  Output: [1, 512]          (global avg pool embedding)")

# Optional: simplify with onnxsim
try:
    import onnxsim
    import onnx
    model_proto = onnx.load(out_path)
    simplified, ok = onnxsim.simplify(model_proto)
    if ok:
        onnx.save(simplified, out_path)
        print("  Simplified with onnxsim")
except ImportError:
    pass
