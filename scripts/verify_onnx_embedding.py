#!/usr/bin/env python3
"""
ONNX reference check for the ResNet18 embedding.

Runs the headless ResNet18 ONNX with the same preprocessing as the C++
binary and prints the first 6 embedding values.  Compare with the AIPU
output to gauge quantisation error.

Preprocessing modes:
  default (--no-imagenet)  pixel/255 only — matches C++ when the model was
                           compiled with random/default calibration (no --transform).
  --imagenet               ImageNet normalise — use only when the model was
                           compiled with --transform imagenet_transform.py.

Usage:
    # compare C++ (random-calib) AIPU output with ONNX pixel/255
    python scripts/verify_onnx_embedding.py \\
        --onnx models/resnet18_embedding.onnx \\
        --image input_images/dog_bike_768x576.rgba \\
        --size 768x576 \\
        --aipu aipu_embedding.txt

    # use ImageNet normalisation (for properly calibrated models)
    python scripts/verify_onnx_embedding.py \\
        --onnx models/resnet18_embedding.onnx \\
        --image input_images/dog_bike_768x576.rgba \\
        --size 768x576 --imagenet \\
        --aipu aipu_embedding.txt
"""
import argparse
import sys
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    sys.exit("ERROR: onnxruntime not found.  pip install onnxruntime")

try:
    import cv2
except ImportError:
    sys.exit("ERROR: opencv-python not found.  pip install opencv-python")


def load_image_rgba(path, w, h):
    raw = np.fromfile(path, dtype=np.uint8)
    return raw.reshape(h, w, 4)


def preprocess(rgba, target=224, imagenet_norm=False):
    """
    Resize to target×target and convert to 1CHW float32.

    imagenet_norm=False (default): pixel/255 only.
      Matches C++ rgba_to_tensor — use when the model was compiled with
      random/default calibration (scale≈1/256, zp=-128).

    imagenet_norm=True: (pixel/255 - mean) / std.
      Use only when the model was compiled with --transform imagenet_transform.py.
    """
    bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
    bgr = cv2.resize(bgr, (target, target), interpolation=cv2.INTER_LINEAR)
    rgb = bgr[:, :, ::-1].astype(np.float32) / 255.0        # HWC float RGB
    if imagenet_norm:
        mean = np.array([0.485, 0.456, 0.406], np.float32)
        std  = np.array([0.229, 0.224, 0.225], np.float32)
        rgb  = (rgb - mean) / std
    return rgb.transpose(2, 0, 1)[np.newaxis]                # 1CHW


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx",      required=True, help="Path to resnet18_embedding.onnx")
    p.add_argument("--image",     default=None,  help="Path to .rgba image")
    p.add_argument("--size",      default=None,  help="WxH for .rgba files, e.g. 768x576")
    p.add_argument("--aipu",      default=None,  help="Path to aipu_embedding.txt (C++ --output-emb)")
    p.add_argument("--imagenet",  action="store_true",
                   help="Apply ImageNet normalisation (use only for ImageNet-calibrated models)")
    args = p.parse_args()

    # Load and preprocess image
    if args.image and args.image.endswith(".rgba"):
        if not args.size:
            sys.exit("--size WxH is required for .rgba files")
        w, h = (int(x) for x in args.size.split("x"))
        rgba = load_image_rgba(args.image, w, h)
    elif args.image:
        bgr = cv2.imread(args.image)
        if bgr is None:
            sys.exit(f"Cannot read: {args.image}")
        rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGBA)
    else:
        print("[WARN] No image — using random input")
        rgba = (np.random.rand(224, 224, 4) * 255).astype(np.uint8)

    x = preprocess(rgba, imagenet_norm=args.imagenet)   # [1, 3, 224, 224]

    # ONNX inference
    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    emb = sess.run(None, {in_name: x})[0][0]   # [512]

    norm = float(np.linalg.norm(emb))
    print(f"[ONNX]  dim={len(emb)}  norm={norm:.4f}")
    print(f"  First 6: [{', '.join(f'{v:.5f}' for v in emb[:6])}]")

    # Compare with AIPU output if provided
    if args.aipu:
        aipu = np.loadtxt(args.aipu, dtype=np.float32)
        if len(aipu) != len(emb):
            print(f"[WARN] Size mismatch: ONNX={len(emb)}, AIPU={len(aipu)}")
        else:
            cos_sim = float(np.dot(emb, aipu) / (np.linalg.norm(emb) * np.linalg.norm(aipu) + 1e-9))
            l2      = float(np.linalg.norm(emb - aipu))
            print(f"\n[AIPU]  dim={len(aipu)}  norm={np.linalg.norm(aipu):.4f}")
            print(f"  First 6: [{', '.join(f'{v:.5f}' for v in aipu[:6])}]")
            print(f"\n[COMPARE]  cosine_similarity={cos_sim:.4f}  L2_error={l2:.4f}")
            if cos_sim > 0.99:
                print("  ✓ Excellent match (cosine > 0.99)")
            elif cos_sim > 0.95:
                print("  ~ Good match (cosine > 0.95)")
            else:
                print("  ! Poor match — check preprocessing alignment")


if __name__ == "__main__":
    main()
