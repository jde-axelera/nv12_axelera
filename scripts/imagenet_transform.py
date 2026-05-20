"""
Calibration preprocessing transform for axcompile.

axcompile calls get_preprocess_transform() on each calibration image.
Applies standard ImageNet validation preprocessing so the compiler can
compute accurate per-channel quantization parameters.
"""
import torchvision.transforms as T
from PIL import Image
import numpy as np


def get_preprocess_transform(image):
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    return transform(image)
