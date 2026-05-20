"""
Calibration preprocessing transform for axcompile (pixel/255 normalization).

Applies the same preprocessing as C++ rgba_to_tensor:
  float_val = pixel / 255.0  (no ImageNet mean/std subtraction)

This matches the default axcompile calibration range.  Use this transform so
the quantization parameters cover exactly [0, 1] input values.

The two fix blocks at the bottom work around axcompile multiprocessing issues:
  Fix 1 — self-registration in sys.modules so pickle can find this function.
  Fix 2 — cloudpickle ForkingPickler so axcompile's own internal lambdas
           (_get_real_image_calibration_dataloader.<locals>.<lambda>) can be
           serialised across worker processes.
"""
import torch
import torchvision.transforms as T
from PIL import Image


def get_preprocess_transform(image) -> torch.Tensor:
    """Resize to 224x224 and normalize to [0, 1] — no ImageNet mean/std."""
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    return T.Compose([
        T.Resize(224),
        T.CenterCrop(224),
        T.ToTensor(),   # PIL → float32 CHW in [0, 1]
    ])(image)


# ── Fix 1: register module in sys.modules so pickle can find this function ──
# axcompile loads us via spec_from_file_location without calling
# sys.modules[spec.name] = module, so pickle's check fails ("not the same object").
import sys as _sys, types as _types
if __name__ not in _sys.modules:
    _m = _types.ModuleType(__name__)
    _m.__dict__.update(globals())
    _sys.modules[__name__] = _m

# ── Fix 2: replace ForkingPickler with cloudpickle so axcompile's internal
# lambdas (_get_real_image_calibration_dataloader.<locals>.<lambda>) can be
# serialised.  cloudpickle handles local-scope functions natively. ──
try:
    import pickle as _pickle
    import cloudpickle as _cp
    import multiprocessing as _mp
    import multiprocessing.reduction as _mpr
    import multiprocessing.queues as _mpq

    if not getattr(_pickle, '_cp_patched', False):
        _pickle.dumps = _cp.dumps
        _pickle.dump  = _cp.dump

        _OrigFP = _mpr.ForkingPickler

        class _CPForkingPickler(_cp.CloudPickler):
            _extra_reducers         = _OrigFP._extra_reducers
            _copyreg_dispatch_table = _OrigFP._copyreg_dispatch_table

        _mpr.ForkingPickler                     = _CPForkingPickler
        _mpq.ForkingPickler                     = _CPForkingPickler
        _mpr.AbstractReducer.ForkingPickler     = _CPForkingPickler
        if hasattr(_mp, 'reduction'):
            _mp.reduction.ForkingPickler        = _CPForkingPickler

        _pickle._cp_patched = True
except Exception:
    pass
