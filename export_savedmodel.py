import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_USE_LEGACY_KERAS"] = "1"

# Workaround for protobuf incompatibility (slower, but avoids crash), protobuf needs to be a lower version for this version of tensorflow to work but bad practice, so using slower pure python version
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"

import sys
import importlib
import tensorflow as tf
import tf2onnx

import tf_keras
from tf_keras.models import load_model
from tf_keras.src.utils import generic_utils as tfk_generic_utils

from das.kapre.time_frequency import Spectrogram, Melspectrogram
from das.kapre.utils import Normalization2D
from das.tcn import tcn as das_tcn

# ---- Patch tf_keras Lambda/function loading (marshal-safe) ----
_real_func_load = tfk_generic_utils.func_load

def _resolve_dotted(name: str):
    mod_name, attr = name.rsplit(".", 1)
    mod = importlib.import_module(mod_name)
    return getattr(mod, attr)

def safe_func_load(code, defaults=None, closure=None, globs=None):
    try:
        return _real_func_load(code, defaults=defaults, closure=closure, globs=globs)
    except Exception:
        if isinstance(code, (tuple, list)):
            for item in code:
                if isinstance(item, str) and "." in item:
                    try:
                        return _resolve_dotted(item)
                    except Exception:
                        pass
        return (lambda x, **kw: x)

tfk_generic_utils.func_load = safe_func_load
# -------------------------------------------------------------

def savedmodel_to_onnx(saved_model_dir: str, onnx_path: str, opset: int = 13):
    old_argv = sys.argv[:]
    try:
        sys.argv = [
            "tf2onnx.convert",
            "--saved-model", saved_model_dir,
            "--opset", str(opset),
            "--output", onnx_path,
        ]
        tf2onnx.convert.main()
    finally:
        sys.argv = old_argv

custom_objects = {
    "Spectrogram": Spectrogram,
    "Melspectrogram": Melspectrogram,
    "Normalization2D": Normalization2D,
    "das.tcn.tcn": das_tcn,
    "tcn": das_tcn,
}

print("TensorFlow:", tf.__version__)
print("tf_keras:", tf_keras.__version__, tf_keras.__file__)

m = load_model("20260301_155538_model.h5", custom_objects=custom_objects, compile=False)

inp = m.inputs[0]
dummy = tf.zeros([1] + list(inp.shape[1:]), dtype=inp.dtype)
_ = m(dummy, training=False)

m.save("das_savedmodel", include_optimizer=False)
print("SavedModel written to ./das_savedmodel")

savedmodel_to_onnx("das_savedmodel", "das_model.onnx", opset=13)
print("ONNX written to das_model.onnx")