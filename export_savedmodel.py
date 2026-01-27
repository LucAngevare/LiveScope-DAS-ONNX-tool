import os
os.environ["CUDA_VISIBLE_DEVICES"] = "" # silence CUDA probing if needed

import tensorflow as tf
from das.kapre.time_frequency import Spectrogram, Melspectrogram
from das.kapre.utils import Normalization2D
from das.tcn import tcn as das_tcn
import tf2onnx
import sys

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
    # DAS sometimes stores this exact string in Lambda configs:
    "das.tcn.tcn": das_tcn,
    "tcn": das_tcn,
}

m = tf.keras.models.load_model("20251116_115659_model.h5", custom_objects=custom_objects, compile=False)

# Build concrete graph once (important for SavedModel export)
inp = m.inputs[0]
dummy = tf.zeros([1] + list(inp.shape[1:]), dtype=inp.dtype)
_ = m(dummy, training=False)

m.save("das_savedmodel", include_optimizer=False)
print("SavedModel written to ./das_savedmodel")

# now that has been done, we need to run python -m tf2onnx.convert --saved-model das_savedmodel --opset 13 --output das_model.onnx
# but because I don't like using bash and I want everything to run in a single script, I will use the module tf2onnx so I can continue to use python

savedmodel_to_onnx("das_savedmodel", "das_model.onnx", opset=13)