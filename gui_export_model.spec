# gui_export_model.spec

from PyInstaller.utils.hooks import collect_all

block_cipher = None

# Collect everything from problematic packages
datas = []
binaries = []
hiddenimports = []

for pkg in ["tensorflow", "tf_keras", "tf2onnx", "onnxruntime"]:
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

# Manually add known missing modules
hiddenimports += [
    "tf2onnx.convert",
    "tf2onnx.tfonnx",
    "tf2onnx.optimizer",
    "tf2onnx.graph",
    "tf2onnx.utils",
    "tf_keras.src.engine.base_layer_v1",
    "tf_keras.src.engine.base_layer",
    "tf_keras.src.engine.input_layer",
]

a = Analysis(
    ["gui_export_model.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="LiveScope Prediction GUI",
    debug=False,
    strip=False,
    upx=True,
    console=False,
    icon="assets/donders.ico"
)