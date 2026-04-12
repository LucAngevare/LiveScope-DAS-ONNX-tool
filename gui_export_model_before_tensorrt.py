import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_USE_LEGACY_KERAS"] = "1"

# Workaround for protobuf incompatibility (slower, but avoids crash), protobuf needs to be a lower version for this version of tensorflow to work but bad practice, so using slower pure python version
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"

import sys
import threading
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import time
import csv
import numpy as np

import tensorflow as tf
import tf2onnx
import importlib

import tf_keras
from tf_keras.models import load_model
from tf_keras.src.utils import generic_utils as tfk_generic_utils

from das.kapre.time_frequency import Spectrogram, Melspectrogram
from das.kapre.utils import Normalization2D
from das.tcn import tcn as das_tcn

# ONNX Runtime for predictions
try:
    import onnxruntime as ort
    HAS_ONNX_RUNTIME = True
except ImportError:
    HAS_ONNX_RUNTIME = False

# Audio file reading
try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False
    try:
        from scipy.io import wavfile
        HAS_SCIPY = True
    except ImportError:
        HAS_SCIPY = False

# GPU monitoring
try:
    import pynvml
    pynvml.nvmlInit()
    HAS_GPU_MONITOR = True
except:
    HAS_GPU_MONITOR = False

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
# ---- End patch ----

custom_objects = {
    "Spectrogram": Spectrogram,
    "Melspectrogram": Melspectrogram,
    "Normalization2D": Normalization2D,
    "das.tcn.tcn": das_tcn,
    "tcn": das_tcn,
}

def savedmodel_to_onnx(saved_model_dir: str, onnx_path: str, opset: int = 13):
    import tf2onnx.convert  # <-- FORCE import for PyInstaller

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

def export_model(h5_path: str, output_dir: str, progress_callback, opset: int = 13):
    try:
        h5_path = Path(h5_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        savedmodel_path = output_dir / "das_savedmodel"
        onnx_path = output_dir / "das_model.onnx"

        progress_callback("Loading model...", 20)
        m = load_model(str(h5_path), custom_objects=custom_objects, compile=False)
        progress_callback("Model loaded successfully", 40)

        progress_callback("Running dummy inference...", 50)
        inp = m.inputs[0]
        dummy = tf.zeros([1] + list(inp.shape[1:]), dtype=inp.dtype)
        _ = m(dummy, training=False)
        progress_callback("Dummy inference completed", 60)

        progress_callback("Saving as SavedModel...", 70)
        m.save(str(savedmodel_path), include_optimizer=False)
        progress_callback(f"SavedModel written to {savedmodel_path}", 80)

        progress_callback("Converting to ONNX...", 85)
        savedmodel_to_onnx(str(savedmodel_path), str(onnx_path), opset=opset)
        progress_callback(f"ONNX written to {onnx_path}", 100)

        progress_callback("SUCCESS", 100, done=True)

    except Exception as e:
        progress_callback(f"ERROR: {str(e)}", 100, done=True)

def load_audio(wav_path: str):
    """Load audio file and return as numpy array."""
    if HAS_SOUNDFILE:
        audio, sr = sf.read(wav_path)
        return audio, sr
    elif HAS_SCIPY:
        sr, audio = wavfile.read(wav_path)
        # Normalize if integer type
        if audio.dtype in [np.int16, np.int32]:
            audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
        return audio, sr
    else:
        raise ImportError("Neither soundfile nor scipy is available for reading audio files")

def predict_with_onnx(onnx_path: str, wav_path: str, output_csv: str, progress_callback):
    """Run prediction on WAV file using ONNX model."""
    try:
        if not HAS_ONNX_RUNTIME:
            raise ImportError("onnxruntime is not installed. Install with: pip install onnxruntime-gpu")
        
        progress_callback("Loading audio file...", 10)
        audio, sr = load_audio(wav_path)
        progress_callback(f"Audio loaded: {len(audio)} samples @ {sr}Hz", 20)
        
        # Initialize GPU monitoring if available
        gpu_handle = None
        initial_vram = 0
        if HAS_GPU_MONITOR:
            try:
                gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
                initial_vram = mem_info.used / (1024**3)  # GB
                progress_callback(f"Initial VRAM: {initial_vram:.3f} GB", 25)
            except:
                gpu_handle = None
        
        progress_callback("Loading ONNX model...", 30)
        # Try GPU first, fall back to CPU
        providers = []
        if ort.get_available_providers():
            if 'CUDAExecutionProvider' in ort.get_available_providers():
                providers.append('CUDAExecutionProvider')
            providers.append('CPUExecutionProvider')
        else:
            providers = ['CPUExecutionProvider']
        
        sess = ort.InferenceSession(onnx_path, providers=providers)
        progress_callback(f"Model loaded with: {sess.get_providers()[0]}", 40)
        
        # Get input shape
        input_info = sess.get_inputs()[0]
        input_name = input_info.name
        input_shape = input_info.shape
        progress_callback(f"Input shape: {input_shape}", 45)
        
        # Prepare input data (assuming mono audio needs reshaping to model input)
        # This may need adjustment based on your model's expected input format
        if len(audio.shape) == 1:
            audio = audio.reshape(-1, 1)  # (samples, 1)
        
        # Determine batch size from model
        batch_size = 1
        expected_samples = input_shape[1] if isinstance(input_shape[1], int) else 8192
        num_channels = input_shape[2] if isinstance(input_shape[2], int) else 1
        
        # Process audio in chunks if needed
        progress_callback("Running inference...", 50)
        start_time = time.perf_counter()
        
        # For now, process the first chunk (or pad/trim to expected size)
        if len(audio) < expected_samples:
            # Pad
            audio_chunk = np.pad(audio, ((0, expected_samples - len(audio)), (0, 0)), mode='constant')
        else:
            # Take first chunk
            audio_chunk = audio[:expected_samples, :]
        
        # Ensure correct number of channels
        if audio_chunk.shape[1] < num_channels:
            audio_chunk = np.tile(audio_chunk, (1, num_channels))[:, :num_channels]
        
        # Reshape to (batch, samples, channels)
        input_data = audio_chunk[np.newaxis, :, :].astype(np.float32)
        
        # Run inference
        outputs = sess.run(None, {input_name: input_data})
        
        end_time = time.perf_counter()
        inference_time = end_time - start_time
        progress_callback(f"Inference completed in {inference_time*1000:.2f}ms", 70)
        
        # Check VRAM usage
        if gpu_handle is not None:
            try:
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
                peak_vram = mem_info.used / (1024**3)  # GB
                vram_used = peak_vram - initial_vram
                progress_callback(f"VRAM used: {vram_used:.3f} GB (Peak: {peak_vram:.3f} GB)", 75)
            except:
                pass
        
        # Process outputs - assuming predictions are in the first output
        predictions = outputs[0]
        progress_callback(f"Output shape: {predictions.shape}", 80)
        
        # Export to CSV
        progress_callback("Exporting results to CSV...", 85)
        with open(output_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write metadata
            writer.writerow(['# Metadata'])
            writer.writerow(['Audio File', wav_path])
            writer.writerow(['Sample Rate', sr])
            writer.writerow(['Audio Length (samples)', len(audio)])
            writer.writerow(['Audio Duration (seconds)', len(audio) / sr])
            writer.writerow(['Inference Time (ms)', f'{inference_time*1000:.4f}'])
            writer.writerow(['Model', onnx_path])
            writer.writerow(['Provider', sess.get_providers()[0]])
            if gpu_handle is not None:
                writer.writerow(['VRAM Used (GB)', f'{vram_used:.4f}'])
            writer.writerow([])
            
            # Write predictions header
            writer.writerow(['# Predictions'])
            if len(predictions.shape) == 3:
                # (batch, time, classes)
                writer.writerow(['Time Step', 'Class Probabilities...'])
                for t in range(predictions.shape[1]):
                    row = [t] + predictions[0, t, :].tolist()
                    writer.writerow(row)
            elif len(predictions.shape) == 2:
                # (batch, classes) or (time, classes)
                writer.writerow(['Index', 'Values...'])
                for i in range(predictions.shape[0]):
                    row = [i] + predictions[i, :].tolist() if len(predictions.shape) > 1 else [i, predictions[i]]
                    writer.writerow(row)
            else:
                # Flatten and write
                writer.writerow(['Index', 'Value'])
                flat = predictions.flatten()
                for i, val in enumerate(flat):
                    writer.writerow([i, val])
        
        progress_callback(f"Results saved to {output_csv}", 95)
        progress_callback(f"SUCCESS - Inference: {inference_time*1000:.2f}ms", 100, done=True)
        
    except Exception as e:
        import traceback
        error_msg = f"ERROR: {str(e)}\n{traceback.format_exc()}"
        progress_callback(error_msg, 100, done=True)

class ModelExporterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("DAS Model Tool - Convert & Predict")
        self.root.geometry("700x550")
        
        # Convert tab variables
        self.model_file = tk.StringVar()
        self.output_dir = tk.StringVar(value=str(Path.cwd()))
        self.opset = tk.IntVar(value=13)
        
        # Predict tab variables
        self.onnx_file = tk.StringVar()
        self.wav_file = tk.StringVar()
        self.csv_output = tk.StringVar(value=str(Path.cwd() / "predictions.csv"))

        self.create_widgets()

    def create_widgets(self):
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        convert_frame = ttk.Frame(notebook, padding=10)
        notebook.add(convert_frame, text="Convert Model")
        self.create_convert_tab(convert_frame)
        
        predict_frame = ttk.Frame(notebook, padding=10)
        notebook.add(predict_frame, text="Predict with ONNX")
        self.create_predict_tab(predict_frame)
        
        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(status_frame, text=f"TensorFlow: {tf.__version__}").pack(side=tk.LEFT)
        if HAS_ONNX_RUNTIME:
            ttk.Label(status_frame, text=f"| ONNX Runtime: {ort.__version__}").pack(side=tk.LEFT)

    # ---------------- CONVERT TAB ----------------
    def create_convert_tab(self, frame):
        ttk.Label(frame, text="Select a .h5 model file to export").pack(anchor="w", pady=5)
        
        file_frame = ttk.Frame(frame)
        file_frame.pack(fill=tk.X)
        ttk.Entry(file_frame, textvariable=self.model_file, width=50, state="readonly").pack(side=tk.LEFT)
        ttk.Button(file_frame, text="Browse", command=self.browse_model).pack(side=tk.LEFT)

        ttk.Label(frame, text="Output Directory").pack(anchor="w", pady=5)
        
        out_frame = ttk.Frame(frame)
        out_frame.pack(fill=tk.X)
        ttk.Entry(out_frame, textvariable=self.output_dir, width=50).pack(side=tk.LEFT)
        ttk.Button(out_frame, text="Browse", command=self.browse_output).pack(side=tk.LEFT)

        ttk.Label(frame, text="ONNX Opset Version").pack(anchor="w", pady=5)
        ttk.Spinbox(frame, from_=1, to=20, textvariable=self.opset, width=5).pack(anchor="w")

        self.convert_progress = ttk.Progressbar(frame, length=600)
        self.convert_progress.pack(pady=10)

        self.convert_log = scrolledtext.ScrolledText(frame, height=10, state="disabled")
        self.convert_log.pack(fill=tk.BOTH, expand=True)

        btn_frame = ttk.Frame(frame)
        btn_frame.pack(pady=5)
        ttk.Button(btn_frame, text="Export", command=self.start_export).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Clear", command=self.clear_convert_fields).pack(side=tk.LEFT)

    def log_convert_message(self, msg, progress_value=0, done=False):
        self.convert_log.configure(state="normal")
        self.convert_log.insert(tk.END, msg + "\n")
        self.convert_log.see(tk.END)
        self.convert_log.configure(state="disabled")

        self.convert_progress['value'] = progress_value

        if done:
            if msg.startswith("SUCCESS"):
                messagebox.showinfo("Export Complete", "Model exported successfully!")
            else:
                messagebox.showerror("Export Failed", msg)

    def start_export(self):
        if not self.model_file.get():
            messagebox.showerror("Error", "Select a model file")
            return

        self.convert_progress['value'] = 0
        self.convert_log.configure(state="normal")
        self.convert_log.delete("1.0", tk.END)
        self.convert_log.configure(state="disabled")

        export_model(
            self.model_file.get(),
            self.output_dir.get(),
            self.log_convert_message,
            self.opset.get()
        )

    def clear_convert_fields(self):
        self.model_file.set("")
        self.output_dir.set(str(Path.cwd()))
        self.opset.set(13)
        self.convert_progress['value'] = 0
        self.convert_log.configure(state="normal")
        self.convert_log.delete("1.0", tk.END)
        self.convert_log.configure(state="disabled")

    # ---------------- PREDICT TAB ----------------
    def create_predict_tab(self, frame):
        if not HAS_ONNX_RUNTIME:
            ttk.Label(frame, text="ONNX Runtime not installed", foreground="red").pack()
            return
        
        ttk.Label(frame, text="Select ONNX model file").pack(anchor="w", pady=5)
        
        f1 = ttk.Frame(frame)
        f1.pack(fill=tk.X)
        ttk.Entry(f1, textvariable=self.onnx_file, width=50).pack(side=tk.LEFT)
        ttk.Button(f1, text="Browse", command=self.browse_onnx).pack(side=tk.LEFT)

        ttk.Label(frame, text="Select WAV file").pack(anchor="w", pady=5)
        
        f2 = ttk.Frame(frame)
        f2.pack(fill=tk.X)
        ttk.Entry(f2, textvariable=self.wav_file, width=50).pack(side=tk.LEFT)
        ttk.Button(f2, text="Browse", command=self.browse_wav).pack(side=tk.LEFT)

        ttk.Label(frame, text="CSV Output").pack(anchor="w", pady=5)
        
        f3 = ttk.Frame(frame)
        f3.pack(fill=tk.X)
        ttk.Entry(f3, textvariable=self.csv_output, width=50).pack(side=tk.LEFT)
        ttk.Button(f3, text="Browse", command=self.browse_csv).pack(side=tk.LEFT)

        self.predict_progress = ttk.Progressbar(frame, length=600)
        self.predict_progress.pack(pady=10)

        self.predict_log = scrolledtext.ScrolledText(frame, height=10, state="disabled")
        self.predict_log.pack(fill=tk.BOTH, expand=True)

        ttk.Button(frame, text="Run Prediction", command=self.start_prediction).pack(pady=5)

    def log_predict_message(self, msg, progress_value=0, done=False):
        self.predict_log.configure(state="normal")
        self.predict_log.insert(tk.END, msg + "\n")
        self.predict_log.see(tk.END)
        self.predict_log.configure(state="disabled")

        self.predict_progress['value'] = progress_value

        if done:
            if msg.startswith("SUCCESS"):
                messagebox.showinfo("Done", "Prediction complete!")
            else:
                messagebox.showerror("Error", msg)

    def start_prediction(self):
        predict_with_onnx(
            self.onnx_file.get(),
            self.wav_file.get(),
            self.csv_output.get(),
            self.log_predict_message
        )

    # ---------------- FILE BROWSERS ----------------
    def browse_model(self):
        path = filedialog.askopenfilename(filetypes=[("H5 Files", "*.h5")])
        if path:
            self.model_file.set(path)

    def browse_output(self):
        path = filedialog.askdirectory()
        if path:
            self.output_dir.set(path)

    def browse_onnx(self):
        path = filedialog.askopenfilename(filetypes=[("ONNX Files", "*.onnx")])
        if path:
            self.onnx_file.set(path)

    def browse_wav(self):
        path = filedialog.askopenfilename(filetypes=[("WAV Files", "*.wav")])
        if path:
            self.wav_file.set(path)

    def browse_csv(self):
        path = filedialog.asksaveasfilename(defaultextension=".csv")
        if path:
            self.csv_output.set(path)

if __name__ == "__main__":
    root = tk.Tk()
    app = ModelExporterApp(root)
    root.mainloop()
