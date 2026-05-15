@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM Stable TensorRT EP test stack for Windows/Python 3.10:
REM   ONNX Runtime GPU 1.22.0 + TensorRT 10.9.0.34 CUDA 12.x
REM Run from an Anaconda/Miniconda Prompt.

set "ENV_NAME=das_script"
set "SCRIPT_DIR=%~dp0"
set "BACKUP_DIR=%SCRIPT_DIR%stack_backups"
if not exist "%BACKUP_DIR%" mkdir "%BACKUP_DIR%"

call :activate_conda
if errorlevel 1 goto :fail

set "TS=%DATE:/=-%_%TIME::=-%"
set "TS=%TS: =0%"
set "TS=%TS:,=.%"
set "BACKUP_PIP=%BACKUP_DIR%\pip_freeze_before_modern_cuda12_%TS%.txt"
set "BACKUP_CONDA=%BACKUP_DIR%\conda_env_before_modern_cuda12_%TS%.yml"

cls
echo ==================================================
echo Installing modern stable TensorRT stack, CUDA 12
echo Environment: %ENV_NAME%
echo Python:
where python
python --version
echo ==================================================
echo.

echo [1/7] Backing up pip package list...
python -m pip freeze > "%BACKUP_PIP%"
if errorlevel 1 echo WARNING: pip freeze backup failed.

echo [2/7] Backing up conda environment...
echo       %BACKUP_CONDA%
REM IMPORTANT: conda is a batch file on Windows. It must be called with CALL.
call conda env export -n "%ENV_NAME%" --no-builds > "%BACKUP_CONDA%"
if errorlevel 1 echo WARNING: conda env export failed, continuing because pip freeze backup may be enough.

echo.
echo [3/7] Removing existing ORT/TensorRT packages...
python -m pip uninstall -y onnxruntime onnxruntime-gpu onnxruntime-directml ^
  tensorrt tensorrt-cu12 tensorrt-cu12-bindings tensorrt-cu12-libs ^
  tensorrt-cu13 tensorrt-cu13-bindings tensorrt-cu13-libs ^
  tensorrt_lean tensorrt_dispatch tensorrt-lean tensorrt-dispatch ^
  tensorrt-lean-cu12 tensorrt-dispatch-cu12 tensorrt-lean-cu13 tensorrt-dispatch-cu13
if errorlevel 1 goto :fail

echo.
echo [4/7] Upgrading pip/setuptools/wheel/packaging...
python -m pip install --upgrade pip setuptools wheel packaging
if errorlevel 1 goto :fail

echo.
echo [5/7] Installing TensorRT 10.9 CUDA 12 packages from NVIDIA PyPI...
python -m pip install --upgrade --extra-index-url https://pypi.nvidia.com "tensorrt-cu12==10.9.0.34"
if errorlevel 1 goto :fail

echo.
echo [6/7] Installing ONNX Runtime GPU 1.22 with CUDA/cuDNN runtime dependencies...
python -m pip install --upgrade "onnxruntime-gpu[cuda,cudnn]==1.22.0"
if errorlevel 1 goto :fail

echo.
echo [7/7] Final package versions:
python -m pip list | findstr /I "onnxruntime tensorrt cuda cudnn numpy tensorflow tf2onnx tf_keras"

echo.
echo Running TensorRT/ORT smoke test...
python "%SCRIPT_DIR%check_trt_stack.py"
if errorlevel 1 goto :fail

echo.
echo ==================================================
echo DONE. If you saw SMOKE_TEST_OK, run:
echo   run_gui_export_model_trt.bat
echo and choose backend: tensorrt
echo ==================================================
exit /b 0

:activate_conda
echo Activating conda environment: %ENV_NAME%
where conda >nul 2>nul
if %ERRORLEVEL%==0 (
  call conda activate "%ENV_NAME%"
  exit /b %ERRORLEVEL%
)
if exist "%USERPROFILE%\miniconda3\Scripts\activate.bat" (
  call "%USERPROFILE%\miniconda3\Scripts\activate.bat" "%ENV_NAME%"
  exit /b %ERRORLEVEL%
)
echo ERROR: Could not find conda. Open an Anaconda/Miniconda Prompt and run this script again.
exit /b 1

:fail
echo.
echo ==================================================
echo INSTALL FAILED at the step shown above.
echo Backups were written to:
echo   %BACKUP_DIR%
echo You can paste the last 30 lines of output back here.
echo ==================================================
exit /b 1
