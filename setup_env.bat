# Frequency-Spatial Vision Mamba Dual Watermarking
# ===================================================
# One-click environment setup for Windows (PowerShell)
#
# Usage:
#   Right-click → "Run with PowerShell"
#   OR: .\setup_env.bat

@echo off
echo ============================================================
echo  Frequency-Spatial Vision Mamba — Environment Setup
echo ============================================================

REM Activate venv
call venv\Scripts\activate.bat

REM Install dependencies
echo Installing dependencies...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo.
echo ============================================================
echo  Setup complete! Activate with:
echo    venv\Scripts\activate
echo.
echo  Then train with:
echo    python -m src.training.train
echo.
echo  Evaluate with:
echo    python -m src.training.evaluate --checkpoint checkpoints/best.pth
echo ============================================================
