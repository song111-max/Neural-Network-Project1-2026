@echo off
REM Part A: train MLP baseline, then evaluate on test set
cd /d "%~dp0"
echo [1/2] Training MLP (5 epochs, may take 1-2 hours)...
python test_train.py
if errorlevel 1 exit /b 1
echo.
echo [2/2] Evaluating on MNIST test set...
python test_model.py
echo Done. Check results\ folder for learning curve and summary.
