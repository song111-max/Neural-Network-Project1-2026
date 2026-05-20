@echo off
cd /d "%~dp0"
echo [1/2] Training CNN (5 epochs, may take longer than MLP)...
python test_train_cnn.py
if errorlevel 1 exit /b 1
echo.
echo [2/2] Evaluating on test set...
python test_model_cnn.py
echo Done. See results\part_b_*
