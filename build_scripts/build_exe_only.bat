@echo off
REM Build only the executable (no installer)
REM Useful for quick testing

echo ============================================================
echo Building Retro ML Trainer Executable
echo ============================================================
echo.

python build_scripts\build_executable.py

if errorlevel 1 (
    echo.
    echo ERROR: Build failed!
    pause
    exit /b 1
)

echo.
echo ============================================================
echo BUILD COMPLETE!
echo ============================================================
echo.
echo Executable location: dist\RetroMLTrainer\RetroMLTrainer.exe
echo.
echo You can now test the application by running:
echo   dist\RetroMLTrainer\RetroMLTrainer.exe
echo.
pause

