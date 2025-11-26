@echo off
REM Complete build script for Retro ML Trainer
REM This script builds the executable and creates the installer

echo ============================================================
echo Retro ML Trainer - Complete Build Script
echo ============================================================
echo.

REM Step 1: Build executable
echo [1/2] Building executable with PyInstaller...
echo.
python build_scripts\build_executable.py
if errorlevel 1 (
    echo.
    echo ERROR: Executable build failed!
    pause
    exit /b 1
)

echo.
echo ============================================================
echo [2/2] Creating Windows installer...
echo ============================================================
echo.

REM Check if Inno Setup is installed
where iscc >nul 2>nul
if errorlevel 1 (
    echo ERROR: Inno Setup not found!
    echo.
    echo Please install Inno Setup from: https://jrsoftware.org/isdl.php
    echo After installation, add it to your PATH or run build_installer.bat manually
    echo.
    pause
    exit /b 1
)

REM Step 2: Build installer
iscc build_scripts\installer.iss
if errorlevel 1 (
    echo.
    echo ERROR: Installer creation failed!
    pause
    exit /b 1
)

echo.
echo ============================================================
echo BUILD COMPLETE!
echo ============================================================
echo.
echo Executable: dist\RetroMLTrainer\
echo Installer:  installer_output\RetroMLTrainer-Setup-1.0.0.exe
echo.
echo You can now:
echo 1. Test the executable in dist\RetroMLTrainer\
echo 2. Distribute the installer from installer_output\
echo.
pause

