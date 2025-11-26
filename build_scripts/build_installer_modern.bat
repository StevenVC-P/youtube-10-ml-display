@echo off
REM Build modern installer for Retro ML Trainer
REM Uses the modernized Inno Setup script with GPU detection and better UX

echo ============================================================
echo Building Modern Retro ML Trainer Installer
echo ============================================================
echo.

REM Check if Inno Setup is installed
where iscc >nul 2>nul
if errorlevel 1 (
    echo ERROR: Inno Setup not found!
    echo.
    echo Please install Inno Setup from: https://jrsoftware.org/isdl.php
    echo After installation, add it to your PATH
    echo.
    pause
    exit /b 1
)

REM Check if installer images exist
if not exist "installer_images\app_icon.ico" (
    echo Installer images not found. Creating placeholder images...
    echo.
    python create_installer_images.py
    if errorlevel 1 (
        echo ERROR: Failed to create installer images
        pause
        exit /b 1
    )
    echo.
)

REM Build the installer
echo Building installer with modern UI...
echo.
iscc installer_modern.iss

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
echo Installer: ..\installer_output\RetroMLTrainer-Setup-v1.0.0.exe
echo.
echo Features:
echo   - Modern dark blue UI theme
echo   - GPU detection before installation
echo   - System requirements check
echo   - Custom wizard images
echo   - Enhanced user experience
echo.
pause
