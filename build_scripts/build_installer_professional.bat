@echo off
REM ==================================================================================
REM Build Professional Modern Installer for Retro ML Trainer
REM ==================================================================================
REM This script:
REM   1. Generates professional installer graphics
REM   2. Builds the executable with PyInstaller
REM   3. Creates a modern Windows installer with Inno Setup
REM ==================================================================================

setlocal enabledelayedexpansion

echo ==================================================================================
echo RETRO ML TRAINER - PROFESSIONAL INSTALLER BUILD
echo ==================================================================================
echo.

REM Change to build_scripts directory
cd /d "%~dp0"

REM Check for Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found! Please install Python 3.8+ and add to PATH.
    pause
    exit /b 1
)

REM Check for Inno Setup
where iscc >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Inno Setup not found! Please install from: https://jrsoftware.org/isdl.php
    echo.
    echo After installation, add Inno Setup to your PATH or run this script from the Inno Setup directory.
    pause
    exit /b 1
)

REM ==================================================================================
REM Step 1: Generate Professional Graphics
REM ==================================================================================
echo.
echo [STEP 1/3] Generating professional installer graphics...
echo ==================================================================================
python create_professional_installer_images.py
if errorlevel 1 (
    echo [ERROR] Failed to generate installer images!
    pause
    exit /b 1
)
echo.
echo ✓ Professional graphics generated successfully!

REM ==================================================================================
REM Step 2: Build Executable (if needed)
REM ==================================================================================
echo.
echo [STEP 2/3] Checking for executable...
echo ==================================================================================

if not exist "..\dist\RetroMLTrainer\RetroMLTrainer.exe" (
    echo Executable not found. Building with PyInstaller...
    echo.
    cd ..
    python build_scripts\build_minimal_executable.py
    if errorlevel 1 (
        echo [ERROR] Failed to build executable!
        pause
        exit /b 1
    )
    cd build_scripts
    echo.
    echo ✓ Executable built successfully!
) else (
    echo ✓ Executable found: ..\dist\RetroMLTrainer\RetroMLTrainer.exe
)

REM ==================================================================================
REM Step 3: Build Installer
REM ==================================================================================
echo.
echo [STEP 3/3] Building professional installer with Inno Setup...
echo ==================================================================================
echo.

iscc installer_professional.iss
if errorlevel 1 (
    echo.
    echo [ERROR] Installer build failed!
    pause
    exit /b 1
)

echo.
echo ==================================================================================
echo ✓ SUCCESS! Professional Installer Built!
echo ==================================================================================
echo.
echo Installer location:
echo   ..\installer_output\RetroMLTrainer-Setup-v1.0.0.exe
echo.
echo File size:
for %%A in ("..\installer_output\RetroMLTrainer-Setup-v1.0.0.exe") do echo   %%~zA bytes (%%~zAk KB)
echo.
echo Next steps:
echo   1. Test the installer on this machine
echo   2. Test on a clean Windows 10/11 VM
echo   3. Verify GPU detection works
echo   4. Verify first-run wizard appears
echo   5. Check start menu and desktop shortcuts
echo.
echo ==================================================================================

pause
