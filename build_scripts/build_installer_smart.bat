@echo off
REM ==================================================================================
REM Build Professional Installer - Auto-detects Inno Setup
REM ==================================================================================

setlocal enabledelayedexpansion

echo ==================================================================================
echo RETRO ML TRAINER - PROFESSIONAL INSTALLER BUILD
echo ==================================================================================
echo.

REM Change to build_scripts directory
cd /d "%~dp0"

REM ==================================================================================
REM Find Inno Setup (ISCC.exe)
REM ==================================================================================
echo [INFO] Searching for Inno Setup...

set ISCC_PATH=
set SEARCH_PATHS="D:\Inno Setup 6" "C:\Program Files (x86)\Inno Setup 6" "C:\Program Files\Inno Setup 6" "C:\Program Files (x86)\Inno Setup 5" "C:\Inno Setup 6"

for %%P in (%SEARCH_PATHS%) do (
    if exist "%%~P\ISCC.exe" (
        set "ISCC_PATH=%%~P\ISCC.exe"
        echo [INFO] Found Inno Setup: !ISCC_PATH!
        goto :found_inno
    )
)

REM Try to find in PATH
where iscc >nul 2>&1
if %errorlevel% equ 0 (
    for /f "tokens=*" %%i in ('where iscc') do set "ISCC_PATH=%%i"
    echo [INFO] Found Inno Setup in PATH: !ISCC_PATH!
    goto :found_inno
)

echo [ERROR] Inno Setup not found!
echo.
echo Searched locations:
for %%P in (%SEARCH_PATHS%) do echo   - %%~P
echo.
echo Please install Inno Setup from: https://jrsoftware.org/isdl.php
echo Or add ISCC.exe to your PATH
pause
exit /b 1

:found_inno

REM ==================================================================================
REM Check for Python
REM ==================================================================================
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found! Please install Python 3.8+ and add to PATH.
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
echo ✓ Professional graphics generated!

REM ==================================================================================
REM Step 2: Build Executable (if needed)
REM ==================================================================================
echo.
echo [STEP 2/3] Checking for executable...
echo ==================================================================================

if not exist "..\dist\RetroMLTrainer\RetroMLTrainer.exe" (
    echo [INFO] Executable not found. Building with PyInstaller...
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
REM Step 3: Build Professional Installer
REM ==================================================================================
echo.
echo [STEP 3/3] Building professional installer...
echo ==================================================================================
echo Using Inno Setup: !ISCC_PATH!
echo.

"!ISCC_PATH!" installer_professional.iss
if errorlevel 1 (
    echo.
    echo [ERROR] Installer build failed!
    pause
    exit /b 1
)

echo.
echo ==================================================================================
echo ✓✓✓ SUCCESS! PROFESSIONAL INSTALLER READY FOR DISTRIBUTION! ✓✓✓
echo ==================================================================================
echo.
echo Installer location:
echo   ..\installer_output\RetroMLTrainer-Setup-v1.0.0.exe
echo.

REM Get file size
for %%A in ("..\installer_output\RetroMLTrainer-Setup-v1.0.0.exe") do (
    set SIZE=%%~zA
    set /a SIZE_MB=!SIZE! / 1048576
    echo File size: !SIZE! bytes (!SIZE_MB! MB)
)

echo.
echo ==================================================================================
echo READY FOR DISTRIBUTION
echo ==================================================================================
echo.
echo This installer is production-ready with:
echo   ✓ Professional gradient graphics (deep blue + cyan theme)
echo   ✓ Smart GPU detection and system checks
echo   ✓ Modern resizable UI
echo   ✓ Complete first-run wizard
echo   ✓ All 5 critical fixes applied
echo   ✓ Settings tab with ROM installation
echo.
echo Next steps:
echo   1. Test on this machine: ..\installer_output\RetroMLTrainer-Setup-v1.0.0.exe
echo   2. Test on clean Windows VM
echo   3. Distribute to users!
echo.
echo Documentation:
echo   - docs\PROFESSIONAL_INSTALLER_GUIDE.md
echo   - docs\DISTRIBUTION_GUIDE.md
echo   - PROFESSIONAL_INSTALLER_SUMMARY.md
echo.
echo ==================================================================================

pause
