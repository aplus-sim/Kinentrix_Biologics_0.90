@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo(
echo ============================================================
echo   KINENTRIX Biologics - one-time setup
echo ============================================================
echo(

rem ---- find Python: py launcher first, then python on PATH
set "PY="
py -3 --version >nul 2>&1 && set "PY=py -3"
if not defined PY (
    python --version >nul 2>&1 && set "PY=python"
)

rem ---- install Python with winget if missing (user scope, no admin)
if not defined PY (
    echo   Python not found. Trying to install it with winget.
    echo   ^(Installs for this user only, no admin rights. Takes 3-5 minutes.^)
    echo(
    winget --version >nul 2>&1
    if errorlevel 1 goto NOWINGET
    winget install --id Python.Python.3.13 -e --scope user --silent ^
        --accept-source-agreements --accept-package-agreements
    echo(
    echo   Install finished. Looking for Python again.
    set "PY="
    py -3 --version >nul 2>&1 && set "PY=py -3"
    if not defined PY (
        for /d %%D in ("%LOCALAPPDATA%\Programs\Python\Python3*") do (
            if exist "%%D\python.exe" set "PY=%%D\python.exe"
        )
    )
)

if not defined PY goto NOPYTHON

echo   Python  : !PY!
for /f "tokens=*" %%V in ('!PY! --version 2^>^&1') do echo   Version : %%V
echo(

rem ---- virtual environment
if exist ".venv\Scripts\python.exe" (
    echo   .venv already exists. Skipping.
) else (
    echo   [1/2] Creating .venv ...
    !PY! -m venv .venv
    if errorlevel 1 goto VENVFAIL
)

echo   [2/2] Installing packages. Takes 2-5 minutes...
echo(
".venv\Scripts\python.exe" -m pip install --upgrade pip --quiet
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto PIPFAIL

echo(
echo   Checking...
".venv\Scripts\python.exe" -c "import streamlit, sklearn, xgboost, catboost, lightgbm, matplotlib; print('   All packages OK')"
if errorlevel 1 goto PIPFAIL

echo(
echo ============================================================
echo   Setup complete. Double-click run_app.vbs to start the app.
echo ============================================================
echo(
exit /b 0

:NOWINGET
echo(
echo   winget is not available, so Python cannot be installed automatically.
echo   Install Python 3.11 or later from python.org, then run this again.
echo   Turn on "Add python.exe to PATH" during installation.
echo(
pause
exit /b 1

:NOPYTHON
echo(
echo   Python still not found.
echo   Close this window and try again. If it still fails, install
echo   Python 3.11 or later from python.org and turn on
echo   "Add python.exe to PATH".
echo(
pause
exit /b 1

:VENVFAIL
echo(
echo   Could not create .venv. Check write permission for this folder.
echo   ^(Try moving the folder to Desktop or Documents.^)
echo(
pause
exit /b 1

:PIPFAIL
echo(
echo   Package installation failed. See the messages above.
echo   On a company network, ask IT whether pypi.org is reachable.
echo(
pause
exit /b 1
