@echo off
REM ============================================================
REM  InsightHub — setup.bat
REM  Windows one-click environment setup script
REM  Run this ONCE to set up your full development environment
REM  Usage: Double-click setup.bat  OR  run in VS Code terminal
REM ============================================================

echo.
echo  ==========================================
echo   InsightHub — Environment Setup (Windows)
echo  ==========================================
echo.

REM ── Step 1: Check Python version ────────────────────────────
echo [1/7] Checking Python version...
python --version 2>NUL
if errorlevel 1 (
    echo  ERROR: Python not found.
    echo  Please install Python 3.10 or 3.11 from https://python.org
    echo  Make sure to check "Add Python to PATH" during install.
    pause
    exit /b 1
)

REM ── Step 2: Create virtual environment ──────────────────────
echo.
echo [2/7] Creating virtual environment (venv)...
if exist venv (
    echo  Virtual environment already exists — skipping creation.
) else (
    python -m venv venv
    echo  Virtual environment created successfully.
)

REM ── Step 3: Activate virtual environment ────────────────────
echo.
echo [3/7] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo  ERROR: Could not activate virtual environment.
    pause
    exit /b 1
)
echo  Activated: venv

REM ── Step 4: Upgrade pip ──────────────────────────────────────
echo.
echo [4/7] Upgrading pip...
python -m pip install --upgrade pip --quiet
echo  pip upgraded.

REM ── Step 5: Install PyTorch CPU first ───────────────────────
echo.
echo [5/7] Installing PyTorch (CPU version)...
echo  This may take a few minutes — PyTorch is a large package.
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cpu --quiet
echo  PyTorch installed.

REM ── Step 6: Install all requirements ────────────────────────
echo.
echo [6/7] Installing all dependencies from requirements.txt...
echo  This will take 5-10 minutes on first run. Please wait...
pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo.
    echo  WARNING: Some packages may have failed. Check output above.
    echo  Try running:  pip install -r requirements.txt
    echo  to see detailed error messages.
)
echo  Dependencies installed.

REM ── Step 7: Install Playwright browsers ─────────────────────
echo.
echo [7/7] Installing Playwright browser (Chromium)...
playwright install chromium --quiet
echo  Playwright browser installed.

REM ── Step 8: Create .env file if not exists ──────────────────
echo.
echo [Setup] Creating .env file from template...
if not exist .env (
    copy .env.example .env >NUL
    echo  .env file created. Please open it and add your API keys.
) else (
    echo  .env file already exists — skipping.
)

REM ── Step 9: Create folder structure ─────────────────────────
echo.
echo [Setup] Creating project folder structure...
if not exist insighthub\config       mkdir insighthub\config
if not exist insighthub\ingestion    mkdir insighthub\ingestion
if not exist insighthub\knowledge    mkdir insighthub\knowledge
if not exist insighthub\agents       mkdir insighthub\agents
if not exist insighthub\interface    mkdir insighthub\interface
if not exist insighthub\evaluation   mkdir insighthub\evaluation
if not exist data\uploads            mkdir data\uploads
if not exist vector_store            mkdir vector_store
if not exist exports                 mkdir exports

REM Create __init__.py files
type NUL > insighthub\__init__.py
type NUL > insighthub\config\__init__.py
type NUL > insighthub\ingestion\__init__.py
type NUL > insighthub\knowledge\__init__.py
type NUL > insighthub\agents\__init__.py
type NUL > insighthub\interface\__init__.py
type NUL > insighthub\evaluation\__init__.py
echo  Folder structure created.

REM ── Step 10: Validate config ─────────────────────────────────
echo.
echo [Setup] Validating configuration...
python insighthub\config\settings.py

REM ── Done ─────────────────────────────────────────────────────
echo.
echo  ==========================================
echo   Setup Complete!
echo  ==========================================
echo.
echo  Next steps:
echo   1. Open .env in VS Code
echo   2. Add your HUGGINGFACE_API_TOKEN
echo      Get free token at: https://huggingface.co/settings/tokens
echo   3. Optionally add COHERE_API_KEY (free reranking)
echo   4. Run: python insighthub\config\settings.py
echo      to verify your configuration
echo.
echo  To activate venv in future VS Code sessions:
echo   Terminal ^> New Terminal, then type:
echo   venv\Scripts\activate
echo.
pause