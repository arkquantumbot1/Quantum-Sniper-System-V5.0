@echo off
echo ?? ??????????GitHub...
echo ========================================
echo.

echo ??1: ??????...
del nul 2>nul
del NUL 2>nul

echo ??2: ??.gitignore...
if not exist .gitignore (
    echo # ???? > .gitignore
    echo nul >> .gitignore
    echo NUL >> .gitignore
    echo *.log >> .gitignore
    echo *.tmp >> .gitignore
    echo *.temp >> .gitignore
    echo __pycache__/ >> .gitignore
    echo *.pyc >> .gitignore
    echo *.pyo >> .gitignore
    echo *.pyd >> .gitignore
    echo .Python >> .gitignore
    echo env/ >> .gitignore
    echo venv/ >> .gitignore
    echo .venv/ >> .gitignore
    echo .idea/ >> .gitignore
    echo .vscode/ >> .gitignore
    echo .DS_Store >> .gitignore
    echo Thumbs.db >> .gitignore
    echo. >> .gitignore
    echo # ??????? >> .gitignore
    echo quantum_sniper_*.log >> .gitignore
)

echo ??3: ?????Git...
git add --verbose .
if %errorlevel% neq 0 (
    echo ??????????...
    git add src/
    git add requirements.txt
    git add production.yaml
    git add config.yaml
    git add README.md
    git add *.py
    git add *.md
    git add *.yaml
    git add *.yml
    git add .gitignore
    git add deploy/
    git add docker/
)

echo.
echo ??4: ????...
git commit -m "feat: ????????V5.0 - ?????"
if %errorlevel% neq 0 (
    echo ? ??????
    pause
    exit /b 1
)
echo ? ??????

echo.
echo ??5: ???GitHub...
git push -u origin main --force
if %errorlevel% neq 0 (
    echo ? ????
    pause
    exit /b 1
)

echo.
echo ?? GitHub???????
echo ??: https://github.com/arkquantumbot1/Quantum-Sniper-System-V5.0
echo.
pause
