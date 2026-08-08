@echo off
if not exist .venv (
    py -3 -m venv .venv
)
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
echo.
echo CP-0002 setup complete.
echo Run test.bat
pause
