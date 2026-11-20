@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo Запускаю сайт... Через полминуты откройте в браузере http://127.0.0.1:8000
echo Не закрывайте это окно, пока пользуетесь сайтом.
start "" cmd /c "timeout /t 25 >nul & start http://127.0.0.1:8000"
venv\Scripts\python.exe app.py
pause
