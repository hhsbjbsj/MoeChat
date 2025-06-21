@echo off
set "PYTHON_PATH=/home/keruis/extra/MoeChat-main/pp/bin/python"

set "SCRIPT_1=script1.py"
set "TITLE_1=PythonScript1"

set "SCRIPT_2=script2.py"
set "TITLE_2=PythonScript2"

set "SCRIPT_3=script3.py"
set "TITLE_3=PythonScript3"

cd /d "%~dp0"

call :ToggleScript "%TITLE_1%" "%SCRIPT_1%"
call :ToggleScript "%TITLE_2%" "%SCRIPT_2%"
call :ToggleScript "%TITLE_3%" "%SCRIPT_3%"

echo Done.
pause
exit /b

:ToggleScript
tasklist /v /fi "imagename eq python.exe" | findstr /i "%~1" >nul
if %errorlevel%==0 (
    echo Found %~1 running. Killing...
    taskkill /f /fi "imagename eq python.exe" /fi "windowtitle eq %~1"
) else (
    echo Starting %~1...
    start "%~1" "%PYTHON_PATH%" "%~2"
)
exit /b