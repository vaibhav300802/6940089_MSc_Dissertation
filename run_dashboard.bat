@echo off
setlocal
cd /d "%~dp0dashboard"
cd ..
python validate_project.py
if errorlevel 1 exit /b 1
python dashboard\startup_validation.py
if errorlevel 1 exit /b 1
cd dashboard
python -m pip install -r requirements.txt
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
endlocal
