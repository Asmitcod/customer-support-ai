@echo off
title AI Customer Support Web UI
color 0B
cls

echo ========================================
echo  Launching Customer Support Web Frontend
echo  Streamlit + LangGraph + Ollama
echo ========================================
echo.

if not exist ".venv\Scripts\streamlit.exe" (
    echo [ERROR] Streamlit not found in .venv!
    pause
    exit /b 1
)

chcp 65001 > nul
set PYTHONIOENCODING=utf-8

echo Starting web server...
echo The app will open in your browser automatically!
echo Press Ctrl+C in this terminal when you want to stop it.
echo.

.venv\Scripts\streamlit run app.py
pause
