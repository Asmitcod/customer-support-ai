@echo off
title AI Customer Support System
color 0A
cls

echo ========================================
echo  AI-Powered Customer Support System
echo  LangGraph + LangChain + Ollama + RAG
echo ========================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found!
    pause
    exit /b 1
)

chcp 65001 > nul
set PYTHONIOENCODING=utf-8

echo Starting the demo...
echo.
echo [NOTE] When you see "Approve request? (y/n):" on Query 4,
echo        type  y  and press ENTER to approve the refund.
echo.
echo ----------------------------------------
echo.

.venv\Scripts\python main.py

echo.
echo ========================================
echo  Demo complete!
echo ========================================
pause
