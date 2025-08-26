@echo off
:: Check for administrative privileges
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo This script requires administrative privileges. Please run it as an administrator.
    pause
    exit /b
)

echo Checking for Windows Subsystem for Linux (WSL)...

:: Check if WSL is installed
wsl --version >nul 2>&1
if %errorlevel% neq 0 (
    echo WSL is not installed on your system.
    echo Installing WSL...
    dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
    echo WSL has been installed. Please restart your computer to complete the installation.
    exit /b
) else (
    echo WSL is already installed on your system.


)