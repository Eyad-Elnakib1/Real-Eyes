# ──────────────────────────────────────────────────────────────────────────────
# RealEyes Environment Setup Script for Windows (PowerShell)
# ──────────────────────────────────────────────────────────────────────────────

$ErrorActionPreference = "Stop"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "            RealEyes Environment Verification & Setup        " -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# 1. Check Python
Write-Host "[1/5] Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVer = py --version 2>&1
    Write-Host "  Found: $pythonVer" -ForegroundColor Green
} catch {
    Write-Host "  ERROR: Python (py) is not installed or not found on PATH." -ForegroundColor Red
    Exit 1
}

# 2. Check Node.js
Write-Host "[2/5] Checking Node.js installation..." -ForegroundColor Yellow
try {
    $nodeVer = node --version 2>&1
    Write-Host "  Found Node.js: $nodeVer" -ForegroundColor Green
} catch {
    Write-Host "  ERROR: Node.js is not installed or not found on PATH." -ForegroundColor Red
    Exit 1
}

# 3. Environment Variables (.env)
Write-Host "[3/5] Checking .env configuration..." -ForegroundColor Yellow
$rootPath = Split-Path -Parent $PSScriptRoot
$envPath = Join-Path $rootPath ".env"
$envExamplePath = Join-Path $rootPath ".env.example"

if (-not (Test-Path $envPath)) {
    if (Test-Path $envExamplePath) {
        Copy-Item $envExamplePath $envPath
        Write-Host "  Created .env from .env.example" -ForegroundColor Green
    } else {
        Write-Host "  WARNING: .env.example not found!" -ForegroundColor Red
    }
} else {
    Write-Host "  .env file already exists." -ForegroundColor Green
}

# 4. Backend Dependencies
Write-Host "[4/5] Installing Backend Python dependencies..." -ForegroundColor Yellow
$backendPath = Join-Path $rootPath "backend"
Push-Location $backendPath
try {
    py -m pip install --upgrade pip | Out-Null
    py -m pip install -r requirements.txt
    Write-Host "  Python requirements installed successfully." -ForegroundColor Green
    
    Write-Host "  Installing spaCy model (en_core_web_sm)..." -ForegroundColor Yellow
    py -m spacy download en_core_web_sm | Out-Null
    Write-Host "  spaCy model ready." -ForegroundColor Green
} catch {
    Write-Host "  ERROR: Failed to install Python dependencies." -ForegroundColor Red
} finally {
    Pop-Location
}

# 5. Frontend Dependencies
Write-Host "[5/5] Installing Frontend Node dependencies..." -ForegroundColor Yellow
$frontendPath = Join-Path $rootPath "frontend"
Push-Location $frontendPath
try {
    npm install
    Write-Host "  Frontend npm packages installed successfully." -ForegroundColor Green
} catch {
    Write-Host "  ERROR: Failed to install npm packages." -ForegroundColor Red
} finally {
    Pop-Location
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "                Setup Complete! Ready to Run.               " -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Start Backend : cd backend; py server.py" -ForegroundColor Gray
Write-Host "Start Frontend: cd frontend; npm run dev" -ForegroundColor Gray
Write-Host ""
