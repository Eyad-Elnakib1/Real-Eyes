#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# RealEyes Environment Setup Script for Linux / macOS
# ──────────────────────────────────────────────────────────────────────────────

set -e

echo "============================================================"
echo "            RealEyes Environment Verification & Setup        "
echo "============================================================"
echo ""

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$( dirname "$SCRIPT_DIR" )"

# 1. Check Python
echo "[1/5] Checking Python installation..."
if command -v python3 &>/dev/null; then
    PYTHON_BIN="python3"
elif command -v python &>/dev/null; then
    PYTHON_BIN="python"
else
    echo "  ERROR: Python is not installed."
    exit 1
fi
echo "  Found Python: $($PYTHON_BIN --version)"

# 2. Check Node.js
echo "[2/5] Checking Node.js installation..."
if command -v node &>/dev/null; then
    echo "  Found Node.js: $(node --version)"
else
    echo "  ERROR: Node.js is not installed."
    exit 1
fi

# 3. Environment Variables
echo "[3/5] Checking .env configuration..."
if [ ! -f "$ROOT_DIR/.env" ]; then
    if [ -f "$ROOT_DIR/.env.example" ]; then
        cp "$ROOT_DIR/.env.example" "$ROOT_DIR/.env"
        echo "  Created .env from .env.example"
    fi
else
    echo "  .env file already exists."
fi

# 4. Backend Dependencies
echo "[4/5] Installing Backend Python dependencies..."
cd "$ROOT_DIR/backend"
$PYTHON_BIN -m pip install --upgrade pip
$PYTHON_BIN -m pip install -r requirements.txt
$PYTHON_BIN -m spacy download en_core_web_sm
echo "  Python dependencies ready."

# 5. Frontend Dependencies
echo "[5/5] Installing Frontend Node dependencies..."
cd "$ROOT_DIR/frontend"
npm install
echo "  Frontend dependencies ready."

echo ""
echo "============================================================"
echo "                Setup Complete! Ready to Run.               "
echo "============================================================"
echo "Start Backend : cd backend && python server.py"
echo "Start Frontend: cd frontend && npm run dev"
echo ""

