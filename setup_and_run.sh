#!/bin/bash
# AgenteIA - Setup & Run Script
# Usage: sudo ./setup_and_run.sh

set -e

echo "=========================================="
echo "🚀 AgenteIA Setup & Run"
echo "=========================================="

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "❌ Please run with sudo: sudo $0"
    exit 1
fi

# Install system dependencies
echo "[1/5] 📦 Installing system dependencies..."
apt-get update -qq
apt-get install -y -qq python3.12-venv python3-pip > /dev/null 2>&1
echo "✅ System dependencies installed"

# Create virtual environment
echo "[2/5] 🐍 Creating virtual environment..."
python3.12 -m venv venv
source venv/bin/activate
echo "✅ Virtual environment created"

# Upgrade pip
echo "[3/5] ⬆️ Upgrading pip..."
pip install --upgrade pip -q
echo "✅ pip upgraded"

# Install Python dependencies
echo "[4/5] 📚 Installing Python dependencies..."
pip install -r requirements_PINNED.txt -q
pip install slowapi -q
echo "✅ Dependencies installed"

# Run the server
echo "[5/5] 🚀 Starting AgenteIA server..."
echo "=========================================="
echo "🌐 Server running at: http://localhost:8002"
echo "📖 API docs at: http://localhost:8002/docs"
echo "🛑 Press Ctrl+C to stop"
echo "=========================================="

python app_gemini_server.py
