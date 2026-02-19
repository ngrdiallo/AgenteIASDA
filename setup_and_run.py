#!/usr/bin/env python3
# Quick install and run script for AgenteIA

import subprocess
import sys
import os

def run_cmd(cmd, description):
    print(f"\n[*] {description}...")
    try:
        subprocess.run(cmd, shell=True, check=True)
        print(f"[OK] {description}")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {description} failed: {e}")
        sys.exit(1)

def main():
    print("\n" + "="*60)
    print("AgenteIA - Setup & Run")
    print("="*60)
    
    # Check Python version
    if sys.version_info < (3, 9):
        print("[ERROR] Python 3.9+ required")
        sys.exit(1)
    
    # Create venv
    if not os.path.exists("venv"):
        run_cmd("python -m venv venv", "Create virtual environment")
    
    # Activate venv and install deps
    if sys.platform == "win32":
        activate = "venv\\Scripts\\activate.bat"
        pip = "venv\\Scripts\\pip"
    else:
        activate = "source venv/bin/activate"
        pip = "venv/bin/pip"
    
    run_cmd(f"{pip} install --upgrade pip", "Upgrade pip")
    run_cmd(f"{pip} install -r requirements_PINNED.txt", "Install dependencies")
    
    # Check .env
    if not os.path.exists(".env"):
        print("\n[!] .env file not found!")
        print("[!] Copy .env.example to .env and add API keys:")
        print("    - GROQ_API_KEY")
        print("    - MISTRAL_API_KEY")
        print("    - HUGGINGFACE_API_KEY (optional)")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("[OK] Setup complete!")
    print("[*] Starting server on http://localhost:8002")
    print("="*60 + "\n")
    
    # Run server
    if sys.platform == "win32":
        os.system("venv\\Scripts\\python app_gemini_server.py")
    else:
        os.system("venv/bin/python app_gemini_server.py")

if __name__ == "__main__":
    main()
