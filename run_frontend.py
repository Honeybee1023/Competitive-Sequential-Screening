#!/usr/bin/env python3
"""
Launch script for Competitive Screening Explorer frontend.

This script:
1. Checks Python version
2. Verifies dependencies
3. Launches Streamlit app
4. Opens browser automatically
"""

import sys
import subprocess
import os
from pathlib import Path

def check_python_version():
    """Ensure Python 3.8+"""
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required")
        print(f"   Current version: {sys.version}")
        sys.exit(1)
    print(f"✓ Python version: {sys.version.split()[0]}")

def check_dependencies():
    """Check if required packages are installed."""
    required = [
        'streamlit',
        'numpy',
        'scipy',
        'matplotlib'
    ]

    missing = []
    for package in required:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            missing.append(package)
            print(f"❌ {package}")

    if missing:
        print(f"\n❌ Missing dependencies: {', '.join(missing)}")
        print("\nInstall with:")
        print(f"  pip3 install {' '.join(missing)}")
        sys.exit(1)

def launch_app():
    """Launch Streamlit app."""
    app_path = Path(__file__).parent / "competitive_screening" / "frontend" / "app.py"

    if not app_path.exists():
        print(f"❌ Error: App not found at {app_path}")
        sys.exit(1)

    print(f"\n🚀 Launching Competitive Screening Explorer...")
    print(f"   App location: {app_path}")
    print(f"\n{'='*70}")
    print("   The app will open in your browser automatically.")
    print("   If not, go to: http://localhost:8501")
    print(f"{'='*70}\n")

    # Launch Streamlit
    try:
        subprocess.run([
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(app_path),
            "--server.headless=false",
            "--browser.gatherUsageStats=false"
        ], check=True)
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down app...")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error launching app: {e}")
        sys.exit(1)

def main():
    print("="*70)
    print("  Competitive Sequential Screening - Interactive Explorer")
    print("="*70)
    print()

    print("Checking requirements...")
    check_python_version()
    check_dependencies()

    print("\n✅ All requirements satisfied!\n")

    launch_app()

if __name__ == "__main__":
    main()
