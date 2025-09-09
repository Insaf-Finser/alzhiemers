#!/usr/bin/env python3
"""
Setup script for Alzheimer's Handwriting Classification Project
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def main():
    print("🚀 Setting up Alzheimer's Handwriting Classification Project")
    print("=" * 60)
    
    # Check if virtual environment exists
    if not os.path.exists("venv"):
        print("❌ Virtual environment not found. Please create it first:")
        print("   python -m venv venv")
        return False
    
    # Activate virtual environment and install packages
    if sys.platform == "win32":
        activate_cmd = "venv\\Scripts\\activate"
        pip_cmd = "venv\\Scripts\\pip"
    else:
        activate_cmd = "source venv/bin/activate"
        pip_cmd = "venv/bin/pip"
    
    # Install requirements
    if not run_command(f"{pip_cmd} install -r requirements.txt", "Installing required packages"):
        return False
    
    # Check if data file exists
    if not os.path.exists("data/data.csv"):
        print("⚠️  Warning: data/data.csv not found")
        print("   Please place your dataset in the data/ folder")
        print("   A sample dataset has been created for testing")
    
    print("\n" + "=" * 60)
    print("✅ Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Activate virtual environment:")
    if sys.platform == "win32":
        print("   venv\\Scripts\\activate")
    else:
        print("   source venv/bin/activate")
    print("2. Start Jupyter Lab:")
    print("   jupyter lab")
    print("3. Open notebooks/01eda.ipynb")
    print("\n🎯 You're ready to run the Alzheimer's classification notebook!")

if __name__ == "__main__":
    main()
