#!/usr/bin/env python3
"""
Comprehensive setup script for Alzheimer's Handwriting Classification Project
"""

import subprocess
import sys
import os
import platform

def run_command(command, description, check=True):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=check, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            return True
        else:
            print(f"⚠️  {description} completed with warnings")
            if result.stderr:
                print(f"   Warning: {result.stderr}")
            return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} is not supported")
        print("   Please install Python 3.8 or higher")
        return False
    else:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True

def create_virtual_environment():
    """Create virtual environment if it doesn't exist"""
    print("📦 Setting up virtual environment...")
    
    if os.path.exists("venv"):
        print("✅ Virtual environment already exists")
        return True
    
    # Create virtual environment
    if not run_command("python -m venv venv", "Creating virtual environment"):
        return False
    
    print("✅ Virtual environment created successfully")
    return True

def install_packages():
    """Install required packages"""
    print("📚 Installing required packages...")
    
    # Determine pip command based on platform
    if platform.system() == "Windows":
        pip_cmd = "venv\\Scripts\\pip"
    else:
        pip_cmd = "venv/bin/pip"
    
    # Upgrade pip first
    run_command(f"{pip_cmd} install --upgrade pip", "Upgrading pip", check=False)
    
    # Install requirements
    if not run_command(f"{pip_cmd} install -r requirements.txt", "Installing required packages"):
        return False
    
    print("✅ All packages installed successfully")
    return True

def create_directories():
    """Create necessary directories"""
    print("📁 Creating project directories...")
    
    directories = [
        "models",
        "static",
        "static/css",
        "static/js",
        "static/images",
        "templates",
        "src",
        "data",
        "notebooks",
        "docs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"   ✅ {directory}/")
    
    return True

def check_data_file():
    """Check if data file exists"""
    print("📊 Checking data file...")
    
    if os.path.exists("data/data.csv"):
        print("✅ data/data.csv found")
        return True
    else:
        print("⚠️  data/data.csv not found")
        print("   Please place your dataset in the data/ folder")
        print("   The dataset should contain handwriting features with a 'class' column")
        return False

def create_sample_data():
    """Create a sample data file for testing"""
    print("📝 Creating sample data file...")
    
    try:
        import pandas as pd
        import numpy as np
        
        # Create sample data with 451 features (same as expected)
        n_samples = 100
        n_features = 451
        
        # Generate random data
        data = np.random.randn(n_samples, n_features)
        
        # Create feature names
        feature_names = [f"feature_{i:03d}" for i in range(n_features)]
        
        # Create DataFrame
        df = pd.DataFrame(data, columns=feature_names)
        
        # Add ID column
        df.insert(0, 'ID', range(1, n_samples + 1))
        
        # Add class column (randomly assign P or H)
        classes = np.random.choice(['P', 'H'], n_samples, p=[0.5, 0.5])
        df['class'] = classes
        
        # Save to CSV
        df.to_csv('data/data.csv', index=False)
        
        print("✅ Sample data file created: data/data.csv")
        print(f"   • Samples: {n_samples}")
        print(f"   • Features: {n_features}")
        print("   • Note: This is sample data for testing only")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating sample data: {e}")
        return False

def run_tests():
    """Run basic tests to verify installation"""
    print("🧪 Running basic tests...")
    
    try:
        # Test imports
        import numpy as np
        import pandas as pd
        import sklearn
        import matplotlib
        import seaborn
        import flask
        print("✅ All required packages imported successfully")
        
        # Test data processing
        from src.data_processor import DataProcessor
        print("✅ Data processor module loaded")
        
        # Test model utilities
        from src.model_utils import AlzheimerModelLoader
        print("✅ Model utilities loaded")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def print_next_steps():
    """Print next steps for the user"""
    print("\n" + "=" * 80)
    print("🎉 SETUP COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    
    print("\n📋 NEXT STEPS:")
    print("1. Activate virtual environment:")
    if platform.system() == "Windows":
        print("   venv\\Scripts\\activate")
    else:
        print("   source venv/bin/activate")
    
    print("\n2. Train the model:")
    print("   python train_model.py")
    
    print("\n3. Start the web interface:")
    print("   python app.py")
    print("   Then open: http://localhost:5000")
    
    print("\n4. Open Jupyter Lab for analysis:")
    print("   jupyter lab")
    print("   Then open: notebooks/01eda.ipynb")
    
    print("\n5. Run tests:")
    print("   python test_model.py")
    
    print("\n📚 PROJECT STRUCTURE:")
    print("   📁 data/           - Dataset files")
    print("   📁 models/         - Trained models and metadata")
    print("   📁 notebooks/      - Jupyter notebooks for analysis")
    print("   📁 src/            - Source code utilities")
    print("   📁 templates/      - Web interface templates")
    print("   📁 static/         - Web interface assets")
    print("   📄 app.py          - Flask web application")
    print("   📄 train_model.py  - Model training script")
    print("   📄 test_model.py   - Test script")
    
    print("\n🎯 READY TO GO!")
    print("   Your Alzheimer's handwriting classification project is ready!")

def main():
    """Main setup function"""
    print("🚀 Setting up Alzheimer's Handwriting Classification Project")
    print("=" * 80)
    
    # Step 1: Check Python version
    if not check_python_version():
        return False
    
    # Step 2: Create virtual environment
    if not create_virtual_environment():
        return False
    
    # Step 3: Create directories
    if not create_directories():
        return False
    
    # Step 4: Install packages
    if not install_packages():
        return False
    
    # Step 5: Check data file
    if not check_data_file():
        print("   Creating sample data for testing...")
        if not create_sample_data():
            print("   ⚠️  Could not create sample data")
    
    # Step 6: Run tests
    if not run_tests():
        print("   ⚠️  Some tests failed, but setup may still work")
    
    # Step 7: Print next steps
    print_next_steps()
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Setup failed. Please check the errors above.")
        sys.exit(1)
    else:
        print("\n✅ Setup completed successfully!")

