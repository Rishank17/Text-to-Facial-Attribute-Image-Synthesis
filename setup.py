"""
Setup script for AttriDiffuser project
"""
import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully!")
        return True
    except Exception as e:
        print(f"❌ Error installing packages: {e}")
        return False

def check_gpu():
    """Check if CUDA is available"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA Version: {torch.version.cuda}")
            return True
        else:
            print("⚠️ No GPU detected. Will use CPU (slower generation)")
            return False
    except ImportError:
        print("⚠️ PyTorch not installed yet")
        return False

def create_directories():
    """Create necessary directories"""
    dirs = ["dataset", "generated_faces", "models"]
    for dir_name in dirs:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print(f"✅ Created directory: {dir_name}")

def main():
    print("=" * 50)
    print("AttriDiffuser Setup")
    print("=" * 50)
    
    # Create directories
    print("\n1. Creating directories...")
    create_directories()
    
    # Install requirements
    print("\n2. Installing requirements...")
    if install_requirements():
        print("\n3. Checking GPU availability...")
        check_gpu()
        
        print("\n" + "=" * 50)
        print("Setup complete! 🎉")
        print("=" * 50)
        print("\nTo run the application:")
        print("  streamlit run app.py")
        print("\n")
    else:
        print("\n❌ Setup failed. Please check the errors above.")

if __name__ == "__main__":
    main()
