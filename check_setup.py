"""
Quick start script to check and install dependencies
Run this before training the model
"""
import subprocess
import sys

def install_package(package):
    """Install a package using pip"""
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def check_installations():
    """Check if required packages are installed"""
    required_packages = {
        'tensorflow': '2.15.0',
        'torch': '2.0.1',
        'sentence-transformers': '2.2.2',
        'streamlit': None,
        'pandas': None,
        'numpy': None,
        'scikit-learn': None
    }
    
    print("=== Checking Installations ===\n")
    
    missing_packages = []
    
    for package, version in required_packages.items():
        try:
            if package == 'scikit-learn':
                __import__('sklearn')
                import sklearn
                print(f"✓ {package}: {sklearn.__version__}")
            else:
                module = __import__(package.replace('-', '_'))
                ver = getattr(module, '__version__', 'unknown')
                print(f"✓ {package}: {ver}")
                
                if version and ver != version:
                    print(f"  ⚠ Warning: Expected version {version}, got {ver}")
        except ImportError:
            print(f"✗ {package}: NOT INSTALLED")
            missing_packages.append(f"{package}=={version}" if version else package)
    
    print("\n" + "=" * 40)
    
    if missing_packages:
        print(f"\n⚠ Missing packages: {', '.join(missing_packages)}")
        print("\nInstall with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    else:
        print("\n✓ All required packages are installed!")
        return True

def check_gpu():
    """Check GPU availability"""
    print("\n=== Checking GPU ===\n")
    
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        
        if gpus:
            print(f"✓ Found {len(gpus)} GPU(s):")
            for gpu in gpus:
                print(f"  - {gpu.name}")
            print("\n✓ GPU is available for training!")
        else:
            print("✗ No GPU found")
            print("  Training will use CPU (slower)")
            
        print(f"\nTensorFlow built with CUDA: {tf.test.is_built_with_cuda()}")
        
    except ImportError:
        print("✗ TensorFlow not installed - cannot check GPU")
    except Exception as e:
        print(f"✗ Error checking GPU: {e}")

def main():
    print("=" * 70)
    print("  Research Paper Recommendation and Subject Area Prediction")
    print("  Using Sentence-BERT and Multi-Label MLP Classification")
    print("  Pre-flight Check")
    print("=" * 70 + "\n")
    
    # Check Python version
    print(f"Python version: {sys.version}\n")
    
    # Check installations
    all_installed = check_installations()
    
    # Check GPU
    check_gpu()
    
    print("\n" + "=" * 50)
    if all_installed:
        print("\n✓ System ready! You can now:")
        print("  1. Run the notebook to train the model")
        print("  2. Then run: streamlit run app.py")
    else:
        print("\n⚠ Please install missing packages first:")
        print("  pip install -r requirements.txt")
    print("\n" + "=" * 50)

if __name__ == "__main__":
    main()
