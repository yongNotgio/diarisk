"""
Test script to verify all dependencies are working correctly for Streamlit deployment.
Run this script to check if all required packages can be imported.
"""

import sys
import importlib

# List of required packages
required_packages = [
    'streamlit',
    'pandas', 
    'numpy',
    'sklearn',
    'joblib',
    'plotly',
    'xgboost',
    'lightgbm',
    'catboost'
]

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing package imports...")
    print("=" * 50)
    
    failed_imports = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package} - OK")
        except ImportError as e:
            print(f"❌ {package} - FAILED: {e}")
            failed_imports.append(package)
    
    print("=" * 50)
    
    if failed_imports:
        print(f"❌ Failed to import: {', '.join(failed_imports)}")
        print("Please install missing packages using:")
        print(f"pip install {' '.join(failed_imports)}")
        return False
    else:
        print("✅ All packages imported successfully!")
        return True

def test_model_loading():
    """Test if model files can be loaded"""
    print("\nTesting model file loading...")
    print("=" * 50)
    
    from pathlib import Path
    import joblib
    import json
    
    models_path = Path("models")
    
    if not models_path.exists():
        print("❌ Models directory not found")
        return False
    
    required_files = [
        "best_model_xgboost_gpu.pkl",
        "scaler.pkl",
        "label_encoder.pkl", 
        "feature_names.txt",
        "model_evaluation_results.json"
    ]
    
    for file in required_files:
        file_path = models_path / file
        if file_path.exists():
            print(f"✅ {file} - Found")
            
            # Try to load the file
            try:
                if file.endswith('.pkl'):
                    joblib.load(file_path)
                    print(f"   └─ Loaded successfully")
                elif file.endswith('.json'):
                    with open(file_path, 'r') as f:
                        json.load(f)
                    print(f"   └─ Loaded successfully")
                elif file.endswith('.txt'):
                    with open(file_path, 'r') as f:
                        content = f.read()
                    print(f"   └─ Read successfully ({len(content)} characters)")
            except Exception as e:
                print(f"   └─ ❌ Error loading: {e}")
                return False
        else:
            print(f"❌ {file} - Missing")
            return False
    
    print("✅ All model files loaded successfully!")
    return True

if __name__ == "__main__":
    print("Diabetes Surgery Risk Assessment - Dependency Test")
    print("=" * 60)
    
    imports_ok = test_imports()
    models_ok = test_model_loading()
    
    print("\n" + "=" * 60)
    if imports_ok and models_ok:
        print("🎉 All tests passed! The app should work correctly.")
    else:
        print("❌ Some tests failed. Please fix the issues before deployment.")
        sys.exit(1)
