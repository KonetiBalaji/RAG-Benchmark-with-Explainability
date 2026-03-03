"""Quick system test script to verify installation and configuration."""

import sys
from pathlib import Path

def test_imports():
    """Test if all required packages can be imported."""
    print("Testing imports...")
    required_packages = [
        ("yaml", "PyYAML"),
        ("dotenv", "python-dotenv"),
        ("loguru", "loguru"),
        ("openai", "openai"),
        ("chromadb", "chromadb"),
        ("langchain", "langchain"),
        ("streamlit", "streamlit"),
    ]
    
    failed = []
    for module_name, package_name in required_packages:
        try:
            __import__(module_name)
            print(f"  [OK] {package_name}")
        except ImportError:
            print(f"  [FAIL] {package_name}")
            failed.append(package_name)
    
    if failed:
        print(f"\n[ERROR] Missing packages: {', '.join(failed)}")
        print("Run: pip install -r requirements.txt")
        return False
    else:
        print("[SUCCESS] All packages installed\n")
        return True


def test_config():
    """Test configuration loading."""
    print("Testing configuration...")
    try:
        from src.utils.config_loader import ConfigLoader
        config = ConfigLoader()
        print(f"  [OK] Config file loaded: {config.config_path}")
        
        # Check essential config sections
        required_sections = ["dataset", "llm", "rag_configs", "evaluation"]
        for section in required_sections:
            if section in config.config:
                print(f"  [OK] Section '{section}' found")
            else:
                print(f"  [FAIL] Section '{section}' missing")
                return False
        
        print("[SUCCESS] Configuration valid\n")
        return True
    except Exception as e:
        print(f"[ERROR] Configuration error: {e}\n")
        return False


def test_env():
    """Test environment variables."""
    print("Testing environment variables...")
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    required_vars = {
        "OPENAI_API_KEY": "Required for LLM and embeddings",
        "COHERE_API_KEY": "Required for reranking (Config 3)"
    }
    
    missing = []
    for var, description in required_vars.items():
        value = os.getenv(var)
        if value:
            masked = value[:8] + "..." + value[-4:] if len(value) > 12 else "***"
            print(f"  [OK] {var}: {masked}")
        else:
            print(f"  [FAIL] {var}: Not set - {description}")
            missing.append(var)
    
    if missing:
        print(f"\n[WARNING] Missing API keys: {', '.join(missing)}")
        print("Create a .env file from .env.example and add your API keys")
        return False
    else:
        print("[SUCCESS] All required API keys set\n")
        return True


def test_directories():
    """Test required directories exist."""
    print("Testing directories...")
    required_dirs = [
        "data/raw",
        "data/processed",
        "data/vector_db",
        "logs",
        "results",
        "configs"
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"  [OK] {dir_path}")
        else:
            print(f"  [WARN] {dir_path} - creating...")
            path.mkdir(parents=True, exist_ok=True)
            all_exist = False
    
    print("[SUCCESS] All directories ready\n")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("RAG Benchmark System - Installation Test")
    print("=" * 60)
    print()
    
    tests = [
        ("Package Imports", test_imports),
        ("Configuration", test_config),
        ("Environment Variables", test_env),
        ("Directories", test_directories),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"[ERROR] {name} failed with error: {e}\n")
            results.append((name, False))
    
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for name, passed in results:
        status = "[PASSED]" if passed else "[FAILED]"
        print(f"{status}: {name}")
    
    print()
    
    all_passed = all(result for _, result in results)
    if all_passed:
        print("[SUCCESS] All tests passed! System is ready to use.")
        print("\nNext steps:")
        print("  1. python main.py prepare-data    # Download and prepare dataset")
        print("  2. python main.py benchmark        # Run benchmarks")
        print("  3. streamlit run src/ui/app.py     # Launch UI")
        return 0
    else:
        print("[WARNING] Some tests failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
