import importlib

# List of all key packages from your requirements.txt
packages = [
    "streamlit",
    "transformers",
    "peft",
    "datasets",
    "accelerate",
    "torch",
    "sentence_transformers",
    "numpy",
    "pydantic",
    "dotenv",         # some systems need 'dotenv' -> try this first
    "faiss",
    "pytest"
]

print("🔍 Checking installed packages...\n")

for pkg in packages:
    try:
        importlib.import_module(pkg)
        print(f"✅ {pkg} is installed and importable.")
    except ModuleNotFoundError:
        # some packages have different import names
        if pkg == "dotenv":
            try:
                importlib.import_module("python_dotenv")
                print(f"✅ {pkg} (via python-dotenv) is installed and importable.")
            except ModuleNotFoundError:
                print(f"❌ {pkg} is MISSING.")
        elif pkg == "sentence_transformers":
            try:
                importlib.import_module("sentence_transformers")
                print(f"✅ sentence-transformers is installed and importable.")
            except ModuleNotFoundError:
                print(f"❌ sentence-transformers is MISSING.")
        else:
            print(f"❌ {pkg} is MISSING.")

print("\n✨ Verification complete!")
