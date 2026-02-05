import os
import sys
import subprocess
import importlib

def ensure_setup(nlp_model="en_core_web_sm", embedding_model="all-MiniLM-L6-v2"):
    """
    Checks for dependencies and models, installing them if missing.
    """
    print("=== Auto-Initialization ===")
    
    # 1. Check Python Dependencies
    required_libs = ["pandas", "spacy", "sentence_transformers", "torch", "transformers", "openpyxl"]
    missing_libs = []
    
    for lib in required_libs:
        # Map some lib names to import names if different
        import_name = lib.replace("-", "_")
        try:
            importlib.import_module(import_name)
        except ImportError:
            missing_libs.append(lib)
            
    if missing_libs:
        print(f"Missing libraries: {', '.join(missing_libs)}")
        print("Installing from requirements.txt...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
            print("Successfully installed dependencies.")
        except Exception as e:
            print(f"Error installing dependencies: {e}")
            sys.exit(1)

    # 2. Check spaCy Model
    import spacy
    print(f"Ensuring spaCy '{nlp_model}' is installed...")
    try:
        spacy.load(nlp_model)
        print(f"  spaCy model '{nlp_model}' ready.")
    except:
        print(f"  Downloading '{nlp_model}'...")
        try:
            subprocess.run([sys.executable, "-m", "spacy", "download", nlp_model], check=True)
            print(f"  Model '{nlp_model}' installed.")
        except Exception as e:
            print(f"  Error downloading spaCy model: {e}")
            sys.exit(1)

    # 3. Check SentenceTransformer
    from sentence_transformers import SentenceTransformer
    print(f"Ensuring SentenceTransformer '{embedding_model}' is available...")
    try:
        # This will download it to the cache if not present
        SentenceTransformer(embedding_model)
        print(f"  SentenceTransformer '{embedding_model}' ready.")
    except Exception as e:
        print(f"  Error loading/downloading SentenceTransformer: {e}")
        # Note: We don't sys.exit here because SentenceTransformer usually handles its own downloads

    # 4. Ensure topics.json exists
    if not os.path.exists("topics.json"):
        print("topics.json missing. Generating from raw inputs...")
        try:
            from generate_topics import generate_topics_json
            generate_topics_json()
            print("  Successfully generated topics.json.")
        except Exception as e:
            print(f"  Error generating topics.json: {e}")

    print("Success: Environment is ready.\n")

if __name__ == "__main__":
    ensure_setup()
