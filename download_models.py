import os
import sys
import subprocess
from sentence_transformers import SentenceTransformer

def download_models():
    """
    Downloads the models required for the issue configuration framework.
    Mirrors the structure of the production pipeline's model management.
    """
    print("=== Model Downloader ===")
    
    # 1. Download spaCy model
    print("Ensuring spaCy 'en_core_web_sm' is installed...")
    try:
        import spacy
        try:
            spacy.load("en_core_web_sm")
            print("  spaCy model already present.")
        except:
            print("  Downloading 'en_core_web_sm'...")
            subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
    except ImportError:
        print("  ERROR: spaCy not installed. Please run 'pip install -r requirements.txt'")
        return

    # 2. Download SentenceTransformer model
    print("Ensuring SentenceTransformer 'all-MiniLM-L6-v2' is available...")
    try:
        # This will download it to the cache if not present
        SentenceTransformer('all-MiniLM-L6-v2')
        print("  SentenceTransformer model ready.")
    except Exception as e:
        print(f"  ERROR downloading SentenceTransformer: {e}")

    print("\nModel setup complete.")

if __name__ == "__main__":
    download_models()
