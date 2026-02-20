"""
setup_utils.py

Auto-installer and environment check for local tools.

Calling ensure_setup() verifies that all Python dependencies and NLP models
are present, installing or downloading anything that is missing. It then
regenerates topics.json so the analyzer always has the latest config.

This module is intended to be called at startup by label_files.py and
local_analysis.py before any heavy imports are attempted.
"""

import os
import sys
import subprocess
import importlib


def ensure_setup(from_raw=False, nlp_model="en_core_web_sm", embedding_model="all-MiniLM-L6-v2"):
    """
    Check for required dependencies and NLP models, installing anything missing.
    Then regenerate topics.json from the appropriate CSV source.

    Args:
        from_raw:        If True, regenerate the intermediate CSV from the raw
                         inputs before compiling topics.json. Pass this through
                         from the --from_raw CLI flag.
        nlp_model:       spaCy model name to check/download.
        embedding_model: SentenceTransformer model name to check/download.
    """
    print("=== Environment Check ===")

    # --- 1. Python package dependencies ---
    required_libs = ["pandas", "spacy", "sentence_transformers", "torch", "transformers", "openpyxl"]
    missing_libs = [
        lib for lib in required_libs
        if not _is_importable(lib.replace("-", "_"))
    ]

    if missing_libs:
        print(f"Missing packages: {', '.join(missing_libs)}")
        print("Installing from requirements.txt...")
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
                check=True
            )
            print("  Packages installed successfully.")
        except Exception as e:
            print(f"  Error installing packages: {e}")
            sys.exit(1)
    else:
        print("  Python packages: OK")

    # --- 2. spaCy language model ---
    import spacy
    try:
        spacy.load(nlp_model)
        print(f"  spaCy model '{nlp_model}': OK")
    except OSError:
        print(f"  Downloading spaCy model '{nlp_model}'...")
        try:
            subprocess.run([sys.executable, "-m", "spacy", "download", nlp_model], check=True)
            print(f"  spaCy model '{nlp_model}' installed.")
        except Exception as e:
            print(f"  Error downloading spaCy model: {e}")
            sys.exit(1)

    # --- 3. SentenceTransformer embedding model ---
    from sentence_transformers import SentenceTransformer
    try:
        SentenceTransformer(embedding_model)
        print(f"  SentenceTransformer model '{embedding_model}': OK")
    except Exception as e:
        # SentenceTransformer usually handles its own downloads; this is a soft warning
        print(f"  Warning: could not verify SentenceTransformer model '{embedding_model}': {e}")

    # --- 4. Compile topics.json ---
    source_desc = "raw inputs CSV" if from_raw else "existing intermediate CSV"
    print(f"  Compiling topics.json from {source_desc}...")
    try:
        from generate_topics import generate_all
        generate_all(from_raw=from_raw)
    except Exception as e:
        print(f"  Error compiling topics.json: {e}")

    print("=== Environment ready ===\n")


def _is_importable(module_name):
    """Return True if the given module can be imported."""
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    ensure_setup()
