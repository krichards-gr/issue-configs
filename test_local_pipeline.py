import os
import subprocess
import sys

def test_local_pipeline():
    """
    Test the issue configuration pipeline.
    This script mirrors the test process used in the production repository.
    """
    print("=== Testing local_analysis.py ===")
    
    # 1. Run local_analysis.py (which internally regenerates topics.json)
    print("Running local_analysis.py...")
    try:
        subprocess.run(
            [sys.executable, "local_analysis.py"], 
            check=True
        )
        print("\nPipeline test SUCCESSFUL.")
    except subprocess.CalledProcessError as e:
        print(f"\nPipeline test FAILED with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_local_pipeline()
