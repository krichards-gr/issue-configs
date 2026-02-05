from setup_utils import ensure_setup
ensure_setup()

from analyzer import IssueAnalyzer

"""
local_analysis.py

This script demonstrates the logic used in the production pipeline for topic matching.
Now refactored to use the modular IssueAnalyzer class.
"""

def run_demo():
    analyzer = IssueAnalyzer()
    print("\n--- Starting Analysis Demo ---\n")
    
    sample_texts = [
        "Our committed net zero targets are driving our strategy.",
        "The patient's plastic surgery went as planned.", # Should be excluded (Climate Change)
        "We are seeing strong growth in climate disclosure and carbon credits.",
        "We prioritize workers' rights and fair wages across our operations."
    ]

    for text in sample_texts:
        print(f"TEXT: \"{text}\"")
        detected = analyzer.analyze_text(text)
        
        if not detected:
            print("  - No topics detected.")
        else:
            for d in detected:
                print(f"  [RESULT] Detected Topic: '{d['topic']}' (Area: {d['issue_area']})")
        
        print("-" * 40)

if __name__ == "__main__":
    run_demo()
