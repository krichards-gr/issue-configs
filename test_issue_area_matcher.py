"""
test_issue_area_matcher.py

Simple smoke-testing script to visually verify that the issue_area_matcher
works as expected against provided text inputs or a hardcoded test suite.

Usage:
    python test_issue_area_matcher.py "Some text containing climate change."
    python test_issue_area_matcher.py --suite
"""

import sys
import argparse
from issue_area_matcher import IssueAreaMatcher

def run_suite():
    print("Initializing Matcher...")
    matcher = IssueAreaMatcher()
    
    # 1. Climate Change (Pattern match, simple)
    # 2. Crime & Safety (Pattern match + Colocation)
    # 3. AI + Equity (Parent Match + Subtopic Match)
    
    tests = [
        {
            "name": "1. Exact match (No Colocation)",
            "text": "The company announced it is committed to net zero emissions.",
            "expected_areas": "Climate Change & Sustainability",
            "expected_subtopics": None
        },
        {
            "name": "2. Exact match BUT fails Exclusion filter",
            "text": "I had plastic surgery yesterday, so I am out of the office.",
            "expected_areas": None,
            "expected_subtopics": None
        },
        {
            "name": "3. Exact match BUT fails Colocation filter",
            "text": "There was a huge crime wave recently in the city.",
            "expected_areas": None,
            "expected_subtopics": None
        },
        {
            "name": "4. Exact match AND passes Colocation filter",
            "text": "There was a huge crime wave. Protesters were violent.",
            "expected_areas": "Crime & Safety",
            "expected_subtopics": None
        },
        {
            "name": "5. Issue Area Match AND Subtopic Match",
            "text": "The new AI algorithm was accused of generating racist slurs.",
            # "AI" matches parent, "racist slurs" matches the "Equity" subtopic
            "expected_areas": "AI",
            "expected_subtopics": "Equity"
        },
        {
            "name": "6. Subtopic match but missing Parent Area (Ignores Subtopic)",
            "text": "There are no complaints of hatespeech or racist slurs here.",
            # "racist slurs" matches the "Equity" subtopic, but "AI" is missing!
            # So the subtopic should NOT trigger.
            "expected_areas": None,
            "expected_subtopics": None
        }
    ]
    
    print("\n--- Running Test Suite ---")
    for t in tests:
        print(f"\n[TEST]: {t['name']}")
        print(f"Text: '{t['text']}'")
        areas, subtopics = matcher.analyze_text(t['text'])
        
        print(f"Result Areas: {areas}")
        print(f"Result Subtopics: {subtopics}")
        
        # Check Areas
        if t['expected_areas'] is None:
            if areas is None:
                print("✅ PASS: correctly returned None for Issue Areas")
            else:
                print(f"❌ FAIL: Expected None for Issue Areas, got {areas}")
        else:
            if areas and t['expected_areas'] in areas:
                print(f"✅ PASS: Found expected Issue Area: {t['expected_areas']}")
            else:
                print(f"❌ FAIL: Did not find Issue Area {t['expected_areas']}")
                
        # Check Subtopics
        if t['expected_subtopics'] is None:
            if subtopics is None:
                print("✅ PASS: correctly returned None for Subtopics")
            else:
                print(f"❌ FAIL: Expected None for Subtopics, got {subtopics}")
        else:
            if subtopics and t['expected_subtopics'] in subtopics:
                print(f"✅ PASS: Found expected Subtopic: {t['expected_subtopics']}")
            else:
                print(f"❌ FAIL: Did not find Subtopic {t['expected_subtopics']}")

def test_custom(text):
    print("Initializing Matcher...")
    matcher = IssueAreaMatcher()
    
    print("\n--- Custom Text Test ---")
    print(f"Text: '{text}'")
    areas, subtopics = matcher.analyze_text(text)
    print(f"\nFinal Matched Issue Areas: {areas}")
    print(f"Final Matched Subtopics: {subtopics}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Issue Area Matcher")
    parser.add_argument("text", nargs="?", default=None, help="Text snippet to test")
    parser.add_argument("--suite", action="store_true", help="Run the automated test suite")
    
    args = parser.parse_args()
    
    if args.suite:
        run_suite()
    elif args.text:
        test_custom(args.text)
    else:
        parser.print_help()
