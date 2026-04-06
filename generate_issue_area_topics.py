"""
generate_issue_area_topics.py

Compiles the 'dataset_info' sheet from issue_definitions.xlsx into a 
clean JSON configuration file (issue_area_topics.json).

This JSON file is used by the issue_area_matcher.py script to identify
Issue Areas in text segments. The parsing process supports:
  - Simple exact match patterns
  - Semantic anchor phrases
  - Standard exclusionary terms
  - Colocated term sets (1a/1b, 2a/2b, 3a/3b) which require AT LEAST ONE 
    term from the 'a' list AND AT LEAST ONE term from the 'b' list to be present anywhere.
"""

import pandas as pd
import json
import os

EXCEL_FILE = 'issue_definitions.xlsx'
DATASET_SHEET = 'dataset_info'
SUBTOPIC_SHEET = 'subtopic_terms'
JSON_OUTPUT = 'issue_area_topics.json'

def parse_semicolon_list(term_str):
    """
    Safely stringify, split by semicolon, and clean up a list of terms.
    If the value is NaN or empty, returns an empty list.
    """
    if pd.isna(term_str):
        return []
    terms = [t.strip() for t in str(term_str).split(';')]
    return [t for t in terms if t]

def generate_json():
    print(f"Reading '{DATASET_SHEET}' and '{SUBTOPIC_SHEET}' from {EXCEL_FILE}...")
    if not os.path.exists(EXCEL_FILE):
        print(f"Error: {EXCEL_FILE} not found.")
        return False
        
    try:
        df_areas = pd.read_excel(EXCEL_FILE, sheet_name=DATASET_SHEET, dtype=str)
        df_subtopics = pd.read_excel(EXCEL_FILE, sheet_name=SUBTOPIC_SHEET, dtype=str)
    except Exception as e:
        print(f"Error reading excel file: {e}")
        return False

    # First, let's parse all subtopics and group them by parent Issue Area
    subtopics_by_area = {}
    for _, row in df_subtopics.iterrows():
        issue_area = str(row.get('Issue Area')).strip()
        subtopic_name = str(row.get('Subtopics')).strip()
        
        if pd.isna(issue_area) or pd.isna(subtopic_name) or not issue_area or not subtopic_name:
            continue
            
        patterns_raw = parse_semicolon_list(row.get('pattern'))
        patterns = [p.lower() for p in patterns_raw]
        
        anchors = parse_semicolon_list(row.get('anchor_phrases'))
        
        exclusions_raw = parse_semicolon_list(row.get('exclusionary_term'))
        exclusions = [e.lower() for e in exclusions_raw]
        
        # Skip if nothing to match
        if not patterns and not anchors:
            continue
            
        if issue_area not in subtopics_by_area:
            subtopics_by_area[issue_area] = []
            
        subtopics_by_area[issue_area].append({
            "name": subtopic_name,
            "patterns": patterns,
            "anchors": anchors,
            "exclusions": exclusions
        })

    # Now parse the main Issue Areas
    topics_list = []
    
    for _, row in df_areas.iterrows():
        issue_area = row.get('Issue Area')
        
        if pd.isna(issue_area) or not str(issue_area).strip():
            continue
            
        issue_area = str(issue_area).strip()
        
        # Parse patterns
        patterns_raw = parse_semicolon_list(row.get('pattern'))
        patterns = [p.lower() for p in patterns_raw]
            
        # Parse anchors
        anchors = parse_semicolon_list(row.get('anchor_phrases'))
        
        # Parse exclusions
        exclusions = [e.lower() for e in parse_semicolon_list(row.get('exclusionary_term'))]
        
        # Parse colocation sets
        colocated_sets = []
        for i in range(1, 4):
            col_a = f'colocated_pattern_{i}a'
            col_b = f'colocated_pattern_{i}b'
            
            if col_a in df_areas.columns and col_b in df_areas.columns:
                terms_a = parse_semicolon_list(row.get(col_a))
                terms_b = parse_semicolon_list(row.get(col_b))
                
                if terms_a and terms_b:
                    colocated_sets.append({
                        "a": [t.lower() for t in terms_a],
                        "b": [t.lower() for t in terms_b]
                    })
                    
        # Skip an issue area if it lacks patterns and anchors entirely
        if not patterns and not anchors:
            print(f"  Warning: Skipping '{issue_area}' because it has no patterns or anchors.")
            continue
            
        topics_list.append({
            "issue_area": issue_area,
            "patterns": patterns,
            "anchors": anchors,
            "exclusions": exclusions,
            "colocated_sets": colocated_sets,
            "subtopics": subtopics_by_area.get(issue_area, [])  # Attach any parsed children subtopics
        })
        
    output_data = {"topics": topics_list}
    
    print(f"Writing {len(topics_list)} issue areas to {JSON_OUTPUT}...")
    try:
        with open(JSON_OUTPUT, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)
        print("Success! Configuration compiled.")
        return True
    except Exception as e:
        print(f"Error writing JSON: {e}")
        return False

if __name__ == "__main__":
    generate_json()

