import pandas as pd
import json
import os
import sys

# Configuration
RAW_INPUTS_FILE = 'issue_config_inputs_raw.csv'
INTERMEDIATE_CSV = 'updated_issue_config_inputs.csv'
JSON_OUTPUT = 'topics.json'

def transform_raw_inputs():
    """
    Reads the raw input CVS (human-friendly format) and transforms it into 
    the intermediate long-format CSV used for generation.
    """
    print(f"Transforming {RAW_INPUTS_FILE} to {INTERMEDIATE_CSV}...")
    
    if not os.path.exists(RAW_INPUTS_FILE):
        print(f"Error: {RAW_INPUTS_FILE} not found.")
        return False

    try:
        # Read raw inputs
        # Expecting cols: issue_area, issue_subtopic, pattern, exclusionary_term, anchor_phrases
        df = pd.read_csv(RAW_INPUTS_FILE)
        
        # Clean up any potential empty columns from trailing commas
        df = df.dropna(how='all', axis=1)
        
        transformed_rows = []
        
        for index, row in df.iterrows():
            area = row.get('issue_area')
            subtopic = row.get('issue_subtopic')
            
            # Skip empty rows
            if pd.isna(area) or pd.isna(subtopic):
                continue
                
            area = str(area).strip()
            subtopic = str(subtopic).strip()
            
            # Helper to add rows
            def add_terms(term_str, term_type):
                if pd.isna(term_str): return
                terms = str(term_str).split(';')
                for t in terms:
                    t = t.strip()
                    if t:
                        transformed_rows.append({
                            'issue_area': area,
                            'issue_subtopic': subtopic,
                            'topic': subtopic, # Mapping subtopic as the main topic key
                            'term': t,
                            'type': term_type
                        })

            # Process each column
            add_terms(row.get('pattern'), 'pattern')
            add_terms(row.get('exclusionary_term'), 'exclusionary term')
            add_terms(row.get('anchor_phrases'), 'anchor term')

        # Create DataFrame and save
        out_df = pd.DataFrame(transformed_rows)
        # Ensure consistent column order
        cols = ['issue_area', 'issue_subtopic', 'topic', 'term', 'type']
        out_df = out_df[cols]
        
        out_df.to_csv(INTERMEDIATE_CSV, index=False)
        print(f"Successfully created {INTERMEDIATE_CSV} with {len(out_df)} rows.")
        return True
        
    except Exception as e:
        print(f"Error during transformation: {e}")
        return False

def generate_topics_json():
    """
    Reads the intermediate CSV and generates the final topics.json file
    used by the analysis pipeline.
    """
    print(f"Generating {JSON_OUTPUT} from {INTERMEDIATE_CSV}...")
    
    if not os.path.exists(INTERMEDIATE_CSV):
        print(f"Error: {INTERMEDIATE_CSV} not found.")
        return

    try:
        df = pd.read_csv(INTERMEDIATE_CSV)
        
        topics_dict = {}
        
        for _, row in df.iterrows():
            topic_label = row['topic']
            term = str(row['term']).strip()
            term_type = row['type']
            issue_area = row.get('issue_area', 'Unknown')
            
            if topic_label not in topics_dict:
                topics_dict[topic_label] = {
                    "label": topic_label,
                    "issue_area": issue_area,
                    "issue_subtopic": row.get('issue_subtopic', 'Unknown'),
                    "patterns": [],
                    "anchors": [],
                    "exclusions": []
                }
                
            if term_type == 'anchor term':
                topics_dict[topic_label]["anchors"].append(term)
            elif term_type == 'pattern':
                # spaCy pattern generation: split by whitespace
                tokens = term.split()
                pattern = [{"LOWER": t.lower()} for t in tokens]
                topics_dict[topic_label]["patterns"].append(pattern)
            elif term_type == 'exclusionary term':
                topics_dict[topic_label]["exclusions"].append(term.lower())
                
        output_data = {
            "topics": list(topics_dict.values())
        }
        
        with open(JSON_OUTPUT, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)
            
        print(f"Successfully generated {JSON_OUTPUT} with {len(output_data['topics'])} topics.")
        
    except Exception as e:
        print(f"Error during JSON generation: {e}")

def generate_all():
    """
    Orchestrates the full generation pipeline:
    1. Raw Inputs -> Intermediate CSV
    2. Intermediate CSV -> Topics JSON
    """
    if transform_raw_inputs():
        generate_topics_json()
    else:
        print("Skipping JSON generation due to transformation failure.")

if __name__ == "__main__":
    generate_all()
