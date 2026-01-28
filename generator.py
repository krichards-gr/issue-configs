import pandas as pd
import json
import os
import sys

def generate_config(csv_path, json_path, include_exclusions=True):
    """
    Transforms a CSV issue definition file into a JSON format used by our analysis pipeline.
    
    Args:
        csv_path (str): Path to the source CSV file (e.g., 'topic_definitions.csv').
        json_path (str): Path where the output JSON will be saved (e.g., 'topics.json').
        include_exclusions (bool): If True, 'exclusionary term' types from the CSV will be included.
    """
    print(f"--- Configuration Generator ---")
    print(f"Source: {csv_path}")
    print(f"Output: {json_path}")
    print(f"Include Exclusions: {include_exclusions}")

    # 1. Check if the source file exists
    if not os.path.exists(csv_path):
        print(f"ERROR: Could not find '{csv_path}'. Please make sure the file exists.")
        return

    # 2. Load the CSV data using pandas
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"ERROR: Failed to read CSV: {e}")
        return

    # 3. Process each row and group them by 'topic'
    topics_dict = {}
    
    for _, row in df.iterrows():
        # Clean up the data: remove extra spaces and handle potential missing values
        topic_label = str(row['topic']).strip()
        term = str(row['term']).strip()
        term_type = str(row['type']).strip().lower()
        
        # Initialize the topic entry if we haven't seen it yet
        if topic_label not in topics_dict:
            topics_dict[topic_label] = {
                "label": topic_label,
                "patterns": [],     # For exact word matching
                "anchors": [],      # For vector/similarity matching
                "exclusions": []    # For negative filtering
            }
            
        # Map the CSS 'type' to our internal structure
        if term_type == 'anchor term':
            topics_dict[topic_label]["anchors"].append(term)
        
        elif term_type == 'pattern':
            # 'patterns' are processed into a specific format for our NLP engine (spaCy)
            # We break the term into lowercase tokens.
            tokens = term.split()
            pattern = [{"LOWER": t.lower()} for t in tokens]
            topics_dict[topic_label]["patterns"].append(pattern)
            
        elif term_type == 'exclusionary term':
            # Only add these if the toggle is enabled
            if include_exclusions:
                topics_dict[topic_label]["exclusions"].append(term.lower())
    
    # 4. Filter out empty exclusion lists if they aren't needed
    # This keeps the final JSON clean.
    final_topics = list(topics_dict.values())
    if not include_exclusions:
        for topic in final_topics:
            if "exclusions" in topic:
                del topic["exclusions"]

    # 5. Save the final dictionary as a pretty-printed JSON file
    output_data = {"topics": final_topics}
    
    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)
        print(f"SUCCESS: Generated {json_path} with {len(final_topics)} topics.")
    except Exception as e:
        print(f"ERROR: Failed to save JSON: {e}")

if __name__ == "__main__":
    # By default, we generate a standard config from the main definitions
    # But you can easily change these parameters below!
    
    # Example 1: Standard Generation
    generate_config(
        csv_path='topic_definitions.csv', 
        json_path='topics.json', 
        include_exclusions=False
    )
    
    # Example 2: Experimental Generation with Exclusions
    # Note: We use the 'updated' CSV for this one as it contains exclusions
    if os.path.exists('updated_issue_config_inputs.csv'):
        generate_config(
            csv_path='updated_issue_config_inputs.csv', 
            json_path='test_topics.json', 
            include_exclusions=True
        )
