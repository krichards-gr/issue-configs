"""
label_issue_areas.py

A CLI tool to apply the new Issue Area matching logic to either:
  1. A local CSV (or Excel) file
  2. A specific BigQuery table and column

The output is always saved as a local CSV file in the 'outputs/' directory.

Usage examples:
  1. Local CSV:
     python label_issue_areas.py --file my_data.csv --column "text_content"

  2. BigQuery Table:
     python label_issue_areas.py --bq-table "my-project.my_dataset.my_table" --column "transcript_content" --limit 100
"""

import os
import argparse
from datetime import datetime
import pandas as pd
from google.cloud import bigquery

# Import our new matcher engine (make sure topics are generated first!)
from issue_area_matcher import IssueAreaMatcher
from generate_issue_area_topics import generate_json

OUTPUT_DIR = "outputs"

def check_and_generate_config():
    """Ensure the JSON configuration file exists before we start."""
    if not os.path.exists('issue_area_topics.json'):
        print("Config 'issue_area_topics.json' not found. Generating it now from Excel...")
        success = generate_json()
        if not success:
            print("Failed to generate configuration. Exiting.")
            exit(1)

def apply_labels_to_dataframe(df, text_column, fuzzy_ratio=85):
    """
    Takes a pandas DataFrame and a target text column, instantiates the matcher,
    and appends a new 'Issue_Areas' column with the results.
    """
    if df.empty:
        print("Warning: DataFrame is empty. Nothing to label.")
        return df
        
    if text_column not in df.columns:
        print(f"Error: Column '{text_column}' not found in the data.")
        print(f"Available columns: {df.columns.tolist()}")
        exit(1)

    print(f"Initializing Issue Area Matcher with fuzzy ratio {fuzzy_ratio}...")
    matcher = IssueAreaMatcher(min_fuzzy_ratio=fuzzy_ratio)
    
    print(f"Applying labels to {len(df)} rows...")
    t_start = datetime.now()
    
    labeled_results = []
    
    # Process each row
    for index, row in df.iterrows():
        text_val = str(row[text_column]) if pd.notna(row[text_column]) else ""
        
        # Analyze the text snippet using our new engine
        matched_areas, matched_subtopics = matcher.analyze_text(text_val)
        
        labeled_results.append({
            "Issue_Areas": matched_areas if matched_areas else "",
            "Issue_Subtopics": matched_subtopics if matched_subtopics else ""
        })
        
        # Simple progress indicator for large datasets
        if (index + 1) % 500 == 0:
            print(f"  Processed {index + 1} / {len(df)} rows...")

    t_end = datetime.now()
    print(f"Labeling complete in {t_end - t_start}.")
    
    # Attach our new results column to the original dataframe
    results_df = pd.DataFrame(labeled_results)
    
    # Reset indices to ensure clean concatenation just in case BQ returned weird indices
    df.reset_index(drop=True, inplace=True)
    results_df.reset_index(drop=True, inplace=True)
    
    final_df = pd.concat([df, results_df], axis=1)
    return final_df

def process_local_file(file_path, text_column, fuzzy_ratio):
    """Reads a local CSV/Excel file, applies labels, and saves the output."""
    print(f"Reading local file: {file_path}")
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        exit(1)
        
    try:
        if file_path.lower().endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.lower().endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:
            print("Error: Unsupported file type. Must be .csv or .xlsx")
            exit(1)
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        exit(1)
        
    final_df = apply_labels_to_dataframe(df, text_column, fuzzy_ratio)
    save_output(final_df, base_name=os.path.basename(file_path).split('.')[0])

def process_bigquery(bq_table, text_column, limit=None, fuzzy_ratio=85):
    """Pulls data from a BigQuery table, applies labels, and saves to a local CSV."""
    print(f"Connecting to BigQuery to pull from table: {bq_table}")
    
    try:
        client = bigquery.Client()
    except Exception as e:
        print(f"Error initializing BigQuery client: {e}")
        print("Tip: Make sure you have run 'gcloud auth application-default login' if running locally.")
        exit(1)
        
    # Construct the query. We select * to keep all original metadata columns along with the text
    query = f"SELECT * FROM `{bq_table}`"
    if limit:
        query += f" LIMIT {limit}"
        
    print(f"Executing query: {query}")
    try:
        query_job = client.query(query)
        df = query_job.result().to_dataframe()
    except Exception as e:
        print(f"Error executing BigQuery query: {e}")
        exit(1)
        
    print(f"Downloaded {len(df)} rows from BigQuery.")
    
    final_df = apply_labels_to_dataframe(df, text_column, fuzzy_ratio)
    
    # Create a safe base name from the table name (e.g., project.dataset.table -> table)
    safe_table_name = bq_table.split('.')[-1]
    save_output(final_df, base_name=f"bq_{safe_table_name}")

def save_output(df, base_name="labeled_results"):
    """Saves the final DataFrame to the outputs/ directory."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{base_name}_issue_areas_{timestamp}.csv"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    
    try:
        df.to_csv(output_path, index=False)
        print(f"\nSUCCESS! Results saved locally to: {output_path}")
    except Exception as e:
        print(f"Error saving output file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply Issue Area labels to a CSV or BigQuery table.")
    
    # Input source options (mutually exclusive)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", help="Path to a local CSV or XLSX file to label.")
    group.add_argument("--bq-table", help="Fully qualified BigQuery table name (e.g., project.dataset.table).")
    
    # Required text column and optional BQ args
    parser.add_argument("--column", required=True, help="The name of the column containing the text to analyze.")
    parser.add_argument("--limit", type=int, default=None, help="(BigQuery only) Maximum number of rows to pull.")
    parser.add_argument("--fuzzy-ratio", type=int, default=85, help="Minimum matching ratio for fuzzy term matching (0-100). Default is 85.")
    
    args = parser.parse_args()
    
    # Ensure config is ready
    check_and_generate_config()
    
    # Route execution based on input source
    if args.file:
        process_local_file(args.file, args.column, args.fuzzy_ratio)
    elif args.bq_table:
        process_bigquery(args.bq_table, args.column, args.limit, args.fuzzy_ratio)
