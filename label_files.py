import os
import pandas as pd
import argparse
from datetime import datetime
from analyzer import IssueAnalyzer

def main():
    parser = argparse.ArgumentParser(description="Label a CSV or XLSX file with issue topics.")
    parser.add_argument("input_file", help="Path to the input file (csv or xlsx)")
    parser.add_argument("--column", help="Name of the column containing text (optional, will prompt if omitted)")
    
    args = parser.parse_args()
    input_path = args.input_file

    if not os.path.exists(input_path):
        print(f"Error: File not found: {input_path}")
        return

    # 1. Load the file
    print(f"Loading {input_path}...")
    try:
        if input_path.endswith('.csv'):
            df = pd.read_csv(input_path)
        elif input_path.endswith('.xlsx'):
            df = pd.read_excel(input_path)
        else:
            print("Error: Unsupported file format. Please provide a .csv or .xlsx file.")
            return
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # 2. Pick the column
    columns = df.columns.tolist()
    target_column = args.column

    if not target_column or target_column not in columns:
        if target_column:
            print(f"Warning: Column '{target_column}' not found.")
        
        print("\nAvailable columns:")
        for i, col in enumerate(columns):
            print(f"[{i}] {col}")
        
        try:
            choice = int(input(f"\nSelect the column index containing the text [0-{len(columns)-1}]: "))
            target_column = columns[choice]
        except (ValueError, IndexError):
            print("Invalid selection.")
            return

    print(f"Processing column: '{target_column}'")

    # 3. Initialize Analyzer
    analyzer = IssueAnalyzer()

    # 4. Apply labeling
    print("Labeling entries...")
    t_start = datetime.now()
    
    results = []
    for text in df[target_column]:
        detected = analyzer.analyze_text(str(text))
        if not detected:
            results.append({"detected_topics": None, "issue_areas": None})
        else:
            topics = "; ".join([d['topic'] for d in detected])
            areas = "; ".join(list(set([d['issue_area'] for d in detected])))
            results.append({"detected_topics": topics, "issue_areas": areas})

    results_df = pd.DataFrame(results)
    final_df = pd.concat([df, results_df], axis=1)

    # 5. Output
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(input_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d")
    output_filename = f"{base_name}_topic_labeled_{timestamp}.csv"
    output_path = os.path.join(output_dir, output_filename)

    final_df.to_csv(output_path, index=False)
    
    t_end = datetime.now()
    print(f"\nSUCCESS: Processing complete in {t_end - t_start}")
    print(f"Results saved to: {output_path}")

if __name__ == "__main__":
    main()
