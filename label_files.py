"""
label_files.py

CLI tool for applying issue topic labels to a CSV or XLSX file.

Usage:
    python label_files.py <input_file> [--column COLUMN] [--from_raw]

    --from_raw   Re-read issue_config_inputs_raw.csv and regenerate the
                 compiled config before labeling. Use this whenever you
                 have edited the raw inputs CSV.
"""

import os
import argparse
from datetime import datetime
from setup_utils import ensure_setup


def main():
    parser = argparse.ArgumentParser(
        description="Label a CSV or XLSX file with issue topic tags.",
        epilog=(
            "Examples:\n"
            "  python label_files.py data.csv\n"
            "  python label_files.py data.csv --column text\n"
            "  python label_files.py data.csv --column text --from_raw\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input_file",
        help="Path to the input file (.csv or .xlsx)."
    )
    parser.add_argument(
        "--column",
        help="Name of the column containing the text to analyze. "
             "If omitted, you will be prompted to choose interactively."
    )
    parser.add_argument(
        "--from_raw",
        action="store_true",
        help="Regenerate the intermediate CSV and topics.json from "
             "issue_config_inputs_raw.csv before labeling. Use this "
             "whenever you have edited the raw inputs file."
    )

    args = parser.parse_args()

    if args.from_raw:
        print("Note: --from_raw is set. Topic config will be recompiled from the raw inputs CSV.\n")

    # Ensure all dependencies and NLP models are ready, then compile topics.json.
    # Heavy imports (pandas, IssueAnalyzer) are deferred until after setup so
    # that missing packages are installed before we try to import them.
    ensure_setup(from_raw=args.from_raw)

    import pandas as pd
    from analyzer import IssueAnalyzer

    input_path = args.input_file

    if not os.path.exists(input_path):
        print(f"Error: File not found: {input_path}")
        return

    # --- Load the input file ---
    print(f"Loading '{input_path}'...")
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

    print(f"  Loaded {len(df):,} rows.\n")

    # --- Select the text column ---
    columns = df.columns.tolist()
    target_column = args.column

    if not target_column or target_column not in columns:
        if target_column:
            print(f"Warning: Column '{target_column}' not found in this file.")

        print("Available columns:")
        for i, col in enumerate(columns):
            print(f"  [{i}] {col}")

        try:
            choice = int(input(f"\nEnter the index of the column containing the text to analyze [0-{len(columns)-1}]: "))
            target_column = columns[choice]
        except (ValueError, IndexError):
            print("Invalid selection. Exiting.")
            return

    print(f"\nAnalyzing column: '{target_column}'")

    # --- Initialize the analyzer (loads NLP models and topics.json) ---
    analyzer = IssueAnalyzer()

    # --- Apply topic labels to each row ---
    print(f"Labeling {len(df):,} rows...\n")
    t_start = datetime.now()

    results = []
    for i, text in enumerate(df[target_column], start=1):
        detected = analyzer.analyze_text(str(text))
        if not detected:
            results.append({"issue_subtopics": None, "issue_areas": None})
        else:
            subtopics = "; ".join([d['topic'] for d in detected])
            areas = "; ".join(list(dict.fromkeys([d['issue_area'] for d in detected])))
            results.append({"issue_subtopics": subtopics, "issue_areas": areas})

        # Print progress every 100 rows (and on the last row)
        if i % 100 == 0 or i == len(df):
            print(f"  {i:,} / {len(df):,} rows processed...")

    results_df = pd.DataFrame(results)
    final_df = pd.concat([df, results_df], axis=1)

    # --- Save output ---
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(input_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d")
    output_filename = f"{base_name}_topic_labeled_{timestamp}.csv"
    output_path = os.path.join(output_dir, output_filename)

    final_df.to_csv(output_path, index=False)

    t_end = datetime.now()
    labeled_count = results_df["issue_subtopics"].notna().sum()

    print(f"\nDone! Processed {len(df):,} rows in {t_end - t_start}.")
    print(f"  {labeled_count:,} rows received at least one topic label.")
    print(f"  Output saved to: {output_path}")


if __name__ == "__main__":
    main()
