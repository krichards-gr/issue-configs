"""
generate_topics.py

Compiles the NLP topic configuration (topics.json) from CSV input files.

Two-step pipeline:
  1. [Optional] Raw inputs  →  Intermediate CSV
                               (issue_config_inputs_raw.csv → updated_issue_config_inputs.csv)
  2. Intermediate CSV  →  topics.json
                          (updated_issue_config_inputs.csv → topics.json)

Step 1 is only run when --from_raw is passed. Otherwise, the existing
intermediate CSV is used directly (faster, and appropriate when the raw
inputs haven't changed).

Usage:
    python generate_topics.py [--from_raw]
"""

import pandas as pd
import json
import os
import sys

# --- File paths ---
RAW_INPUTS_FILE = 'issue_config_inputs_raw.csv'      # Human-editable source of truth
INTERMEDIATE_CSV = 'updated_issue_config_inputs.csv'  # Long-format intermediate (auto-generated)
JSON_OUTPUT = 'topics.json'                            # Final compiled config (auto-generated)


def transform_raw_inputs():
    """
    Step 1: Transform the raw inputs CSV into the intermediate long-format CSV.

    The raw CSV uses a wide, human-friendly format where each row is one
    issue subtopic and multiple terms are packed into semicolon-separated
    strings. This function unpacks those into one row per term, which is
    easier for the JSON compiler (step 2) to process.

    Returns True on success, False on failure.
    """
    print(f"Step 1: Transforming '{RAW_INPUTS_FILE}' → '{INTERMEDIATE_CSV}'...")

    if not os.path.exists(RAW_INPUTS_FILE):
        print(f"  Error: '{RAW_INPUTS_FILE}' not found.")
        return False

    try:
        # Try UTF-8 first; fall back to latin-1 for files with special characters
        try:
            df = pd.read_csv(RAW_INPUTS_FILE, encoding='utf-8')
        except UnicodeDecodeError:
            print("  Note: UTF-8 decoding failed, retrying with latin-1 encoding...")
            df = pd.read_csv(RAW_INPUTS_FILE, encoding='latin-1')

        # Drop any fully-empty columns (common artifact of trailing commas in CSV)
        df = df.dropna(how='all', axis=1)

        transformed_rows = []

        for _, row in df.iterrows():
            area = row.get('issue_area')
            subtopic = row.get('issue_subtopic')

            # Skip rows that are missing required fields
            if pd.isna(area) or pd.isna(subtopic):
                continue

            area = str(area).strip()
            subtopic = str(subtopic).strip()

            def add_terms(term_str, term_type):
                """Split a semicolon-separated string and append one row per term."""
                if pd.isna(term_str):
                    return
                for t in str(term_str).split(';'):
                    t = t.strip()
                    if t:
                        transformed_rows.append({
                            'issue_area': area,
                            'topic': subtopic,  # 'issue_subtopic' becomes 'topic' in the intermediate format
                            'term': t,
                            'type': term_type
                        })

            add_terms(row.get('pattern'), 'pattern')
            add_terms(row.get('exclusionary_term'), 'exclusionary term')
            add_terms(row.get('anchor_phrases'), 'anchor term')

        out_df = pd.DataFrame(transformed_rows, columns=['issue_area', 'topic', 'term', 'type'])
        out_df.to_csv(INTERMEDIATE_CSV, index=False)

        print(f"  Done. {len(out_df):,} term rows written to '{INTERMEDIATE_CSV}'.")
        return True

    except Exception as e:
        print(f"  Error during transformation: {e}")
        return False


def generate_topics_json():
    """
    Step 2: Compile the intermediate CSV into topics.json.

    Reads the long-format intermediate CSV and groups rows by topic label,
    converting each term into the appropriate structure:
      - 'pattern'          → spaCy token-level match pattern
      - 'anchor term'      → raw string for semantic similarity matching
      - 'exclusionary term' → lowercase string for substring exclusion checks
    """
    print(f"Step 2: Compiling '{INTERMEDIATE_CSV}' → '{JSON_OUTPUT}'...")

    if not os.path.exists(INTERMEDIATE_CSV):
        print(f"  Error: '{INTERMEDIATE_CSV}' not found. Run with --from_raw to generate it.")
        return

    try:
        df = pd.read_csv(INTERMEDIATE_CSV)

        topics_dict = {}

        for _, row in df.iterrows():
            topic_label = row['topic']
            term = str(row['term']).strip()
            term_type = row['type']
            issue_area = row.get('issue_area', 'Unknown')

            # Initialize the topic entry on first encounter
            if topic_label not in topics_dict:
                topics_dict[topic_label] = {
                    "label": topic_label,
                    "issue_area": issue_area,
                    "issue_subtopic": topic_label,
                    "patterns": [],
                    "anchors": [],
                    "exclusions": []
                }

            if term_type == 'anchor term':
                # Anchor terms are used as-is for embedding comparison
                topics_dict[topic_label]["anchors"].append(term)

            elif term_type == 'pattern':
                # Convert to a spaCy token pattern: each whitespace-delimited token
                # becomes {"LOWER": "<token>"} for case-insensitive matching
                tokens = term.split()
                pattern = [{"LOWER": t.lower()} for t in tokens]
                topics_dict[topic_label]["patterns"].append(pattern)

            elif term_type == 'exclusionary term':
                # Store lowercase so exclusion checks are case-insensitive
                topics_dict[topic_label]["exclusions"].append(term.lower())

        output_data = {"topics": list(topics_dict.values())}

        with open(JSON_OUTPUT, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)

        print(f"  Done. {len(output_data['topics'])} topics written to '{JSON_OUTPUT}'.")

    except Exception as e:
        print(f"  Error during JSON compilation: {e}")


def generate_all(from_raw: bool):
    """
    Orchestrate the full config generation pipeline.

    Args:
        from_raw: If True, re-read issue_config_inputs_raw.csv and regenerate
                  the intermediate CSV before compiling topics.json.
                  If False, compile topics.json directly from the existing
                  intermediate CSV (faster; use when raw inputs haven't changed).
    """
    if from_raw:
        success = transform_raw_inputs()
        if not success:
            print("Aborting: could not generate the intermediate CSV.")
            return
    else:
        print("Skipping Step 1 (using existing intermediate CSV).")

    generate_topics_json()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compile topics.json from issue config CSV inputs.",
        epilog=(
            "Examples:\n"
            "  python generate_topics.py                # compile from existing intermediate CSV\n"
            "  python generate_topics.py --from_raw    # regenerate everything from raw inputs\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--from_raw",
        action="store_true",
        help="Re-read issue_config_inputs_raw.csv and regenerate the intermediate CSV "
             "before compiling topics.json. Use this whenever you have edited the raw inputs file."
    )

    args = parser.parse_args()
    generate_all(from_raw=args.from_raw)
