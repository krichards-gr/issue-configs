import pandas as pd
import json
import shutil

try:
    shutil.copyfile('issue_definitions.xlsx', 'temp_issue_definitions.xlsx')
    xl = pd.ExcelFile('temp_issue_definitions.xlsx')
    
    output = f"Available sheets: {xl.sheet_names}\n\n"
    
    checked_count = 0
    for sheet in xl.sheet_names:
        if sheet == 'dataset_info':
            continue
            
        output += f"\n--- SHEET: {sheet} ---\n"
        df = pd.read_excel('temp_issue_definitions.xlsx', sheet_name=sheet)
        output += f"Columns: {df.columns.tolist()}\n"
        
        # Select columns of interest
        cols_to_show = [c for c in df.columns if isinstance(c, str) and any(x in c.lower() for x in ['issue', 'subtopic', 'pattern', 'anchor', 'exclu', 'colo'])]
        if not cols_to_show:
            cols_to_show = df.columns.tolist()[:8]
            
        output += "First 2 rows:\n"
        records = df[cols_to_show].head(2).to_dict(orient='records')
        output += json.dumps(records, indent=2, default=str) + "\n"
        
        checked_count += 1
        if checked_count >= 3: 
            break
            
    with open('inspect_subtopics_output.txt', 'w') as f:
        f.write(output)
    print("Successfully wrote to inspect_subtopics_output.txt")
except Exception as e:
    print(f"Error: {e}")
