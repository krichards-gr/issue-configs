import pandas as pd
import json

try:
    df = pd.read_excel('issue_definitions.xlsx', sheet_name='dataset_info')
    
    cols = df.columns.tolist()
    
    output = "--- COLUMNS ---\n"
    for c in cols:
        output += f"'{c}'\n"
        
    output += "\n--- SAMPLE DATA ---\n"
    records = df.head(10).to_dict(orient='records')
    output += json.dumps(records, indent=2, default=str)
    
    with open('inspect_output.txt', 'w') as f:
        f.write(output)
    print("Successfully wrote to inspect_output.txt")
except Exception as e:
    print(f"Error: {e}")
