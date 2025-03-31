
BASE_DIR = ""
DB_DIR = "data/visEval_dataset/databases"
import os
import pandas as pd

table_dir = os.path.join(BASE_DIR, DB_DIR, "assets_maintenance")
# Read the relevant CSV files
fault_log_parts = pd.read_csv(os.path.join(table_dir, 'Fault_Log_Parts.csv'))
skills_required_to_fix = pd.read_csv(os.path.join(table_dir, 'Skills_Required_To_Fix.csv'))
skills = pd.read_csv(os.path.join(table_dir, 'Skills.csv'))

# Merge the dataframes to get the skill descriptions for each fault
merged_df = fault_log_parts.merge(skills_required_to_fix, on='part_fault_id')
merged_df = merged_df.merge(skills, on='skill_id')

# Count the number of faults for each skill description
fault_counts = merged_df['skill_description'].value_counts().reset_index()
fault_counts.columns = ['skill_description', 'fault_count']

# Sort the dataframe by fault count in descending order
result = fault_counts.sort_values(by='fault_count', ascending=False)
print(result)