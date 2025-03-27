import pandas as pd
import os

BASE_DIR = ""
DB_DIR = "data/visEval_dataset/databases"
TABLE_PATH = os.path.join(BASE_DIR, DB_DIR, "activity_1")
# Read the Faculty and Student CSV files
faculty_df = pd.read_csv(f'{TABLE_PATH}/Faculty.csv')
student_df = pd.read_csv(f'{TABLE_PATH}/Student.csv')

# Join the Faculty and Student dataframes on the FacID and Advisor columns
merged_df = pd.merge(faculty_df, student_df, left_on='FacID', right_on='Advisor')

# Group by the Rank column and count the number of students in each group
result = merged_df.groupby('Rank').size().reset_index(name='Number of Students')

# Print the result
print(result)