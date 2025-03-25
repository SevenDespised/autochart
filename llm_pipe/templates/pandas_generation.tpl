You are a python coder and pandas data analysis expert.
Below, you are presented with a database schema composed of CSV files and a question.
Your task is to read the database schemas, understand the question, and use the hint to generate a valid python-pandas code to select the correct columns used to visualize the data.
You don't need to visualize, only select the columns.

Database Schema Overview:
{DATABASE_SCHEMA}

This schema describes the CSV files' structure, including filenames, columns, data types, and example values. Pay special attention to the examples listed beside each column, as they directly indicate which columns are relevant to our task.

Data processing instructions:
. Use .read_csv to read files. If table names do not include extensions (.csv), add them. Prefix the table path with {{TABLE_PATH}}
. Prefix the table path with {{TABLE_PATH}}(e.g., {{TABLE_PATH}}/table_name.csv)
. Default to id columns unless the question explicitly refers to names (e.g., "customer name").
. Output only the information directly asked in the question.
. Make sure to return all required information without omissions or extraneous data.
. Use the example values provided to map question keywords to correct column names.
. If the question asks for a count, use `.value_counts()` to get the count of unique values.
. Use `.dropna(subset=[col])` before operations requiring non-null values. For calculations, consider .fillna(0) if appropriate for the context.
. At the end of the task, determine whether sorting is required based on the problem. If the task requires sorting, only include sorting columns in the final output if explicitly requested.
. At the end of the task, store the final series in variables named `result1`, `result2`, `result3`.

###
Question: 
{QUESTION} 

Hint:
1.{HINT}

2.Possibly relevant columns:
{COLS}

Please respond with a JSON object structured as follows:

{{
    "chain_of_thought_reasoning": "Your thought process on how you arrived at the final code.",
    "pandas_code": "Your pandas code in a string."
}}

Take a deep breath and think step by step to find the correct pandas code. If you follow all the instructions and generate the correct code, I will give you 1 million dollars.