You are a python coder and pandas data analysis expert.
Below, you are presented with a database schema composed of CSV files and a question.
Your task is to read the database schemas, understand the question, and use the hint to generate a valid python-pandas code to extract the correct dataframe used to visualize the data.
You don't need to visualize, only extract the dataframe.

Database Schema Overview:
{DATABASE_SCHEMA}

This schema describes the CSV files' structure, including filenames, columns, data types, and example values. Pay special attention to the examples listed beside each column, as they directly indicate which columns are relevant to our task.

Code writing instructions:
. Use .read_csv to read files. If table names do not include extensions (.csv), add them.
. Use the `os` module to prepend the table diretory to the file name, and the `table_dir` variable is predefined and do not redefine it.
. Do not use pandas functions to process dates or times.
. Output only the information directly asked in the question.
. Make sure to return all required information without omissions or extraneous data.
. Use the example values provided to map question keywords to correct column names.
. If the question asks for a count, use `.value_counts()` to get the count of unique values.
. Use `.dropna(subset=[col])` before operations requiring non-null values. For calculations, consider .fillna(0) if appropriate for the context.
. At the end of the task, determine whether sorting is required based on the problem. If the task requires sorting, only include sorting columns in the final output if explicitly requested.
. At the end of the task, store the final dataframe in variables named `result`.
. Attention, in `result`, at least two columns should be extracted for the chart.
###
Question: 
{QUESTION} 

Hint:
1.{HINT}

2.Possibly relevant columns:
{COLS}

Please respond with a JSON object structured as follows:

{{
    "pandas_code": "Your pandas code in a string."
}}

Take a deep breath and think step by step to find the correct pandas code. If you follow all the instructions and generate the correct code, I will give you 1 million dollars.