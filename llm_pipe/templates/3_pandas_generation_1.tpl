You are a python coder and pandas data analysis expert.
Below, you are presented with a database schema composed of CSV files and a question.
Your task is to read the database schemas, understand the question, and use the hint to generate a valid python-pandas code to extract the correct dataframe used to visualize the data.
You don't need to visualize, only extract the dataframe.

Database Schema Overview:
{DATABASE_SCHEMA}

This schema describes the CSV files' structure, including filenames, columns, data types, and example values. Pay special attention to the examples listed beside each column, as they directly indicate which columns are relevant to our task.

Code writing instructions:
1. Use .read_csv to read files. If table names do not include extensions (.csv), add them.
2. Use the `os` module to prepend the table diretory to the file name, and the `table_dir` variable is predefined and do not redefine it.
3. Please ensure that the data type of each column is supported by JSON.
4. Output only the information directly asked in the question.
5. Make sure to return all required information without omissions or extraneous data.
6. Use the example values provided to map question keywords to correct column names and data format.
7. The quesition may specify the format of the data, such as converting the date into the abbreviations of months or days of the week.
8. If the question asks for a count, use `.value_counts()` to get the count of unique values.
9. Use `.dropna(subset=[col])` before operations requiring non-null values. For calculations, consider .fillna(0) if appropriate for the context.
10. At the end of the task, determine whether sorting is required based on the problem. If the task requires sorting, only include sorting columns in the final output if explicitly requested.
11. At the end of the task, store the final dataframe in variables named `result`.
12. Attention, if there is only one column in the `result`, please extract another column to serve as its x-axis

###
Question: 
{QUESTION} 

Hint:
1.{HINT}

2.Possibly relevant columns:
{COLS}

Please respond with a JSON object structured as follows:

{{
    "pandas_code": "Your pandas code in a string.",
    "chain_thought": "Your chain of thought in a string."
}}
Only output a json as your response, without any other text.
Take a deep breath and think step by step to find the correct pandas code. If you follow all the instructions and generate the correct code, I will give you 1 million dollars.

