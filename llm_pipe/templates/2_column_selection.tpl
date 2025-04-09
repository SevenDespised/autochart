You are an expert and very smart data analyst.
Your task is to examine the provided database schema, understand the posed question, and use the hint to pinpoint the specific columns within every tables that are essential for writing python-Pandas code for answering the question.

Database Schema Overview:
{DATABASE_SCHEMA}

This schema provides a detailed definition of the database's structure, including tables and their columns
For key phrases mentioned in the question, we have provided the most similar values within the columns denoted by "examples". This is a critical hint to identify the tables that will be used in the SQL query.

Question:
{QUESTION}

Hint:
{HINT}

The hint aims to direct your focus towards the specific elements of the database schema that are crucial for answering the question effectively.

Task:
Based on the database schema, question, and hint provided, your task is to identify all and only the columns that are essential for crafting a SQL query to answer the question.
For each of the selected columns, explain why exactly it is necessary for answering the question. Your reasoning should be concise and clear, demonstrating a logical connection between the columns and the question asked.

Tip: If you are choosing a column for filtering a value within that column, make sure that column has the value as an example.


Please respond with a JSON object structured as follows:

```json
{{
  "chain_thought": "Your chain of thought in a string."
  "TABLE_NAME1": ["COLUMN1", "COLUMN2", ...],
  "TABLE_NAME2": ["COLUMN1", "COLUMN2", ...],
  ...
}}
```
Ensure that table names include the file extension such as ".csv".
Make sure your response includes the table names as keys, each associated with a list of column names that are necessary for writing a SQL query to answer the question.
For each aspect of the question, provide a clear and concise explanation of your reasoning behind selecting the columns.
Take a deep breath and think logically. If you do the task correctly, I will give you 1 million dollars.

Only output a json as your response.