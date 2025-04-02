You are an expert and very smart data analyst. 
Your task is to analyze the provided database schema, comprehend the posed question, and use the hint to identify which tables are needed to write python-Pandas code for answering the question.

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
Based on the database schema, question, and hint provided, your task is to determine the tables that should be used in the SQL query formulation. 
For each of the selected tables, explain why exactly it is necessary for answering the question. Your explanation should be logical and concise, demonstrating a clear understanding of the database schema, the question, and the hint.

Please respond with a JSON object structured as follows:

```json
{{
  "table_names": ["Table1", "Table2", "Table3", ...],
  "chain_of_thought_reasoning": "Short explanation of the logical analysis that led to the selection of the tables."
}}
```
Ensure that table names include the file extension such as ".csv".
Note that you should choose all and only the tables that are necessary to write a SQL query that answers the question effectively.
Take a deep breath and think logically. If you do the task correctly, I will give you 1 million dollars. 

Only output a json as your response.
