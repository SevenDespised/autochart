import json
import os
from pprint import pprint

from llm_pipe.eval.data_loader import DataLoader
from llm_pipe.src.pipe.pipeline import PipelineProcessor
from llm_pipe.utils.data_preprocess import extract_key_values
from llm_pipe.utils.report_process import store_report

BASE_DIR = ""
DATA_DIR = "data/visEval_dataset/visEval_clear.json"
#CONF_DIR = "llm_pipe/config/hw_deepseek_v3.json"
#CONF_DIR = "llm_pipe/config/config.json"
CONF_DIR = "llm_pipe/config/experiment_70b_pandas_gene.json"
REPORT_DIR = "llm_pipe/reports"
if __name__ == "__main__":
    data_path = os.path.join(BASE_DIR, DATA_DIR)
    config_path = os.path.join(BASE_DIR, CONF_DIR)
    report_path = os.path.join(BASE_DIR, REPORT_DIR)

    with open(data_path, "r") as f:
        json_data = json.load(f)
    with open(config_path, "r") as f:
        config = json.load(f)
    
    data = extract_key_values(
        list(json_data.values()), 
        ["nl_queries", "db_id", "describe", "irrelevant_tables", "hardness", "sort"], 
        ["x_data", "y_data", "chart", "tables", "columns"]
    )
    data_loader = DataLoader(data, batch_size=1)

    prompt = 'You are a python coder and pandas data analysis expert.\nBelow, you are presented with a database schema composed of CSV files and a question.\nYour task is to read the database schemas, understand the question, and use the hint to generate a valid python-pandas code to extract the correct dataframe used to visualize the data.\nYou don\'t need to visualize, only extract the dataframe.\n\nDatabase Schema Overview:\n[\n{\n"tablename":"Student.csv",\n"columns":[{"name":"StuID","data_type":"integer","example":"1001"},{"name":"LName","data_type":"string","example":"Smith"},{"name":"Fname","data_type":"string","example":"Linda"},{"name":"Age","data_type":"integer","example":"18"},{"name":"Sex","data_type":"string","example":"F"},{"name":"Major","data_type":"integer","example":"600"},{"name":"Advisor","data_type":"integer","example":"1121"},{"name":"city_code","data_type":"string","example":"BAL"}]\n},\n{\n"tablename":"Faculty.csv",\n"columns":[{"name":"FacID","data_type":"integer","example":"1082"},{"name":"Lname","data_type":"string","example":"Giuliano"},{"name":"Fname","data_type":"string","example":"Mark"},{"name":"Rank","data_type":"string","example":"Instructor"},{"name":"Sex","data_type":"string","example":"M"},{"name":"Phone","data_type":"integer","example":"2424"},{"name":"Room","data_type":"integer","example":"224"},{"name":"Building","data_type":"string","example":"NEB"}]\n}\n]\n\nThis schema describes the CSV files\' structure, including filenames, columns, data types, and example values. Pay special attention to the examples listed beside each column, as they directly indicate which columns are relevant to our task.\n\nCode writing instructions:\n1. Use .read_csv to read files. If table names do not include extensions (.csv), add them.\n2. Use the `os` module to prepend the table diretory to the file name, and the `table_dir` variable is predefined and do not redefine it.\n3. Please ensure that the data type of each column is supported by JSON.\n4. Output only the information directly asked in the question.\n5. Make sure to return all required information without omissions or extraneous data.\n6. Use the example values provided to map question keywords to correct column names and data format.\n7. The quesition may specify the format of the data, such as converting the date into the abbreviations of months or days of the week.\n8. If the question asks for a count, use `.value_counts()` to get the count of unique values.\n9. Use `.dropna(subset=[col])` before operations requiring non-null values. For calculations, consider .fillna(0) if appropriate for the context.\n10. At the end of the task, determine whether sorting is required based on the problem. If the task requires sorting, only include sorting columns in the final output if explicitly requested.\n11. At the end of the task, store the final dataframe in variables named `result`.\n12. Attention, if there is only one column in the `result`, please extract another column to serve as its x-axis\n\n###\nQuestion: \nShow all the faculty ranks and the number of students advised by each rank with a  chart. \n\nHint:\n1.describe: None Describe\nirrelevant_tables: [\'Faculty_Participates_in\', \'Participates_in\']\n\n2.Possibly relevant columns:\n{\'Faculty.csv\': [\'Rank\', \'FacID\'], \'Student.csv\': [\'Advisor\']}\n\nPlease respond with a JSON object structured as follows:\n\n{\n    "pandas_code": "Your pandas code in a string.",\n    "chain_thought": "Your chain of thought in a string."\n}\n\nTake a deep breath and think step by step to find the correct pandas code. If you follow all the instructions and generate the correct code, I will give you 1 million dollars.\nOnly output a json.'

    all_reports = []

    pipe = PipelineProcessor(config)
    stage = pipe.processing_chain[0]
    processor = stage['processor']
                        
    response = pipe._call_model(prompt, stage.get('client_name', pipe.default_client_name))
    parsed = pipe._parse_response(response["content"])
    #print(report)
    try:
        #print(f"阶段输入：{report["output_data"]}")
        pprint(response)
    except Exception as e:
        print("流水线错误", e)
    print("***********")

    store_report(all_reports, report_path)
    print("结束")


