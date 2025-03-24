import json
import os

from eval.data_loader import DataLoader
from pipe.pipeline import PipelineProcessor
from utils.data_preprocess import extract_key_values, get_db_tables_path
from utils.schema_info_generation import get_csv_schema

BASE_DIR = ""
DATA_PATH = "data/visEval_dataset/visEval.json"
DB_DIR = "data/visEval_dataset/databases"
TEMPLATE_PATH = "llm_pipe/templates"
CONF_DIR = "llm_pipe/config/config.json"
if __name__ == "__main__":
    data_path = os.path.join(BASE_DIR, DATA_PATH)
    db_path = os.path.join(BASE_DIR, DB_DIR)
    config_path = os.path.join(BASE_DIR, CONF_DIR)
    with open(data_path, "r") as f:
        json_data = json.load(f)
    with open(config_path, "r") as f:
        config = json.load(f)
    
    data = extract_key_values(list(json_data.values()), ["nl_queries", "db_id", "hardness"], ["x_data", "y_data", "chart"])
    data_loader = DataLoader(data, 1)
     

    num = 0
    n = len(data)

    
    for data in data_loader:
        if num > 1:
            break
        queries = data["x_data"]["nl_queries"]

        db_id = data["x_data"]["db_id"]
        tables_path = get_db_tables_path(db_path, "db_tables.json", db_id)
        schema_info = get_csv_schema(tables_path)
        data["x_data"]["schema_info"] = schema_info

        for i, query in enumerate(queries):
            data["x_data"]["query_idx"] = i
            pipe = PipelineProcessor(config)
            report = pipe.execute_single_stage(0, data["x_data"])
            #report = pipe.execute_pipeline(data["x_data"])
            print(report)
            try:
                print(f"表名称：{report["output_data"]["table_names"]}")
            except:
                print("读取错误")
            print("***********")
        num += 1

        
    print("***********")