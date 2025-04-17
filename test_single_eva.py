import json
import os
import sys
from pprint import pprint
from pathlib import Path
import multiprocessing as mp
from functools import partial
import time

current_dir = Path(__file__).parent
root_dir = current_dir.parent
sys.path.insert(0, str(root_dir))

from llm_pipe.eval.pandas_evaluator import Evaluator
from llm_pipe.utils.data_preprocess import extract_key_values
from llm_pipe.utils.data_preprocess import safe_json_serialize

# 设置并行处理的进程数
NUM_PROCESSES = 5
# 设置评估的测试用例数量
LIMIT = -1
# 配置路径
BASE_DIR = ""
DATA_DIR = "data/visEval_dataset/visEval_with_tables_columns.json"
#DATA_DIR = "data/visEval_dataset/visEval.json"
CONF_DIR = "llm_pipe/config/experiment_pandas_gene.json"
REPORT_DIR = "experiment_res/evaluation_report.json"
EXPERIMENT_DIR = "experiment_res/"


def main():
    # 加载数据和配置
    data_path = os.path.join(BASE_DIR, DATA_DIR)
    config_path = os.path.join(BASE_DIR, CONF_DIR)
    report_path = os.path.join(BASE_DIR, REPORT_DIR)

    print(f"加载数据: {data_path}")
    print(f"加载配置: {config_path}")

    with open(data_path, "r") as f:
        json_data = json.load(f)
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # 提取数据
    data = extract_key_values(
        list(json_data.values()), 
        ["nl_queries", "db_id", "describe", "irrelevant_tables", "hardness", "sort"], 
        ["x_data", "y_data", "chart", "tables", "columns"]
    )

    # 准备测试用例
    
    for i, item in enumerate(data):
        if item["x_data"]["nl_queries"][0] == "How many dogs departed in each day? Visualize with a bar chart that groups by departed date.":
            print("find it")
                
            # 构建测试用例
            test_case = {
                'input': item["x_data"],
                'expected_output': item["y_data"]
            }
            evaluator = Evaluator(config)
            res = evaluator.evaluate_single(test_case=test_case)
            print(res)
            break
    
    

if __name__ == "__main__":
    # 避免Windows下的递归创建子进程问题

    main()