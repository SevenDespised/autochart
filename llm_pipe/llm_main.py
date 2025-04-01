import json
import os
from pprint import pprint

from eval.data_loader import DataLoader
from src.pipe.pipeline import PipelineProcessor
from utils.data_preprocess import extract_key_values
from utils.report_process import store_report

BASE_DIR = ""
DATA_DIR = "data/visEval_dataset/visEval.json"
CONF_DIR = "llm_pipe/config/hw_deepseek_v3.json"
CONF_DIR = "llm_pipe/config/config.json"
REPORT_DIR = "llm_pipe/reports"
if __name__ == "__main__":
    data_path = os.path.join(BASE_DIR, DATA_DIR)
    config_path = os.path.join(BASE_DIR, CONF_DIR)
    report_path = os.path.join(BASE_DIR, REPORT_DIR)

    with open(data_path, "r") as f:
        json_data = json.load(f)
    with open(config_path, "r") as f:
        config = json.load(f)
    
    data = extract_key_values(list(json_data.values()), ["nl_queries", "db_id", "hardness"], ["x_data", "y_data", "chart"])
    data_loader = DataLoader(data, batch_size=1)

    num = 1
    n = len(data)
    all_reports = []
    for data in data_loader:
        #if num > 5000:
        #    break
        if data["x_data"]["nl_queries"][0] != "A bar chart listing the number of faults for different description of skills required to fix them, and sort x axis in descending order.":
            num += 1
            continue
        else:
            print(f"已找到。第{num}个样本")
        pipe = PipelineProcessor(config)
        processor = pipe.processing_chain[1]['processor']
        #print(processor.generate_prompt(data["x_data"]))
        report = pipe.execute_pipeline(data["x_data"])
        all_reports.append(report)
        #print(report)
        try:
            #print(f"阶段输入：{report["output_data"]}")
            pprint(report)
        except Exception as e:
            print("流水线错误", e)
        print("***********")
        num += 1
    store_report(all_reports, report_path)
    print("结束")


