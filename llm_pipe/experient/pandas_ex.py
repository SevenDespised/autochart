import json
import os
import sys
from pprint import pprint
from pathlib import Path

from llm_pipe.eval.evaluator import Evaluator
from eval.data_loader import DataLoader
from utils.data_preprocess import extract_key_values

# 配置路径
BASE_DIR = ""
DATA_DIR = "data/visEval_dataset/visEval.json"
CONF_DIR = "llm_pipe/config/config.json"
REPORT_DIR = "llm_pipe/reports/evaluation_report.json"

def main():
    # 加载数据和配置
    data_path = os.path.join(BASE_DIR, DATA_DIR)
    config_path = os.path.join(BASE_DIR, CONF_DIR)
    report_path = os.path.join(BASE_DIR, REPORT_DIR)

    print(f"加载数据: {data_path}")
    print(f"加载配置: {config_path}")

    try:
        with open(data_path, "r") as f:
            json_data = json.load(f)
        with open(config_path, "r") as f:
            config = json.load(f)
        
        # 提取数据
        data = extract_key_values(
            list(json_data.values()), 
            ["nl_queries", "db_id", "hardness"], 
            ["x_data", "y_data", "chart"]
        )
        
        # 创建评估器
        evaluator = Evaluator(config)
        
        # 准备测试用例 - 默认评估前10个样本
        test_cases = []
        limit = 10
        
        for i, item in enumerate(data):
            if i >= limit:
                break
                
            # 构建测试用例
            test_case = {
                'input': item["x_data"],
                'expected_output': item["y_data"]  # y_data作为期望输出
            }
            test_cases.append(test_case)
        
        print(f"准备评估 {len(test_cases)} 条测试用例")
        
        # 运行评估
        print("开始评估...")
        evaluation_report = evaluator.evaluate_pipeline(test_cases)
        
        # 保存评估报告
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(evaluation_report, f, indent=4)
        
        # 打印评估结果
        print("\n评估结果统计:")
        pprint(evaluator.get_statistics())
        print(f"\n详细评估报告已保存至: {report_path}")
        
    except Exception as e:
        print(f"评估过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()