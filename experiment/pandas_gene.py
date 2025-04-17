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
#CONF_DIR = "llm_pipe/config/experiment_ds_pandas_gene.json"
CONF_DIR = "llm_pipe/config/experiment_pandas_gene.json"
REPORT_DIR = "experiment_res/evaluation_report.json"
EXPERIMENT_DIR = "experiment_res/"

def process_batch(batch, config, save_path, batch_num = 0):
    """处理一批测试用例"""
    evaluator = Evaluator(config)
    batch_results = evaluator.evaluate_pipeline(batch, save_path=save_path, save_interval=10, batch_num=batch_num)
    return batch_results

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
    test_cases = []
    limit = LIMIT
    
    for i, item in enumerate(data):
        if i >= limit and limit >= 0:
            break
            
        # 构建测试用例
        test_case = {
            'input': item["x_data"],
            'expected_output': item["y_data"]
        }
        test_cases.append(test_case)
    
    print(f"准备评估 {len(test_cases)} 条测试用例")
    
    # 并发处理设置
    num_processes = NUM_PROCESSES  # 获取CPU核心数
    print(f"使用 {num_processes} 个并行进程进行评估")
    
    # 将测试用例分成多个批次
    batch_size = max(1, len(test_cases) // num_processes)
    batches = [test_cases[i:i+batch_size] for i in range(0, len(test_cases), batch_size)]
    
    start_time = time.time()
    
    # 创建进程池
    with mp.Pool(processes=num_processes) as pool:
        # 时间戳路径
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        res_path = os.path.join(EXPERIMENT_DIR, f"mp_{timestamp}")
        # 创建保存路径
        os.makedirs(res_path, exist_ok=True)
        # 准备带有批次编号的参数
        batch_args = [(batch, config, res_path, i) for i, batch in enumerate(batches)]
        
        # 并行执行
        print("开始并行评估...")
        results = pool.starmap(process_batch, batch_args)
    
    # 合并结果
    evaluation_report = {
        'total_cases': 0,
        'evaluation': {
            "final":{
                'correct_cases': 0,
                'accuracy': 0,
            },
            "table": {
                'correct_cases': 0,
                'accuracy': 0,
            },
            "column": {
                'correct_cases': 0,
                'accuracy': 0,
            }
        },
        'correct_cases': 0,
        'accuracy': 0,
        'total_tokens': 0,
        'results': []
    }   
    
    for batch_result in results:
        # 合并评估报告
        evaluation_report['results'].extend(batch_result['results'])
        evaluation_report['total_cases'] += batch_result['total_cases']
        evaluation_report['correct_cases'] += batch_result['correct_cases']
        evaluation_report['total_tokens'] += batch_result['total_tokens']
        evaluation_report['evaluation']['final']['correct_cases'] += batch_result['evaluation']['final']['correct_cases']
        evaluation_report['evaluation']['table']['correct_cases'] += batch_result['evaluation']['table']['correct_cases']
        evaluation_report['evaluation']['column']['correct_cases'] += batch_result['evaluation']['column']['correct_cases']

    # 计算准确率
    if evaluation_report['total_cases'] > 0:
        evaluation_report['accuracy'] = evaluation_report['correct_cases'] / evaluation_report['total_cases']
        evaluation_report['evaluation']['final']['accuracy'] = evaluation_report['evaluation']['final']['correct_cases'] / evaluation_report['total_cases']
        evaluation_report['evaluation']['table']['accuracy'] = evaluation_report['evaluation']['table']['correct_cases'] / evaluation_report['total_cases']
        evaluation_report['evaluation']['column']['accuracy'] = evaluation_report['evaluation']['column']['correct_cases'] / evaluation_report['total_cases']
    
    end_time = time.time()
    print(f"并行评估完成，耗时: {end_time - start_time:.2f} 秒")
    
    # 保存评估报告
    evaluation_report = safe_json_serialize(evaluation_report)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(evaluation_report, f, indent=4)
    
    # 打印评估结果
    print(f"\n详细评估报告已保存至: {report_path}")

if __name__ == "__main__":
    # 避免Windows下的递归创建子进程问题
    mp.freeze_support()
    main()