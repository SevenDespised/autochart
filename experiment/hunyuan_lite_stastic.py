import os
import json
import glob

from collections import Counter

def read_batch_results(directory="experiment_res/eval_results_20250407_145404"):
    # 规范化路径
    directory = os.path.normpath(directory)
    
    # 构建文件搜索模式
    file_pattern = os.path.join(directory, "results_batch_*.json")
    
    # 查找所有匹配的文件
    json_files = glob.glob(file_pattern)
    
    results = []
    
    # 读取每个文件
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                results.append({
                    'file_name': os.path.basename(file_path),
                    'content': data
                })
        except Exception as e:
            print(f"读取文件 {file_path} 出错: {e}")
    
    print(f"共读取了 {len(results)} 个JSON文件")
    return results

if __name__ == "__main__":
    # 调用函数读取文件
    all_results = read_batch_results()

    total = 0
    correct = 0
    hardness_counter = Counter()
    hardness_correct_counter = Counter()
    for result in all_results:
        content = result['content']
        for batch_result in content["batch_results"]:
            if batch_result['status'] == 'success':
                total += 1
                if batch_result['correct']:
                    correct += 1
                    hardness_correct_counter[batch_result["input"]['hardness']] += 1
                hardness_counter[batch_result["input"]['hardness']] += 1
    
    print(f"Total: {total}, Correct: {correct}, Accuracy: {correct / total}")
    print(f"Hardness distribution: {hardness_counter}")
    print(f"Hardness correct distribution: {hardness_correct_counter}")
    print(f"Hardness accuracy distribution: {dict((k, hardness_correct_counter[k] / hardness_counter[k]) for k in hardness_counter)}")
