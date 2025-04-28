import json
import pandas as pd

res_dir = "experiment_res/auseful_result/600bevaluation_report.json"
data_dir = "data/visEval_dataset/visEval_clear.json"
with open(res_dir, "r", encoding="utf-8") as f:
    res = json.load(f)

with open(data_dir, "r", encoding="utf-8") as f:
    data = json.load(f)

new_result = {}
for result in res["results"]:
    ipt = result["input"]
    nl_queries = ipt["nl_queries"]
    ir = ipt["irrelevant_tables"]
    flag = False
    for k, q in data.items():
        if q["nl_queries"] == nl_queries and q["irrelevant_tables"] == ir:
            new_result[k] = result
            flag = True
            break
    if not flag:
        print(f"not found {nl_queries[0]}")