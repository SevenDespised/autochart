import json
import os

json_path = "data/visEval_dataset/visEval.json"

#读取visEval.json数据
with open(json_path, "r") as f:
    data = json.load(f)

new_data = {}

# 删除"sql_part"键有重复的数据
unique_data = {}
for k, item in data.items():
    sql_part = item["vis_query"]["data_part"]["sql_part"]
    if sql_part not in unique_data:
        unique_data[sql_part] = item
        new_data[k] = item

# 保存新的json文件
with open("data/visEval_dataset/visEval_clear.json", "w") as f:
    json.dump(new_data, f, indent=4)
        
