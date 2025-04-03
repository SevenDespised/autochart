from llm_pipe.utils.data_preprocess import safe_json_serialize
import json
import time
import pandas as pd
# 建立一个pandas period类型的列表
period_list = [[pd.Period('2023-01-01')], pd.Period('2023-02-01'), pd.Period('2023-03-01')]

# 将period_list转换为JSON安全的数据结构
safe_period_list = safe_json_serialize(period_list)  
print(safe_period_list)
