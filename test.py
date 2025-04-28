import pandas as pd
import json
from collections import Counter

# 读取数据
df = pd.read_table(
    "data/raw_data_all.csv",
    #on_bad_lines = "skip",
    #error_bad_lines=False,
    chunksize=1000,
    encoding='utf-8'
)
# 初始化类型统计
type_values = []
for i, chunk in enumerate(df):
    if i >= 50:  # 只处理前10个数据块
        break
    print(f"处理{(i+1) * chunk.shape[0]}行")

    # 分析每个chart_data中的类型
    for chart_data_str in chunk['chart_data']:
        try:
            # 解析JSON字符串
            chart_json = json.loads(chart_data_str)
            for data in chart_json:
                types = []
                types.append(data.get('type'))
                type_values.append(types)
            
        except Exception as e:
            print(f"解析错误: {e}")

# 统计并打印类型的分布
type_counts = Counter()
for types in type_values:
    for type_ in types:
        type_counts[type_] += 1

# 打印每种类型的数量
for type_, count in type_counts.items():
    print(f"类型: {type_}, 数量: {count}")

print(f"\n总计: 发现{len(type_counts)}种不同的图表类型")
