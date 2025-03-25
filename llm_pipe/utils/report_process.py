import json
import os
from datetime import datetime

def store_report(report, path = ""):
    timestamp = datetime.now().strftime("%m-%d-%H-%M-%S")
    model_name = report[0]["model_name"]
    # 构造文件名
    filename = f"report_{timestamp}_{model_name}.json"
    full_path = os.path.join(path, filename)

    # 写入文件
    with open(full_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"文件已保存：{full_path}")