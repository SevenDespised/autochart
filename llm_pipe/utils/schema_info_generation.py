import pandas as pd
import os
import json
from typing import List, Dict

def get_csv_schema(file_paths: List[str], delimiter: str = ',', encoding: str = 'utf-8') -> str:
    """
    提取多个CSV文件的元数据信息并返回格式化JSON字符串
    """
    schemas = []
    
    for file_path in file_paths:
        if not file_path.lower().endswith('.csv'):
            continue
        
        try:
            # 读取前5行并保持原始类型
            df = pd.read_csv(
                file_path,
                delimiter=delimiter,
                encoding=encoding,
                nrows=5,
                on_bad_lines='skip',
                dtype='string'  # 禁用自动类型推断
            )
            
            schema = {
                "filename": os.path.basename(file_path),
                "columns": []
            }

            for col in df.columns:
                series = df[col].dropna()
                if series.empty:
                    continue  # 跳过完全空的列

                # 检测数据类型
                dtype = 'unknown'
                first_value = series.iloc[0]
                
                try:
                    pd.to_datetime(first_value, errors='raise')
                    dtype = 'datetime'
                except:
                    try:
                        float(first_value)
                        if '.' in first_value:
                            dtype = 'float'
                        else:
                            dtype = 'integer'
                    except:
                        dtype = 'string'

                # 保持示例值原样（不修改时间格式）
                example = str(first_value)  # 直接转换为字符串
                
                schema["columns"].append({
                    "name": col,
                    "data_type": dtype,
                    "example": example
                })
            
            schemas.append(schema)
        
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
    
    # 生成格式化JSON字符串
    return json.dumps(schemas, indent=2, ensure_ascii=False)


# 使用示例
if __name__ == "__main__":
    # 配置参数
    CSV_FILES = [
        "data/visEval_dataset/databases/activity_1/Faculty.csv",
        "data/visEval_dataset/databases/activity_1/Activity.csv"
    ]
    DELIMITER = ','
    ENCODING = 'utf-8'

    # 处理文件
    csv_schemas = get_csv_schema(CSV_FILES, DELIMITER, ENCODING)
    
    # 输出JSON格式结果

    print(csv_schemas)