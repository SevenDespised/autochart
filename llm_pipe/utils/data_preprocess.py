def extract_key_values(data_list: list, target_x_keys: list, target_y_keys: list):
    """
    遍历数据集，提取每条数据中的多个 target_x_key 和多个 target_y_key 对应的值

    :param data_list: 包含多条数据的列表，每条数据是一个嵌套字典或列表
    :param target_x_keys: 需要提取的多个 x 键的列表
    :param target_y_keys: 需要提取的多个 y 键的列表
    :return: 包含所有有效数据对的字典列表，每个字典包含 x_data 和 y_data 键
    """
    extracted_res = []

    def find_key_values(data_item):
        """
        递归查找单个数据条目中的多个 target_x_key 和多个 target_y_key 值
        """
        x_values = {}
        y_values = {}

        def traverse_nested_structure(sub_item):
            nonlocal x_values, y_values
            if isinstance(sub_item, dict):
                for key, value in sub_item.items():
                    if key in target_x_keys and key not in x_values:
                        x_values[key] = value
                    elif key in target_y_keys and key not in y_values:
                        y_values[key] = value
                    # 提前终止条件
                    if len(x_values) == len(target_x_keys) and len(y_values) == len(target_y_keys):
                        return
                    traverse_nested_structure(value)
                    if len(x_values) == len(target_x_keys) and len(y_values) == len(target_y_keys):
                        return
            elif isinstance(sub_item, list):
                for element in sub_item:
                    if len(x_values) == len(target_x_keys) and len(y_values) == len(target_y_keys):
                        break
                    traverse_nested_structure(element)

        traverse_nested_structure(data_item)
        return {'x_data': x_values, 'y_data': y_values}

    extracted_res = [find_key_values(item) for item in data_list]
    return extracted_res

def get_db_tables_path(path, file_name, db_id) -> list:
    import json 
    import os
    with open(os.path.join(path, file_name), 'r') as f:
        file = json.load(f)
    return [os.path.join(path, db_id, x + ".csv") for x in file[db_id]]

def mask_chart_types(string: str):
    def case_insensitive_replace(text, old, new = ""):
        index = 0
        result = ""
        while index < len(text):
            if text[index:index + len(old)].lower() == old.lower():
                result += new
                index += len(old)
            else:
                result += text[index]
                index += 1
        return result
    chart_types = ['Grouping Scatter', 'Scatter', 'Pie', 'Line', 'Stacked Bar', 'Grouping Line', 'Bar']

    for chart_type in chart_types:
        tmp_string = case_insensitive_replace(string, chart_type)
        if tmp_string != string:
            string = tmp_string
    return string

def safe_json_serialize(obj):
    """
    将输入对象转换为JSON安全的数据结构
    
    处理以下JSON不支持的数据类型:
    - 非字符串键的字典
    - 集合(set)
    - 复数(complex)
    - 日期时间对象
    - 字节数据(bytes/bytearray)
    - 特殊浮点值(NaN, Infinity等)
    - 自定义对象
    
    返回:
        转换后的JSON安全数据
    """
    import datetime
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    
    elif isinstance(obj, (datetime.date, datetime.datetime)):
        return obj.isoformat()
    
    elif isinstance(obj, dict):
        # 处理字典，确保键是字符串
        return {str(key): safe_json_serialize(value) for key, value in obj.items()}
    
    elif isinstance(obj, list) or isinstance(obj, tuple):
        # 处理列表和元组
        return [safe_json_serialize(item) for item in obj]
    
    elif isinstance(obj, set):
        # 将集合转换为列表
        return [safe_json_serialize(item) for item in obj]
    
    elif isinstance(obj, complex):
        # 将复数转换为字符串表示
        return str(obj)
    
    
    elif isinstance(obj, bytes) or isinstance(obj, bytearray):
        # 将字节转换为base64编码的字符串
        import base64
        return base64.b64encode(obj).decode('ascii')
    
    elif isinstance(obj, float) and (obj != obj or obj == float('inf') or obj == float('-inf')):
        # 处理特殊浮点值：NaN, Infinity, -Infinity
        return str(obj)
    
    else:
        # 其他类型转为字符串
        return str(obj)



