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

if __name__ == "__main__":
    # 示例数据结构（包含嵌套字典和列表）
    test_data = [
        {
            "user": {
                "info": {
                    "name": "Alice",
                    "age": 30,
                    "contact": {
                        "email": "alice@example.com"
                    }
                }
            },
            "scores": [
                {"subject": "Math", "score": 95},
                {"subject": "Physics", "score": 88}
            ]
        },
        {
            "user": {
                "info": {
                    "name": "Bob",
                    "age": 25
                }
            },
            "scores": [
                {"subject": "Chemistry", "score": 92}
            ]
        },
        {
            "user": {
                "info": {
                    "name": "Charlie"
                }
            },
            "scores": []
        }
    ]

    # 定义要提取的多个x键和y键
    x_keys = ['name', 'age']
    y_keys = ['score', 'subject']

    # 调用函数提取数据
    result = extract_key_values(test_data, x_keys, y_keys)

    # 打印结果
    for idx, sample in enumerate(result, 1):
        print(f"Sample {idx}:")
        print(f"  x_data: {sample['x_data']}")
        print(f"  y_data: {sample['y_data']}\n")

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



