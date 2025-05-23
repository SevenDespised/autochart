import json
import pandas as pd

def merge_data(x_data, y_data, classify):
    """
    将x_data、y_data和classify合并成一个DataFrame
    
    参数:
    x_data -- 包含单位的列表的列表，例如：[["MW", "PW", "TW"]]
    y_data -- 包含值的列表的列表，例如：[[0, 3, 0], [1, 2, 2], [4, 2, 1]]
    classify -- 类别列表，例如：["Electrical", "Mechanical", "TV, Video"]
    
    返回:
    包含单位、值和类别的DataFrame
    """
    # 创建空列表存储结果
    res = [[], [], []]
    units = res[0]
    values = res[1]
    categories = res[2]
    
    # 处理数据
    for category_index, category in enumerate(classify):
        for unit_index, unit in enumerate(x_data[0]):
            units.append(unit)
            values.append(y_data[category_index][unit_index])
            categories.append(category)

    return res

def are_nested_lists_equal(nested_list1, nested_list2, consider_order=False):
    """
    判断两个列表列表是否相等
    
    Args:
        nested_list1: 第一个列表列表
        nested_list2: 第二个列表列表  
        consider_order: 是否考虑内部列表元素顺序
    """
    if len(nested_list1) > len(nested_list2):
        return False

    def are_values_equal(val1, val2):
        """判断两个值是否相等，处理数值型和字符串型的匹配"""
        if val1 == val2:
            return True
        # 处理数字
        if isinstance(val1, str):
            try:
                val1 = int(val1)
            except ValueError:
                try:
                    val1 = float(val1)
                except ValueError:
                    ...
        if isinstance(val2, str):
            try:
                val2 = int(val2)
            except ValueError:
                try:
                    val2 = float(val2)
                except ValueError:
                    ...
        if isinstance(val1, int) or isinstance(val2, int):
            try:
                if int(val1) == int(val2):
                    return True
            except ValueError:
                ...
        # 处理浮点型匹配，将小数位数较多的数字截断到较少的位数
        try:
            # 转换为字符串
            str_a = str(val1)
            str_b = str(val2)
            
            # 分离整数部分和小数部分
            if '.' in str_a:
                int_a, dec_a = str_a.split('.')
            else:
                int_a, dec_a = str_a, ''
            
            if '.' in str_b:
                int_b, dec_b = str_b.split('.')
            else:
                int_b, dec_b = str_b, ''
            
            # 确定较短的小数位数
            min_decimal_places = min(len(dec_a), len(dec_b))
            
            # 截断小数部分到相同位数
            dec_a_truncated = dec_a[:min_decimal_places]
            dec_b_truncated = dec_b[:min_decimal_places]
            
            # 重新组合数字
            if min_decimal_places > 0:
                num_a_truncated = float(f"{int_a}.{dec_a_truncated}")
                num_b_truncated = float(f"{int_b}.{dec_b_truncated}")
            else:
                num_a_truncated = float(int_a)
                num_b_truncated = float(int_b)
            
            # 带有容忍度的判断
            if abs(num_a_truncated - num_b_truncated) < 1e-2:
                return True
        except (ValueError, TypeError):
            ...
        
        # 构建星期映射关系
        weekday_mapping = {
            # 全称到数字的映射
            "monday": 1, "tuesday": 2, "wednesday": 3, 
            "thursday": 4, "friday": 5, "saturday": 6, "sunday": 7,
            # 缩写到数字的映射
            "mon": 1, "tue": 2, "wed": 3, 
            "thu": 4, "fri": 5, "sat": 6, "sun": 7,
            "thur": 4
        }
        
        # 构建月份映射关系
        month_mapping = {
            # 全称到数字的映射
            "january": 1, "february": 2, "march": 3, "april": 4,
            "may": 5, "june": 6, "july": 7, "august": 8,
            "september": 9, "october": 10, "november": 11, "december": 12,
            # 缩写到数字的映射
            "jan": 1, "feb": 2, "mar": 3, "apr": 4,
            "may": 5, "jun": 6, "jul": 7, "aug": 8,
            "sep": 9, "oct": 10, "nov": 11, "dec": 12
        }
        
        # 将输入转换为字符串并小写化
        str_val1 = str(val1).lower() if val1 is not None else ""
        str_val2 = str(val2).lower() if val2 is not None else ""
        
        # 检查是否为星期的不同表示
        weekday1 = weekday_mapping.get(str_val1)
        weekday2 = weekday_mapping.get(str_val2)
        
        # 处理数字形式的星期
        if weekday1 is None and str_val1.isdigit() and 1 <= int(str_val1) <= 7:
            weekday1 = int(str_val1)
        if weekday2 is None and str_val2.isdigit() and 1 <= int(str_val2) <= 7:
            weekday2 = int(str_val2)
            
        if weekday1 is not None and weekday2 is not None:
            return weekday1 == weekday2
        
        # 检查是否为月份的不同表示
        month1 = month_mapping.get(str_val1)
        month2 = month_mapping.get(str_val2)
        
        # 处理数字形式的月份
        if month1 is None and str_val1.isdigit() and 1 <= int(str_val1) <= 12:
            month1 = int(str_val1)
        if month2 is None and str_val2.isdigit() and 1 <= int(str_val2) <= 12:
            month2 = int(str_val2)
            
        if month1 is not None and month2 is not None:
            return month1 == month2
        return False


    def are_lists_equal(list1, list2, consider_order=False):
        """
        判断两个值列表是否相等
        
        Args:
            list1: 第一个列表
            list2: 第二个列表
            consider_order: 是否考虑元素顺序
        """
        if len(list1) != len(list2):
            return False
        # 考虑顺序时，逐个元素比较
        if consider_order:
            return all(are_values_equal(a, b) for a, b in zip(list1, list2))
        
        items2 = list(list2)
        for item1 in list1:
            found = False
            for i, item2 in enumerate(items2):
                if are_values_equal(item1, item2):
                    items2.pop(i)
                    found = True
                    break
            if not found:
                return False
        return True
    
    indexes_to_remove = set()
    for sublist in nested_list1:
        indexes_to_remove.update(idx for idx, item in enumerate(sublist) if item == 0)

    result = []
    for sublist in nested_list1:
        filtered_sublist = [item for idx, item in enumerate(sublist) if idx not in indexes_to_remove]
        result.append(filtered_sublist)
    nested_list1 = result
    
    indexes_to_remove = set()
    for sublist in nested_list2:
        indexes_to_remove.update(idx for idx, item in enumerate(sublist) if item == 0)

    result = []
    for sublist in nested_list2:
        filtered_sublist = [item for idx, item in enumerate(sublist) if idx not in indexes_to_remove]
        result.append(filtered_sublist)
    nested_list2 = result

    # 列表列表元素无序，需要两两匹配
    remaining = list(nested_list2)
    for sublist1 in nested_list1:
        found = False
        for i, sublist2 in enumerate(remaining):
            if are_lists_equal(sublist1, sublist2, consider_order):
                remaining.pop(i)
                found = True
                break
        if not found:
            return False
    return True

res_dir = "experiment_res/auseful_result/607bevaluation_report.json"
data_dir = "data/visEval_dataset/visEval_clear.json"
with open(res_dir, "r", encoding="utf-8") as f:
    data = json.load(f)
with open(data_dir, "r", encoding="utf-8") as f:
    dataset = json.load(f)

res = data["results"]
c2r_count = 0
r2c_count = 0

for i in range(len(res)):
    try:
        if i + 1 == 101:
            print("")
        data_id = res[i]["data_id"]
        eva = res[i]["evaluation"]["final"]
        nl_query = res[i]["input"]["nl_queries"][0]
        sort = False if res[i]["input"]["sort"] is None else True
        sort = False
        expected = eva["expected"]
        predicted = eva["predicted"]
        if not eva["correct"]:
            if are_nested_lists_equal(expected, predicted, sort):
                c2r_count += 1
                print(f"案例 {i+1} 预测正确，但被标记为错误")
                data["results"][i]["evaluation"]["final"]["correct"] = True
            else:
                if dataset[data_id]["vis_obj"]["classify"]:
                    x = dataset[data_id]["vis_obj"]["x_data"]
                    y = dataset[data_id]["vis_obj"]["y_data"]
                    classify = dataset[data_id]["vis_obj"]["classify"]
                    merged_data = merge_data(x, y, classify)
                    if are_nested_lists_equal(merged_data, predicted, sort):
                        print(f"案例 {i+1} 预测正确，但因嵌套被标记为错误")
                        data["results"][i]["evaluation"]["final"]["correct"] = True

        else:
            if not are_nested_lists_equal(expected, predicted, sort):
                r2c_count += 1
                print(f"案例 {i+1} 预测错误，但被标记为正确")
                data["results"][i]["evaluation"]["final"]["correct"] = False
                print(f"案例 {i+1} 的输入: {nl_query}")
                print(f"案例 {i+1} 的预期: {expected}")
                print(f"案例 {i+1} 的预测: {predicted}")
    except Exception as e:
        print(f"处理案例 {i+1} 时发生错误: {e}")

print(f"标记为错误但实际正确的案例: {c2r_count} 个")
print(f"标记为正确但实际错误的案例: {r2c_count} 个")

# 重新统计
correct_case = sum(1 for res_item in res if res_item.get("evaluation", {}).get("final", {}).get("correct", False))
data["evaluation"]["final"]["correct_cases"] = correct_case
data["evaluation"]["final"]["accuracy"] = correct_case / data["total_cases"]
data["correct_cases"] = correct_case
data["accuracy"] = correct_case / data["total_cases"]

with open(res_dir.replace(".json", "_reeval.json"), "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)