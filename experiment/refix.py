import json

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

    def are_values_equal(val1, val2, epsilon=1e-2):
        """判断两个值是否相等，处理数值型和字符串型的匹配"""
        if val1 == val2:
            return True
        
        # 处理数值型和字符串型的匹配（如"1"和1）
        try:
            float_val1 = float(val1)
            float_val2 = float(val2)
            # 使用容忍误差比较浮点数
            return abs(float_val1 - float_val2) < epsilon
        except (ValueError, TypeError):
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

res_dir = "experiment_res/evaluation_report.json"
with open(res_dir, "r", encoding="utf-8") as f:
    data = json.load(f)

res = data["results"]
c2r_count = 0
r2c_count = 0

for i in range(len(res)):
    try:
        eva = res[i]["evaluation"]["final"]
        nl_query = res[i]["input"]["nl_queries"][0]
        sort = False if res[i]["input"]["sort"] == None else True
        if i + 1 == 170:
            print("#")
        expected = eva["expected"]
        predicted = eva["predicted"]
        if not eva["correct"]:
            if are_nested_lists_equal(expected, predicted, sort):
                c2r_count += 1
                print(f"案例 {i+1} 预测正确，但被标记为错误")
                data["results"][i]["evaluation"]["final"]["correct"] = True
        else:
            if not are_nested_lists_equal(expected, predicted):
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

with open(res_dir.replace(".json", "_fixed.json"), "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)