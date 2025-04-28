import json
import os

res_dir = "experiment_res/useful_result/600bevaluation_report.json"
with open(res_dir, "r", encoding="utf-8") as f:
    data = json.load(f)

res = data["results"]
changed_count = 0
total_incorrect = 0
processed_count = 0  # 已处理案例计数

# 定义保存函数，避免代码重复
def save_current_progress(is_final=False):
    output_file = res_dir.replace(".json", "_manual.json")

    correct_case = sum(1 for res_item in data["results"] if res_item.get("evaluation", {}).get("final", {}).get("correct", False))
    data["evaluation"]["final"]["correct_cases"] = correct_case
    data["evaluation"]["final"]["accuracy"] = correct_case / data["total_cases"]
    data["correct_cases"] = correct_case
    data["accuracy"] = correct_case / data["total_cases"]
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    if not is_final:
        print(f"\n已自动保存当前进度，已处理 {processed_count} 个案例")
    else:
        print(f"更新后的结果已保存至: {output_file}")

# 定义纠正原因映射
correction_reasons = {
    "0": "abnormal_gt",
    "1": "format",
    "2": "decimal",
    "3": "merge"
}

for i in range(len(res)):
    eva = res[i]["evaluation"]["final"]
    try:
        if not eva["correct"]:
            total_incorrect += 1
            nl_query = res[i]["input"]["nl_queries"][0]
            expected = eva["expected"]
            predicted = eva["predicted"]
            
            # 打印信息供人工判断
            print(f"\n案例 {i+1}/{len(res)}:")
            print(f"查询: {nl_query}")
            print(f"预期结果: {expected}")
            print(f"当前结果结果: {predicted}")
            
            user_input = input("请输入选择: ")
            if user_input in correction_reasons:
                data["results"][i]["evaluation"]["final"]["correct"] = True
                data["results"][i]["evaluation"]["final"]["correction_reason"] = correction_reasons[user_input]
                changed_count += 1
                print(f"已将此案例标记为正确，纠正原因: {correction_reasons[user_input]}")
            else:
                print("保持此案例标记为不正确")
            data["results"][i]["evaluation"]["final"]["manual"] = True
            processed_count += 1
            
            # 每处理10个案例自动保存一次
            if processed_count % 10 == 0:
                save_current_progress()

    except Exception as e:
        print(f"处理案例 {i+1} 时发生错误: {e}")

# 输出统计信息
print(f"\n已完成人工审核。共有{total_incorrect}个不正确案例，已更改{changed_count}个案例的状态。")

# 保存最终结果
save_current_progress(is_final=True)