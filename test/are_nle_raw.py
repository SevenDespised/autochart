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

expected = [['bangla', 'english'], [7.5, 7]]
predicted = [['bangla', 'english'], [7.5, 7.0], [4, 2]]
if are_nested_lists_equal(expected, predicted, consider_order=True):
    print("相等")
else:
    print("不相等")