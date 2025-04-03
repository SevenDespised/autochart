from collections import Counter

def are_values_equal(val1, val2):
    """判断两个值是否相等，处理数值型和字符串型的匹配"""
    if val1 == val2:
        return True
    
    # 处理数值型和字符串型的匹配（如"1"和1）
    try:
        return float(val1) == float(val2)
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
    
    # 不考虑顺序时，使用计数比较法
    # 由于需要特殊处理数值和字符串匹配，不能直接使用Counter
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

def are_nested_lists_equal(nested_list1, nested_list2, consider_order=False):
    """
    判断两个列表列表是否相等
    
    Args:
        nested_list1: 第一个列表列表
        nested_list2: 第二个列表列表  
        consider_order: 是否考虑内部列表元素顺序
    """
    if len(nested_list1) != len(nested_list2):
        return False
    
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

if __name__ == "__main__":
    # 测试值相等
    print(are_values_equal("1", 1))  # True
    
    # 测试列表相等（不考虑顺序）
    print(are_lists_equal([1, "2", 3], ["2", 3, 1]))  # True
    
    # 测试列表相等（考虑顺序）
    print(are_lists_equal([1, "2", 3], [1, "2", 3], True))  # True
    print(are_lists_equal([1, "2", 3], ["2", 3, 1], True))  # False
    
    # 测试列表列表相等
    list1 = [[1, 2], [3, "4"]]
    list2 = [[2, 1], ["4", 3]]
    print(are_nested_lists_equal(list1, list2))  # True
    
    # 考虑内部列表顺序的情况
    print(are_nested_lists_equal(list1, list2, True))  # False
     
    a = None
    is_sorted = False if not a else True