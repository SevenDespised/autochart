from llm_pipe.utils.data_preprocess import safe_json_serialize
import json
import time
import pandas as pd

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
        if num_a_truncated == num_b_truncated:
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
        "thu": 4, "fri": 5, "sat": 6, "sun": 7
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

print(are_values_equal("1.0", 1))  # True
print(are_values_equal("1.00", 1))  # True
print(are_values_equal("feb", 2))  # True
print(are_values_equal("3.37", 3.3))  # True

# 基本相等测试
print(are_values_equal(100, 100))  # True
print(are_values_equal("hello", "hello"))  # True
print(are_values_equal(None, None))  # True

# 数值型字符串与数字比较
print(are_values_equal("42", 42))  # True
print(are_values_equal(42, "42.0"))  # True
print(are_values_equal("42", 43))  # False

# 浮点数比较
print(are_values_equal(3.1415, 3.14))  # True
print(are_values_equal("3.140", "3.14"))  # True
print(are_values_equal("3.141", 3.142))  # False

# 星期表示方式
print(are_values_equal("monday", 1))  # True
print(are_values_equal("Monday", "mon"))  # True
print(are_values_equal("friday", "5"))  # True
print(are_values_equal("monday", "tuesday"))  # False

# 月份表示方式
print(are_values_equal("january", 1))  # True
print(are_values_equal("JAN", "january"))  # True
print(are_values_equal(7, "july"))  # True
print(are_values_equal("december", "12"))  # True

# 边界情况
print(are_values_equal("", ""))  # True
print(are_values_equal("", 0))  # False
print(are_values_equal("abc", 123))  # False
print(are_values_equal(8, "monday"))  # False
print(are_values_equal(13, "january"))  # False