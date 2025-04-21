from typing import List, Dict, Any, Optional
import re
import json
def parse_response(response: str) -> Dict:
    """增强的JSON响应解析方法
    
    实现功能：
    1. 清理非标准JSON标记
    2. 多回合JSON解析尝试
    3. 结构验证
    4. 错误信息定位
    
    返回结构示例：
    {
        "valid": True,
        "data": {...},  # 解析后的数据
        "errors": [],    # 错误列表
        "original": response  # 原始响应文本
    }
    """
    result_template = {
        "valid": False,
        "data": None,
        "errors": [],
        "original": response
    }

    # 阶段1：响应预处理
    cleaned = _clean_response_text(response)
    
    # 阶段2：解析尝试
    parsed = _safe_json_parse(cleaned, result_template)
    if not parsed:
        return result_template
    
    # 阶段3：结构验证
    return _validate_structure(parsed, result_template)

def _clean_response_text(text: str) -> str:
    """响应文本清理"""
    # 移除JSON代码块标记
    text = text.strip().replace("```json", "").replace("```", "")
    
    # 处理可能的BOM头
    if text.startswith("\ufeff"):
        text = text[1:]
    
    # 尝试提取JSON内容（删除JSON前后的额外文本）
    start_obj = text.find('{')
    start_arr = text.find('[')
    
    if start_obj != -1 or start_arr != -1:
        # 确定可能的起始位置
        if start_obj != -1 and (start_arr == -1 or start_obj < start_arr):
            # JSON对象在前
            start = start_obj
            end = text.rfind('}') + 1 if text.rfind('}') != -1 else len(text)
        else:
            # JSON数组在前
            start = start_arr
            end = text.rfind(']') + 1 if text.rfind(']') != -1 else len(text)
        
        if start < end:
            potential_json = text[start:end]
            try:
                # 尝试解析提取的内容
                json.loads(potential_json, strict=False)
                text = potential_json  # 如果是有效JSON，则使用提取的内容
            except json.JSONDecodeError:
                # 解析失败时保留原文本
                pass
    
    # 移除行末逗号（处理不标准JSON）
    text = re.sub(r',(\s*[}\]])', r'\1', text)
    return text.strip()

def _safe_json_parse(text: str, result: Dict) -> Optional[Dict]:
    """安全的JSON解析"""
    try:
        return json.loads(text, strict=False)
    except json.JSONDecodeError as e:
        # 尝试容错解析
        repaired = _try_repair_json(text)
        if repaired:
            try:
                return json.loads(repaired)
            except json.JSONDecodeError as e2:
                result["errors"].append(f"JSON修复失败: {str(e2)}")
        else:
            result["errors"].append(f"JSON解析错误: {str(e)}")
        return None

def _try_repair_json(text: str) -> Optional[str]:
    """尝试修复常见JSON格式错误"""
    original_text = text
    # 尝试1：包裹根层级为对象
    if not (text.startswith("{") and text.endswith("}")):
        try:
            repaired = "{" + text + "}"
            json.loads(repaired)
            return repaired
        except:
            pass
    
    # 尝试2：转义非法字符
    # text = re.sub(r'[\x00-\x1f\x7f-\x9f]', lambda m: f"\\\u{ord(m.group(0)):04x}", text)
    
    # 尝试3：处理单引号
    text = text.replace("'", '"')
    
    return text if text != original_text else None

def _validate_structure(data: Any, result: Dict) -> Dict:
    """验证数据结构"""
    
    # 检查根对象类型
    if not isinstance(data, dict):
        result["errors"].append("根元素非JSON对象")
        return result
    
    # 生成最终结果
    if not result["errors"]:
        result.update({
            "valid": True,
            "data": data
        })
    return result