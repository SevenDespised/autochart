import json
import os
from typing import List, Dict, Any, Union
import sys
from pathlib import Path

# 添加项目根目录到系统路径
current_dir = Path(__file__).parent
root_dir = current_dir.parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from llm_pipe.src.pipe.pipeline import PipelineProcessor


class Evaluator:
    """
    大模型输出评估器
    
    评估标准：所有预测的数据列表必须在真实值中找到匹配项，顺序无关紧要。
    即给定若干数据列表作为预测值，若干数据列表作为真实值，
    如果每个对应的数据列表都完全相等则判断为正确，否则判断为错误。
    """
    
    def __init__(self, pipeline_config: Dict = None):
        """初始化评估器"""
        self.pipeline = None
        if pipeline_config:
            self.pipeline = PipelineProcessor(pipeline_config)
        self.results = []
        
    def set_pipeline(self, pipeline_config: Dict):
        """设置流水线处理器"""
        self.pipeline = PipelineProcessor(pipeline_config)
    
    def evaluate_lists(self, predicted_lists: List[List], actual_lists: List[List]) -> bool:
        """
        评估预测列表和真实列表是否匹配
        
        Args:
            predicted_lists: 预测的数据列表
            actual_lists: 真实的数据列表
            
        Returns:
            bool: 评估结果，True表示匹配成功
        """
        # 首先检查列表数量是否相同
        if len(predicted_lists) != len(actual_lists):
            return False
        
        # 创建真实列表的副本，用于标记已匹配项
        remaining_actual = actual_lists.copy()
        
        # 检查每个预测列表是否都能在真实列表中找到匹配
        for pred_list in predicted_lists:
            found_match = False
            
            for i, act_list in enumerate(remaining_actual):
                if pred_list == act_list:
                    # 找到匹配，从剩余真实列表中移除
                    remaining_actual.pop(i)
                    found_match = True
                    break
            
            if not found_match:
                # 有预测列表未能匹配到任何真实列表
                return False
        
        # 所有预测列表都匹配成功
        return True
    
    def evaluate_pipeline(self, test_cases: List[Dict], config: Dict = None) -> Dict:
        """
        评估流水线处理器
        
        Args:
            test_cases: 测试用例列表，每个用例包含输入和期望输出
            config: 可选的流水线配置
            
        Returns:
            Dict: 评估报告
        """
        if config:
            self.set_pipeline(config)
        
        if not self.pipeline:
            raise ValueError("Pipeline not configured. Please set a pipeline configuration.")
        
        results = []
        correct_count = 0
        total_tokens = 0
        
        for i, test_case in enumerate(test_cases):
            input_data = test_case['input']
            expected_output = test_case['expected_output']
            
            # 执行流水线
            pipeline_result = self.pipeline.execute_pipeline(input_data)
            predicted_output = pipeline_result['output_data']
            
            # 评估结果
            is_correct = self.evaluate_lists(predicted_output, expected_output)
            if is_correct:
                correct_count += 1
            
            # 记录本次执行的token消耗
            tokens_used = pipeline_result.get('tokens', 0)
            total_tokens += tokens_used
            
            result = {
                'test_case_id': i,
                'input': input_data,
                'expected': expected_output,
                'predicted': predicted_output,
                'correct': is_correct,
                'execution_time': pipeline_result['execution_time'],
                'tokens': tokens_used
            }
            results.append(result)
        
        # 计算总体评估结果
        evaluation_report = {
            'total_cases': len(test_cases),
            'correct_cases': correct_count,
            'accuracy': correct_count / len(test_cases) if test_cases else 0,
            'total_tokens': total_tokens,
            'results': results
        }
        
        self.results = results
        return evaluation_report
    
    def evaluate_single(self, predicted_lists: List[List], actual_lists: List[List]) -> bool:
        """评估单个预测结果和实际结果"""
        return self.evaluate_lists(predicted_lists, actual_lists)
    
    def get_statistics(self) -> Dict:
        """获取评估统计信息"""
        if not self.results:
            return {"error": "No evaluation results available"}
        
        total = len(self.results)
        correct = sum(1 for result in self.results if result['correct'])
        total_tokens = sum(result.get('tokens', 0) for result in self.results)
        
        return {
            'total_cases': total,
            'correct_cases': correct,
            'accuracy': correct / total if total > 0 else 0,
            'total_tokens': total_tokens
        }