import os
import sys
import datetime
import json
from typing import List, Dict, Any, Union
from pathlib import Path

from ..utils.data_preprocess import safe_json_serialize
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
        self.reports = []
        
    def set_pipeline(self, pipeline_config: Dict):
        """设置流水线处理器"""
        self.pipeline = PipelineProcessor(pipeline_config)
    
    def evaluate_pipeline(self, test_cases: List[Dict], config: Dict = None, save_interval: int = 10, save_path: str = None, batch_num: int = -1) -> Dict:
        """
        评估流水线处理器
        
        Args:
            test_cases: 测试用例列表，每个用例包含输入和期望输出
            config: 可选的流水线配置
            save_interval: 每处理多少条数据保存一次中间结果，默认为10
            
        Returns:
            Dict: 评估报告
        """
        if config:
            self.set_pipeline(config)
        
        if not self.pipeline:
            raise ValueError("Pipeline not configured. Please set a pipeline configuration.")
        
        # 如果未建立多进程，直接创建结果保存目录
        if batch_num == -1:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.results_dir = os.path.join(save_path, f"eval_results_{timestamp}")
        else:
            # 如果是多进程，使用传入的目录路径
            self.results_dir = os.path.join(save_path, f"eval_results_{batch_num}")
        os.makedirs(self.results_dir, exist_ok=True)
        
        success_count = 0
        correct_count = 0
        hit2_count = 0
        total_rank = 0
        total_tokens = 0
        n = len(test_cases)
        for i, test_case in enumerate(test_cases):
            try:
                print(f"[{i + 1}/{n}] 正在评估测试用例...")
                input_data = test_case['input']
                expected_output = test_case['expected_output']

                #####
                #if "Show me about the distribution of date_address_to and the amount of date_address_to, and group by attribute other_details and bin date_address_to by month in a bar chart." not in input_data["nl_queries"]:
                #    print(f"跳过测试用例 {i + 1}，输入数据不符合预期格式。")
                #    continue
                #####
                
                # 执行流水线
                pipeline_report = self.pipeline.execute_pipeline(input_data)
                pipeline_report['test_case_id'] = i
                self.reports.append(pipeline_report)
                predicted_output = pipeline_report['output_data'].get('type', [])

                # 评估结果
                metrics = self.get_metrics(predicted_output, expected_output)
                if metrics['is_correct']:
                    correct_count += 1
                if metrics['is_correct_hit2']:
                    hit2_count += 1
                total_rank += metrics['rank']

                # 记录本次执行的token消耗
                tokens_used = pipeline_report.get('tokens', 0)
                total_tokens += tokens_used
                if pipeline_report["success"]:
                    success_count += 1
                    result = {
                        'status': 'success',
                        'batch_num': batch_num,
                        'test_case_id': i,
                        'input': input_data,
                        'expected': expected_output,
                        'predicted': predicted_output,
                        'correct': metrics['is_correct'],
                        'correct_hit2': metrics['is_correct_hit2'],
                        'rank': metrics['rank'],
                        'execution_time': pipeline_report['execution_time'],
                        'tokens': tokens_used
                    }
                else:
                    result = {
                        'status': 'failure',
                        'test_case_id': i,
                    }
                self.results.append(result)
            except Exception as e:
                print(f"处理测试批次 {batch_num} 用例 {i} 时发生错误: {e}")
                result = {
                    'status': 'error',
                    'batch_num': batch_num,
                    'test_case_id': i,
                    'error_message': str(e),
                }
                self.results.append(result)

            # 每处理save_interval条数据保存一次中间结果
            try:
                self.save_interim_results(interval=save_interval)
            except Exception as e:
                print(f"json错误: {e}")
        
        # 保存最后一个不完整chunk的数据
        self.save_interim_results(interval=save_interval, force_save=True)

        # 计算总体评估结果
        evaluation_report = {
            'total_cases': len(test_cases),
            'success_cases': success_count,
            'correct_cases': correct_count,
            'hit2_cases': hit2_count,
            'accuracy': correct_count / len(test_cases) if len(test_cases) > 0 else 0,
            'hit@2': hit2_count / len(test_cases) if len(test_cases) > 0 else 0,
            'mean_rank': total_rank / success_count if success_count > 0 else 0,
            'total_tokens': total_tokens,
            'total_rank': total_rank,
            'results': self.results
        }   
        return evaluation_report

    def evaluate_single(self, predicted_lists: List[List], actual_lists: List[List]) -> bool:
        """评估单个预测结果和实际结果"""
        return self.evaluate_lists(predicted_lists, actual_lists)

    def save_interim_results(self, interval: int = 10, results_dir: str = None, force_save: bool = False) -> str:
        """
        每处理n条数据保存一次中间结果
        
        Args:
            interval: 保存频率，每处理interval条数据保存一次
            results_dir: 可选的结果保存目录路径
            force_save: 是否强制保存当前chunk(用于保存最后不完整的chunk)
            
        Returns:
            str: 结果保存的目录路径
        """
        # 初始化结果目录
        if not hasattr(self, 'results_dir') or not self.results_dir:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.results_dir = results_dir or os.path.join(os.getcwd(), f"eval_results_{timestamp}")
            os.makedirs(self.results_dir, exist_ok=True)
        
        # 只在有结果时保存
        if not hasattr(self, 'results') or not self.results:
            return self.results_dir
        
        # 计算当前chunk号和剩余数量
        total_results = len(self.results)
        current_chunk = total_results // interval
        remainder = total_results % interval
        
        # 判断是否需要保存数据
        should_save = (remainder == 0 and current_chunk > 0) or (force_save and remainder > 0)
        
        if should_save:
            # 确定要保存的chunk范围
            if remainder == 0 and current_chunk > 0:
                # 完整chunk
                start_idx = (current_chunk - 1) * interval
                end_idx = current_chunk * interval
                chunk_number = current_chunk
            else:
                # 不完整chunk
                start_idx = current_chunk * interval
                end_idx = total_results
                chunk_number = current_chunk + 1
            
            # 提取当前chunk的结果
            current_results = self.results[start_idx:end_idx]
            current_reports = self.reports[start_idx:end_idx] if hasattr(self, 'reports') and self.reports else []
            
            # 创建带有chunk号的文件名
            filename = os.path.join(self.results_dir, f"results_chunk_{chunk_number}.json")
            
            # 计算当前chunk的统计信息
            success_count = sum(1 for result in current_results if result['status'] == 'success')
            correct_cases = sum(1 for result in current_results if result.get('correct', False))
            correct_hit2_cases = sum(1 for result in current_results if result.get('correct_hit2', False))
            total_rank = sum(result.get('rank', 0) for result in current_results)

            total_tokens = sum(result.get('tokens', 0) for result in current_results)
            
            # 创建当前chunk的评估报告
            chunk_report = {
                'chunk_number': chunk_number,
                'chunk_size': len(current_results),
                'correct_cases': correct_cases,
                'correct_hit2_cases': correct_hit2_cases,
                'total_rank': total_rank,
                'mean_rank': total_rank / success_count if success_count > 0 else 0,
                'chunk_accuracy': correct_cases / len(current_results) if current_results else 0,
                'chunk_hit2_accuracy': correct_hit2_cases / len(current_results) if current_results else 0,
                'overall_progress': f"{total_results} cases processed",
                'total_tokens_in_chunk': total_tokens,
                'chunk_results': current_results,
                'chunk_reports': current_reports
            }
            
            try:
                # 应用JSON安全序列化处理
                safe_report = safe_json_serialize(chunk_report)
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(safe_report, f, ensure_ascii=False, indent=2)
            except Exception as e:
                # 如果完整报告无法保存，保存基本信息
                basic_report = {
                    'chunk_number': chunk_number,
                    'chunk_size': len(current_results),
                    'correct_cases': correct_cases,
                    'correct_hit2_cases': correct_hit2_cases,
                    'chunk_accuracy': correct_cases / len(current_results) if current_results else 0,
                    'chunk_hit2_accuracy': correct_hit2_cases / len(current_results) if current_results else 0,
                    'chunk_mean_rank': total_rank / success_count if success_count > 0 else 0,
                    'overall_progress': f"{total_results} cases processed",
                    'dump_error': str(e),
                }
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(basic_report, f, ensure_ascii=False, indent=2)
            
        return self.results_dir

    def get_metrics(self, predicted_output: Union[str, List[str]], expected_output: str) -> Dict[str, Any]:
        """
        评估预测输出与期望输出的匹配程度
        
        Args:
            predicted_output: 预测的图表类型或类型列表
            expected_output: 期望的图表类型
            
        Returns:
            Dict: 包含评估指标的字典
        """
        # 确保 predicted_output 是列表
        if not isinstance(predicted_output, list) or len(predicted_output) == 0:
            raise ValueError("预测结果为空")
        
        # 初始化评估指标
        metrics = {
            'is_correct': False,      # 第一个预测是否正确
            'is_correct_hit2': False, # 前两个预测中是否包含正确答案
            'rank': len(predicted_output) + 1,  # 正确答案的排名，默认设为最大值+1
        }
        
        # 寻找正确答案在预测列表中的位置
        for i, prediction in enumerate(predicted_output):
            if prediction == expected_output:
                metrics['rank'] = i + 1
                if i == 0:
                    metrics['is_correct'] = True
                if i < 2:
                    metrics['is_correct_hit2'] = True
                break
        
        return metrics