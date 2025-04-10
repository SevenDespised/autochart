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
        
        correct_count = 0
        table_correct_count = 0
        columns_correct_count = 0
        total_tokens = 0
        n = len(test_cases)
        for i, test_case in enumerate(test_cases):
            try:
                print(f"[BATCH:{batch_num}][{i + 1}/{n}] 正在评估测试用例...")
                input_data = test_case['input']
                expected_output = test_case['expected_output']
                table_expected_output = expected_output.get("tables", [])
                columns_expected_output = expected_output.get("columns", {})
                final_expected_output = expected_output["x_data"] + expected_output["y_data"]
                #####
                #if "Show me about the distribution of date_address_to and the amount of date_address_to, and group by attribute other_details and bin date_address_to by month in a bar chart." not in input_data["nl_queries"]:
                #    print(f"跳过测试用例 {i + 1}，输入数据不符合预期格式。")
                #    continue
                #####
                
                # 执行流水线
                pipeline_report = self.pipeline.execute_pipeline(input_data)
                pipeline_report['test_case_id'] = i
                self.reports.append(pipeline_report)
                predicted_output_dict = pipeline_report['output_data'].get("columns_data", {})

                # 提取阶段的输出数据
                table_output = self.pipeline.execution_data.get_output("table_selection")["table_names"]
                columns_output = self.pipeline.execution_data.get_output("column_selection")
                # 提取字典的值列表
                predicted_output = list(predicted_output_dict.values())

                is_sorted = False if not input_data["sort"] else True
                # 评估结果
                is_correct = self.are_nested_lists_equal(final_expected_output, predicted_output, is_sorted)
                if is_correct:
                    correct_count += 1
                
                # 评估阶段结果
                is_table_stage_correct = self.are_tables_output_equal(table_output, table_expected_output)
                if is_table_stage_correct:
                    table_correct_count += 1
                is_columns_stage_correct = self.are_columns_output_equal(columns_output, columns_expected_output)
                if is_columns_stage_correct:
                    columns_correct_count += 1
                # 记录本次执行的token消耗
                tokens_used = pipeline_report.get('tokens', 0)
                total_tokens += tokens_used
                if pipeline_report["success"]:
                    result = {
                        'status': 'success',
                        'test_case_id': i,
                        'input': input_data,
                        'evaluation': {
                            "final":{
                                'expected': final_expected_output,
                                'predicted': predicted_output,
                                'correct': is_correct,
                            },
                            "table": {
                                'expected': table_expected_output,
                                'predicted': table_output,
                                'correct': is_table_stage_correct,
                            },
                            "column": {
                                'expected': columns_expected_output,
                                'predicted': columns_output,
                                'correct': is_columns_stage_correct,
                            }
                        },
                        'execution_time': pipeline_report['execution_time'],
                        'tokens': tokens_used
                    }
                else:
                    result = {
                        'status': 'failure',
                        'test_case_id': i,
                        'evaluation': {
                            "final":{
                                'expected': final_expected_output,
                                'predicted': predicted_output,
                                'correct': is_correct,
                            },
                            "table": {
                                'expected': table_expected_output,
                                'predicted': table_output,
                                'correct': is_table_stage_correct,
                            },
                            "column": {
                                'expected': columns_expected_output,
                                'predicted': columns_output,
                                'correct': is_columns_stage_correct,
                            }
                        }
                    }
                self.results.append(result)
            except Exception as e:
                print(f"[BATCH:{batch_num}] 处理测试用例 {i} 时发生错误: {e}")
                result = {
                    'status': 'error',
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
            'evaluation': {
                "final":{
                    'correct_cases': correct_count,
                    'accuracy': correct_count / len(test_cases) if test_cases else 0,
                },
                "table": {
                    'correct_cases': table_correct_count,
                    'accuracy': table_correct_count / len(test_cases) if test_cases else 0,
                },
                "column": {
                    'correct_cases': columns_correct_count,
                    'accuracy': columns_correct_count / len(test_cases) if test_cases else 0,
                }
            },
            'correct_cases': correct_count,
            'accuracy': correct_count / len(test_cases) if test_cases else 0,
            'total_tokens': total_tokens,
            'results': self.results
        }   
        return evaluation_report
    def are_nested_lists_equal(self, nested_list1, nested_list2, consider_order=False):
        """
        判断两个列表列表是否相等
        
        Args:
            nested_list1: 第一个列表列表
            nested_list2: 第二个列表列表  
            consider_order: 是否考虑内部列表元素顺序
        """
        if len(nested_list1) != len(nested_list2):
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


    def evaluate_single(self, predicted_lists: List[List], actual_lists: List[List]) -> bool:
        """评估单个预测结果和实际结果"""
        return self.evaluate_lists(predicted_lists, actual_lists)
    
    def get_statistics(self) -> Dict:
        """获取评估统计信息"""
        if not self.results:
            return {"error": "No evaluation results available"}
        
        total = len(self.results)
        correct = sum(1 for result in self.results if result.get('correct', False))
        total_tokens = sum(result.get('tokens', 0) for result in self.results)
        
        return {
            'total_cases': total,
            'correct_cases': correct,
            'accuracy': correct / total if total > 0 else 0,
            'total_tokens': total_tokens
        }

    def save_interim_results(self, interval: int = 10, results_dir: str = None, force_save: bool = False) -> str:
        """
        每处理n条数据保存一次中间结果，只保存当前chunk的n条数据
        
        Args:
            interval: 保存频率，每处理interval条数据保存一次
            results_dir: 可选的结果保存目录路径
            force_save: 是否强制保存当前chunk(用于保存最后不完整的chunk)
            
        Returns:
            str: 结果保存的目录路径
        """
        # 如果没有提供目录，在当前目录创建带时间戳的results目录
        if not hasattr(self, 'results_dir') or not self.results_dir:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.results_dir = results_dir or os.path.join(os.getcwd(), f"eval_results_{timestamp}")
            os.makedirs(self.results_dir, exist_ok=True)
        
        # 只有在有结果时才保存
        if hasattr(self, 'results') and self.results:
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
                current_reports = self.reports[start_idx:end_idx] if hasattr(self, 'reports') else []
                
                # 创建带有chunk号的文件名
                filename = os.path.join(self.results_dir, f"results_chunk_{chunk_number}.json")
                
                # 计算当前chunk的统计信息
                # 最终结果正确案例计数
                final_correct_cases = sum(1 for result in current_results 
                                        if result.get('evaluation', {}).get('final', {}).get('correct', False))
                # 表格选择正确案例计数
                table_correct_cases = sum(1 for result in current_results 
                                        if result.get('evaluation', {}).get('table', {}).get('correct', False))
                # 列选择正确案例计数
                column_correct_cases = sum(1 for result in current_results 
                                        if result.get('evaluation', {}).get('column', {}).get('correct', False))
                
                chunk_size = len(current_results)
                
                # 创建当前chunk的评估报告
                chunk_report = {
                    'chunk_number': chunk_number,
                    'chunk_size': chunk_size,
                    'evaluation': {
                        'final': {
                            'correct_cases': final_correct_cases,
                            'accuracy': final_correct_cases / chunk_size if chunk_size else 0
                        },
                        'table': {
                            'correct_cases': table_correct_cases,
                            'accuracy': table_correct_cases / chunk_size if chunk_size else 0
                        },
                        'column': {
                            'correct_cases': column_correct_cases,
                            'accuracy': column_correct_cases / chunk_size if chunk_size else 0
                        }
                    },
                    'overall_progress': f"{total_results} cases processed",
                    'chunk_results': current_results,
                    'chunk_reports': current_reports
                }
                
                # 应用JSON安全序列化处理
                safe_report = safe_json_serialize(chunk_report)
                with open(filename, 'w', encoding='utf-8') as f:
                    try:
                        json.dump(safe_report, f, ensure_ascii=False, indent=2)
                    except Exception as e:
                        # 保存基本统计信息（若完整报告保存失败）
                        json.dump({
                                    'chunk_number': chunk_number,
                                    'chunk_size': chunk_size,
                                    'evaluation': {
                                        'final': {
                                            'correct_cases': final_correct_cases,
                                            'accuracy': final_correct_cases / chunk_size if chunk_size else 0
                                        },
                                        'table': {
                                            'correct_cases': table_correct_cases,
                                            'accuracy': table_correct_cases / chunk_size if chunk_size else 0
                                        },
                                        'column': {
                                            'correct_cases': column_correct_cases,
                                            'accuracy': column_correct_cases / chunk_size if chunk_size else 0
                                        }
                                    },
                                    'overall_progress': f"{total_results} cases processed",
                                    'dump_error': str(e),
                                  }, f, ensure_ascii=False, indent=2)
        return self.results_dir

    def are_tables_output_equal(self, predicted_tables, ground_truth_tables):
        """
        评估预测的表名列表与真实表名列表是否相等（不区分大小写）。
        
        参数:
            predicted_tables: 预测的表名列表
            ground_truth_tables: 真实的表名列表
            
        返回:
            bool: 如果预测值包含所有真实值则返回True，否则返回False
        """
        # 将所有表名转换为小写以实现不区分大小写的比较
        predicted_lower = [table.lower() for table in predicted_tables]
        predicted_lower = [table[:-4] if table.endswith(".csv") else table for table in predicted_lower] # 去除.csv后缀
        truth_lower = [table.lower() for table in ground_truth_tables]
        
        # 检查所有真实表名是否都在预测表名中
        for table in truth_lower:
            if table not in predicted_lower:
                return False
        
        return True


    def are_columns_output_equal(self, predicted_columns, ground_truth_columns):
        """
        评估预测的列名与真实列名是否相等（不区分大小写）。
        
        参数:
            predicted_columns: 以表名为键，列名列表为值的字典（预测值）
            ground_truth_columns: 以表名为键，列名列表为值的字典（真实值）
            
        返回:
            bool: 如果预测值包含所有真实值则返回True，否则返回False
        """
        # 检查每个真实表及其列
        for gt_table, gt_columns in ground_truth_columns.items():
            gt_table_lower = gt_table.lower()
            # 寻找预测中对应的表（不区分大小写）
            found_table = False
            for pred_table, pred_columns in predicted_columns.items():
                pred_table_lower = pred_table.lower()
                pred_table_lower = pred_table_lower[:-4] if pred_table_lower.endswith(".csv") else pred_table_lower
                if pred_table_lower == gt_table_lower:
                    found_table = True
                    
                    # 将列名转换为小写
                    pred_cols_lower = [col.lower() for col in pred_columns]
                    gt_cols_lower = [col.lower() for col in gt_columns]
                    
                    # 检查所有真实列名是否都在预测列名中
                    for col in gt_cols_lower:
                        if col not in pred_cols_lower:
                            return False
                    
                    break
            
            # 如果没有找到对应的表，返回False
            if not found_table:
                return False
        
        return True