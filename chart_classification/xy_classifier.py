import os
import logging
import logging.handlers
import pandas as pd
import numpy as np
import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from lightgbm import log_evaluation
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# 导入配置和数据处理模块
from lgb_config import load_config
from data_utils import load_dataset, preprocess_data, create_encode_mapping, load_encode_mapping
from logger_setup import get_logger


class AxisClassifier:
    """轴线分类模型，使用LightGBM实现"""
    
    def __init__(self, config_path: Optional[str] = None):
        """初始化分类器
        
        Args:
            config_path: 配置文件路径，不提供则使用默认配置
        """
        # 加载配置
        self.config = load_config(config_path)
        self.paths = self.config['paths']
        self.params = self.config['xy_model_params']
        self.training_config = self.config['training']
        
        # 设置回调函数
        self.callbacks = [log_evaluation(period=self.training_config['log_period'])]
        
        # 设置日志
        self.logger = self._setup_logger()
        
        # 加载图表类型映射
        self.xy_mapping = None
        
    def _setup_logger(self) -> logging.Logger:
        """设置日志"""
        return get_logger('chart_classifier', self.paths['log'])
    
    def train(self, save_model: bool = True) -> lgb.Booster:
        """训练模型
        
        Args:
            save_model: 是否保存模型
            
        Returns:
            训练好的模型
        """
        self.logger.info("开始训练模型...")
        
        # 加载数据 - 使用导入的函数
        data_path = self.paths['train']
        X, Y = load_dataset(data_path, mode='train')
        
        # 预处理数据 - 使用导入的函数
        X, Y = preprocess_data(X, Y)
        Y = Y['is_x_or_y']
        # 分割训练集和验证集
        x_train, x_val, y_train, y_val = train_test_split(
            X, Y, 
            test_size=self.training_config['test_size'], 
            random_state=self.training_config['random_state']
        )
        
        # 构建数据集
        train_data = lgb.Dataset(data=x_train, label=y_train)
        val_data = lgb.Dataset(data=x_val, label=y_val)
        
        # 训练模型
        model = lgb.train(
            self.params,
            train_data,
            valid_sets=[val_data],
            callbacks=self.callbacks
        )
        
        # 保存模型
        if save_model:
            model_path = os.path.join(self.paths['model_path'], self._get_model_name())
            model.save_model(model_path)
            self.logger.info(f"模型已保存至: {model_path}")
            
            # 生成并保存图表类型映射
            mapping_path = os.path.join(self.paths['output'], 'xy_mapping.pkl')
            self.xy_mapping = create_encode_mapping(
                train_path=data_path,
                output_path=mapping_path,
                which_map='is_x_or_y',
                logger=self.logger
            )
        
        return model
    
    def predict(self, data: Optional[pd.DataFrame] = None, 
                model: Optional[lgb.Booster] = None,
                return_metrics: bool = True) -> Tuple[np.ndarray, Dict[str, float]]:
        """使用模型进行预测
        
        Args:
            data: 待预测数据，不提供则使用测试集
            model: 预测模型，不提供则加载默认模型
            return_metrics: 是否返回评估指标
            
        Returns:
            预测结果和评估指标
        """
        # 加载测试数据 - 使用导入的函数
        if data is None:
            test_data, truth_data = load_dataset(self.paths['test'], mode='test')
        else:
            test_data = data
            truth_data = None
            
        # 加载模型
        if model is None:
            model = self.load_model(self.paths['default_model'])
            
        # 预测
        y_pred = model.predict(test_data, num_iteration=model.best_iteration)
        
        # 如果有真实标签，计算指标
        metrics = {}
        if truth_data is not None and return_metrics:
            # 对真实标签进行编码
            le = LabelEncoder()
            y_true = le.fit_transform(truth_data['is_x_or_y'].astype('str'))
            metrics = self.calculate_metrics(y_pred, y_true)
            self.logger.info(f"评估指标: {metrics}")
            self.logger.info(f"特征重要性: {model.feature_importance()}")
            
        # 返回预测类别
        y_pred_class = np.argmax(y_pred, axis=1)
        
        return y_pred_class, metrics
    
    def predict_single(self, data_path: Optional[str] = None, 
                     model_path: Optional[str] = None, 
                     save_result: bool = True) -> pd.DataFrame:
        """预测单条数据的图表类型
        
        Args:
            data_path: 数据路径，默认使用配置中的预测路径
            model_path: 模型路径，默认使用配置中的默认模型
            save_result: 是否保存结果到CSV文件
        
        Returns:
            预测结果DataFrame
        """
        self.logger.info("开始预测单条数据...")
        
        # 使用默认路径
        if data_path is None:
            data_path = self.paths['predict']
        if model_path is None:
            model_path = self.paths['default_model']
        
        # 加载模型
        model = self.load_model(model_path)
        
        # 加载数据
        test_data_raw = pd.read_csv(data_path)
        
        # 保存字段ID和文件ID
        field_ids = test_data_raw['field_id'].copy() if 'field_id' in test_data_raw.columns else None
        fids = test_data_raw['fid'].copy() if 'fid' in test_data_raw.columns else None
        
        # 处理数据，移除不用于预测的列
        test_data = test_data_raw.copy()
        columns_to_remove = ['field_id', 'fid', 'is_xsrc', 'is_ysrc', 
                           'trace_type_n', 'is_x_or_y', 'trace_type']
        for col in columns_to_remove:
            if col in test_data.columns:
                test_data.pop(col)
        
        # 执行预测
        y_pred_proba = model.predict(test_data)
        y_pred_idx = np.argmax(y_pred_proba, axis=1)
        
        # 获取图表类型映射
        if self.xy_mapping is None:
            self.load_xy_mapping()
        
        # 将预测索引转换为图表类型
        predicted_types = []
        for idx in y_pred_idx:
            # 如果索引存在于映射中，使用映射的类型，否则使用"unknown"
            predicted_types.append(self.xy_mapping.get(idx, "unknown"))
        
        # 创建结果DataFrame
        results = pd.DataFrame({
            'field_id': field_ids if field_ids is not None else range(len(test_data)),
            'fid': fids if fids is not None else ['unknown'] * len(test_data),
            'predicted_trace_type': predicted_types,
            'confidence': np.max(y_pred_proba, axis=1)
        })
        
        # 添加前两个最可能的类型及其概率
        for i in range(len(results)):
            top_indices = np.argsort(-y_pred_proba[i])[:2]
            top_types = [self.xy_mapping.get(idx, "unknown") for idx in top_indices]
            top_probs = [y_pred_proba[i][idx] for idx in top_indices]
            
            results.loc[i, 'top1_type'] = top_types[0]
            results.loc[i, 'top1_prob'] = top_probs[0]
            if len(top_types) > 1:
                results.loc[i, 'top2_type'] = top_types[1]
                results.loc[i, 'top2_prob'] = top_probs[1]
        
        # 保存结果
        if save_result:
            result_path = os.path.join(self.paths['output'], 'prediction_result.csv')
            results.to_csv(result_path, index=False)
            self.logger.info(f"预测结果已保存到: {result_path}")
        
        return results
    
    def create_xy_mapping(self, train_path: Optional[str] = None, mapping_name: str = 'xy_mapping.pkl') -> Dict[int, str]:
        """创建并保存xy映射
        
        Args:
            train_path: 训练数据路径，默认使用配置中的训练路径
        
        Returns:
            映射字典 {索引: 图表类型}
        """
        if train_path is None:
            train_path = self.paths['train']
        
        mapping_path = os.path.join(self.paths['output'], mapping_name)
        self.xy_mapping = create_encode_mapping(
            train_path=train_path,
            output_path=mapping_path,
            logger=self.logger,
            which_map='is_x_or_y'
        )
        return self.xy_mapping
    
    def load_xy_mapping(self, mapping_name: str = 'xy_mapping.pkl') -> Dict[int, str]:
        """加载xy编码映射
        
        Returns:
            映射字典 {索引: 图表类型}
        """
        mapping_path = os.path.join(self.paths['output'], mapping_name)
        try:
            self.xy_mapping = load_encode_mapping(
                mapping_path=mapping_path,
                train_path=self.paths['train'],
                output_dir=self.paths['output'],
                logger=self.logger
            )
        except Exception as e:
            self.logger.error(f"加载轴线编码映射失败: {e}")
            self.logger.info("正在重新创建轴线编码映射...")
            self.xy_mapping = self.create_xy_mapping()
        
        return self.xy_mapping
    
    def parameter_search(self, param_key: str, start: float, end: float, step: float) -> Dict[float, Dict[str, float]]:
        """参数搜索
        
        Args:
            param_key: 要搜索的参数名
            start: 起始值
            end: 结束值
            step: 步长
            
        Returns:
            每个参数值对应的评估指标
        """
        self.logger.info(f"开始参数搜索: {param_key} 从 {start} 到 {end}, 步长 {step}")
        
        results = {}
        for param_value in np.arange(start, end, step):
            # 更新参数
            self.params[param_key] = param_value
            
            # 训练和评估
            model = self.train(save_model=False)
            _, metrics = self.predict(model=model)
            
            # 记录结果
            results[param_value] = metrics
            self.logger.info(f"参数 {param_key}={param_value}, 指标: {metrics}")
            
        return results
    
    def load_model(self, model_path: str) -> lgb.Booster:
        """加载模型
        
        Args:
            model_path: 模型路径
            
        Returns:
            加载的模型
        """
        return lgb.Booster(model_file=model_path)
    
    def calculate_metrics(self, y_pred: np.ndarray, y_true: np.ndarray, 
                          metrics: List[str] = ['acc', 'hit@2', 'MR']) -> Dict[str, float]:
        """计算评估指标
        
        Args:
            y_pred: 预测概率
            y_true: 真实标签
            metrics: 要计算的指标列表
            
        Returns:
            评估指标字典
        """
        result = {}
        n = len(y_pred)
        
        # 准确率
        if 'acc' in metrics:
            result['acc'] = accuracy_score(y_true, np.argmax(y_pred, axis=1))
        
        # Hit@K 指标
        if 'hit@2' in metrics:
            cnt1 = 0
            cnt2 = 0
            mr = 0
            
            for i in range(n):
                idxes = np.argsort(-y_pred[i])
                ranks = np.zeros_like(y_pred[i])
                for rank, idx in enumerate(idxes):
                    ranks[idx] = rank
                
                rank_p = ranks[y_true[i]]
                if rank_p <= 0:
                    cnt1 += 1
                if rank_p <= 1:
                    cnt2 += 1
                mr += rank_p
                
            result['hit@1'] = float(cnt1) / float(n)
            result['hit@2'] = float(cnt2) / float(n)
            
            if 'MR' in metrics:
                result['MR'] = float(mr) / float(n) + 1
                
        return result
    
    def _get_model_name(self) -> str:
        """生成模型名称"""
        timestamp = datetime.now().strftime("%m%d%H%M%S")
        param_str = f"leave_{self.params.get('num_leaves', 'NA')}_tree_{self.params.get('num_trees', 'NA')}_"
        return f"py12_axis_model_{param_str}{timestamp}.txt"


if __name__ == "__main__":
    # 创建分类器实例
    classifier = AxisClassifier()
    
    # 参数搜索示例
    # classifier.parameter_search(
    #     param_key='min_child_samples',
    #     start=10,
    #     end=110,
    #     step=10
    # )
    
    # 或者直接训练和评估
    model = classifier.train()
    y_pred, metrics = classifier.predict(model=model)
    print(f"评估指标: {metrics}")
    print(f"预测结果: {y_pred}")