"""数据处理工具"""

import os
import pickle
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from sklearn.preprocessing import LabelEncoder

def load_dataset(path: str, mode: str = 'test') -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """加载数据集
    
    Args:
        path: 数据集路径
        mode: 模式，'test'/'train'/'val'/'predict'
        
    Returns:
        数据特征和标签元组
    """
    if mode not in ['test', 'train', 'val', 'predict']:
        raise ValueError("模式必须是 'test', 'train', 'val' 或 'predict'")
    
    data = pd.read_csv(path)
    
    if mode == 'predict':
        return data, None
        
    # 提取和预处理特征
    def pop_with_default(dataframe, column_name, default=None):
        if column_name in dataframe.columns:
            return dataframe.pop(column_name)
        return default
        
    pop_with_default(data, 'field_id')
    pop_with_default(data, 'fid')
    pop_with_default(data, 'is_xsrc')
    pop_with_default(data, 'is_ysrc')
    pop_with_default(data, 'trace_type_n')
    is_x_or_y = pop_with_default(data, 'is_x_or_y')
    trace_type = pop_with_default(data, 'trace_type')
    
    truth = is_x_or_y.to_frame()
    truth.insert(loc=truth.shape[1], column='trace_type', value=trace_type, allow_duplicates=False)
    
    return data, truth

def preprocess_data(X: pd.DataFrame, Y: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """预处理数据
    
    Args:
        X: 特征数据
        Y: 标签数据
        
    Returns:
        处理后的特征和标签
    """
    # 标签编码
    le = LabelEncoder()
    Y_processed = Y.copy()
    Y_processed['is_x_or_y'] = le.fit_transform(Y_processed['is_x_or_y'].astype('str'))
    Y_processed['trace_type'] = le.fit_transform(Y_processed['trace_type'].astype('str'))
    
    # 这里可以添加更多特征工程步骤
    
    return X, Y_processed['trace_type']

def create_chart_type_mapping(train_path: str, output_path: str, logger=None) -> Dict[int, str]:
    """创建并保存图表类型映射
    
    Args:
        train_path: 训练数据路径
        output_path: 保存映射文件的路径
        logger: 可选的日志记录器
        
    Returns:
        映射字典 {索引: 图表类型}
    """
    if logger:
        logger.info("创建图表类型映射...")
    else:
        print("创建图表类型映射...")
    
    # 加载训练数据
    _, truth_data = load_dataset(train_path)
    
    # 获取唯一图表类型
    unique_chart_types = truth_data['trace_type'].unique()
    
    # 创建编码器并拟合
    le = LabelEncoder()
    encoded_types = le.fit_transform(unique_chart_types.astype('str'))
    
    # 创建映射字典
    mapping = {idx: chart_type for idx, chart_type in zip(encoded_types, unique_chart_types)}
    
    # 保存映射到文件
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(mapping, f)
    
    if logger:
        logger.info(f"图表类型映射已保存到: {output_path}")
        logger.debug("图表类型映射内容:")
        for idx, chart_type in sorted(mapping.items()):
            logger.debug(f"  {idx}: {chart_type}")
    else:
        print(f"图表类型映射已保存到: {output_path}")
        print("图表类型映射内容:")
        for idx, chart_type in sorted(mapping.items()):
            print(f"  {idx}: {chart_type}")
    
    return mapping

def load_chart_type_mapping(mapping_path: str, train_path: Optional[str] = None, 
                            output_dir: Optional[str] = None, logger=None) -> Dict[int, str]:
    """加载图表类型映射
    
    Args:
        mapping_path: 映射文件路径
        train_path: 如果映射文件不存在，用于创建映射的训练数据路径
        output_dir: 输出目录路径，用于保存新创建的映射
        logger: 可选的日志记录器
        
    Returns:
        映射字典 {索引: 图表类型}
    """
    if os.path.exists(mapping_path):
        try:
            with open(mapping_path, 'rb') as f:
                mapping = pickle.load(f)
            return mapping
        except Exception as e:
            if logger:
                logger.error(f"加载映射文件时出错: {e}")
            else:
                print(f"加载映射文件时出错: {e}")
    
    # 如果映射文件不存在或加载失败，且提供了训练路径，则创建新映射
    if train_path and output_dir:
        if logger:
            logger.info("映射文件不存在或无效，正在创建新映射...")
        else:
            print("映射文件不存在或无效，正在创建新映射...")
        return create_chart_type_mapping(train_path, mapping_path, logger)
    else:
        raise FileNotFoundError(f"映射文件 {mapping_path} 不存在，且未提供创建所需的训练数据路径")
