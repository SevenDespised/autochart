import os
import pickle
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader

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

def preprocess_data(X: pd.DataFrame, Y: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, LabelEncoder]]:
    """预处理数据
    
    Args:
        X: 特征数据
        Y: 标签数据
        
    Returns:
        处理后的特征和标签，以及原始标签编码器（用于反向映射）
    """
    # 标签编码器（保留用于后续的类别映射）
    le_xy = LabelEncoder()
    le_trace = LabelEncoder()
    
    # 先用LabelEncoder转换标签，记录映射关系
    le_xy.fit(Y['is_x_or_y'].astype('str'))
    le_trace.fit(Y['trace_type'].astype('str'))
    
    # 独热编码
    Y_processed = Y.copy()
    
    # 创建独热编码
    xy_one_hot = pd.get_dummies(Y_processed['is_x_or_y'].astype('str'), prefix='xy')
    trace_one_hot = pd.get_dummies(Y_processed['trace_type'].astype('str'), prefix='trace')
    
    # 替换原始标签列
    Y_processed = pd.concat([xy_one_hot, trace_one_hot], axis=1)
    
    # 返回编码器，便于之后反向映射
    encoders = {
        'is_x_or_y': le_xy,
        'trace_type': le_trace
    }
    
    return X, Y_processed, encoders

class ChartDataset(Dataset):
    """图表数据集，用于PyTorch模型训练"""
    
    def __init__(self, features: pd.DataFrame, labels: pd.DataFrame = None, task: str = 'both'):
        """
        初始化数据集
        
        Args:
            features: 特征数据
            labels: 标签数据，预测模式下可为None
            task: 任务类型，'xy'(预测是X还是Y)、'trace'(预测图表类型)或'both'(两者都预测)
        """
        self.features = features
        self.labels = labels
        self.task = task
        
        # 将DataFrame转换为numpy数组，加快访问速度
        self.X = features.values.astype(np.float32)
        
        if labels is not None:
            # 获取独热编码列
            xy_cols = [col for col in labels.columns if col.startswith('xy_')]
            trace_cols = [col for col in labels.columns if col.startswith('trace_')]
            
            if task == 'xy':
                self.y = labels[xy_cols].values.astype(np.float32)
            elif task == 'trace':
                self.y = labels[trace_cols].values.astype(np.float32)
            else:  # both
                self.y_xy = labels[xy_cols].values.astype(np.float32)
                self.y_trace = labels[trace_cols].values.astype(np.float32)
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.features)
    
    def __getitem__(self, idx):
        """获取数据集中的一个样本"""
        X = torch.tensor(self.X[idx], dtype=torch.float32)
        
        if self.labels is None:
            return X
        
        if self.task == 'xy':
            y = torch.tensor(self.y[idx], dtype=torch.float32)
            return X, y
        elif self.task == 'trace':
            y = torch.tensor(self.y[idx], dtype=torch.float32)
            return X, y
        else:  # both
            y_xy = torch.tensor(self.y_xy[idx], dtype=torch.float32)
            y_trace = torch.tensor(self.y_trace[idx], dtype=torch.float32)
            return X, (y_xy, y_trace)

def get_data_loaders(
    data_path: str,
    batch_size: int = 32,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    task: str = 'both',
    num_workers: int = 4,
    random_seed: int = 42
) -> Dict[str, DataLoader]:
    """
    创建训练、验证和测试数据加载器
    
    Args:
        data_path: 数据文件路径
        batch_size: 批次大小
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        task: 任务类型,'xy'/'trace'/'both'
        num_workers: 数据加载线程数
        random_seed: 随机种子，用于数据集分割
        
    Returns:
        包含训练、验证和测试数据加载器的字典
    """
    # 设置随机种子
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    # 加载完整数据集
    X, y = load_dataset(data_path, mode='train')
    X, y, encoders = preprocess_data(X, y)
    
    # 计算划分点
    n_samples = len(X)
    indices = np.random.permutation(n_samples)
    
    train_end = int(train_ratio * n_samples)
    val_end = int((train_ratio + val_ratio) * n_samples)
    
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    
    # 创建各数据子集
    train_X, train_y = X.iloc[train_idx].copy(), y.iloc[train_idx].copy()
    val_X, val_y = X.iloc[val_idx].copy(), y.iloc[val_idx].copy() 
    test_X, test_y = X.iloc[test_idx].copy(), y.iloc[test_idx].copy()
    
    # 对数据集进行处理
    with open("feature_extraction/feature_list_float_bool.pkl", 'rb') as f:
        feature_list = pickle.load(f)
    def cut_off(x, q1, q3):
        if x >= q1 and x <= q3:
            return x
        elif x < q1:
            return q1
        else:
            return q3

    dict_cut_off = {}

    for i in feature_list['float']:

        train_X[i] = np.array(train_X[i].astype('float32'))
        quantile_1 = np.quantile(train_X[i][np.isfinite(train_X[i])], 0.05)
        quantile_3 = np.quantile(train_X[i][np.isfinite(train_X[i])], 0.95)

        dict_cut_off[i] = (quantile_1, quantile_3)
        train_X[i] = train_X[i].apply(lambda x: cut_off(x, quantile_1, quantile_3))
    
    for i in feature_list['float']:
        quantile_1, quantile_3 = dict_cut_off[i]
        test_X[i] = np.array(test_X[i].astype('float32'))
        test_X[i] = test_X[i].apply(lambda x: cut_off(x, quantile_1, quantile_3))
        val_X[i] = np.array(val_X[i].astype('float32'))
        val_X[i] = val_X[i].apply(lambda x: cut_off(x, quantile_1, quantile_3))

    train_X[feature_list['bool']] = train_X[feature_list['bool']].fillna(False)
    test_X[feature_list['bool']] = test_X[feature_list['bool']].fillna(False)
    val_X[feature_list['bool']] = val_X[feature_list['bool']].fillna(False)

    # 对数值特征进行标准化
    scaler = StandardScaler()
    train_X[feature_list['float']] = scaler.fit_transform(train_X[feature_list['float']])
    test_X[feature_list['float']] = scaler.transform(test_X[feature_list['float']])
    val_X[feature_list['float']] = scaler.transform(val_X[feature_list['float']])

    # 创建Dataset对象
    train_dataset = ChartDataset(train_X, train_y, task=task)
    val_dataset = ChartDataset(val_X, val_y, task=task)
    test_dataset = ChartDataset(test_X, test_y, task=task)
    
    # 创建DataLoader对象
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'encoders': encoders,
        'task': task
    }

def create_predict_loader(
    data_path: str,
    batch_size: int = 32,
    num_workers: int = 4
) -> DataLoader:
    """
    创建用于预测的数据加载器
    
    Args:
        data_path: 预测数据文件路径
        batch_size: 批次大小
        num_workers: 数据加载线程数
        
    Returns:
        预测数据加载器
    """
    # 加载预测数据
    X, _ = load_dataset(data_path, mode='predict')
    
    # 特征预处理
    # 这里应根据实际情况添加与训练数据相同的预处理步骤
    
    # 创建数据集和加载器
    predict_dataset = ChartDataset(X, labels=None)
    
    predict_loader = DataLoader(
        predict_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return predict_loader, X