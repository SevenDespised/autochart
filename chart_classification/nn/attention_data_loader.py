import os
import pickle
import pandas as pd
import numpy as np
import json
from typing import Tuple, Optional, Dict, Any, List
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
import warnings

# 临时忽视 RuntimeWarning
warnings.filterwarnings("ignore", category=RuntimeWarning)

pd.set_option('future.no_silent_downcasting', True)

def load_dataset(path: str, mode: str = 'test') -> Dict[str, Dict]:
    """加载数据集，将具有相同id的行合并到一个列表中
    
    Args:
        path: 数据集路径
        mode: 模式，'test'/'train'/'val'/'predict'
        
    Returns:
        字典，键为id，值为该id对应的数据，包含特征和标签
    """
    if mode not in ['test', 'train', 'val', 'predict']:
        raise ValueError("模式必须是 'test', 'train', 'val' 或 'predict'")
    
    data = pd.read_csv(path)
    
    # 提取需要作为标签的列
    labels = {}
    if mode != 'predict':
        if 'is_x_or_y' in data.columns:
            labels['is_x_or_y'] = data['is_x_or_y'].copy()
        if 'trace_type' in data.columns:
            labels['trace_type'] = data['trace_type'].copy()
    
    # 从数据中移除不需要的列
    to_drop = ['is_xsrc', 'is_ysrc', 'trace_type_n', 'field_id', 'exists']
    for col in to_drop:
        if col in data.columns:
            data.drop(col, axis=1, inplace=True)
    
    result = {}
    # 按id分组，将每组数据合并为一个条目
    for id_val, group in data.groupby('fid'):
        features = group.drop(['is_x_or_y', 'trace_type'], axis=1, errors='ignore')
        
        # 如果是预测模式，没有标签
        if mode == 'predict':
            result[id_val] = {'x_data': features.to_dict('records')}
        else:
            y_data = {}
            if 'is_x_or_y' in labels:
                y_data['is_x_or_y'] = labels['is_x_or_y'][group.index].values
            if 'trace_type' in labels:
                y_data['trace_type'] = labels['trace_type'][group.index].values
            
            result[id_val] = {
                'x_data': features.to_dict('records'),
                'y_data': y_data
            }
    # 删除fid
    for id_val in result.keys():
        for feature_dict in result[id_val]['x_data']:
            if 'fid' in feature_dict:
                del feature_dict['fid']
    return result

class MappingEncoder:
    """替代LabelEncoder的映射编码器"""
    
    def __init__(self, mapping_dict):
        self.mapping = mapping_dict
        self.inverse_mapping = {v: k for k, v in mapping_dict.items()}
        self.classes_ = list(mapping_dict.keys())
    
    def transform(self, labels):
        return np.array([self.mapping.get(label, -1) for label in labels])
    
    def inverse_transform(self, indices):
        return np.array([self.inverse_mapping.get(idx, "unknown") for idx in indices])

def preprocess_data(data: Dict[str, Dict], mapping_file: str = "chart_classification/output/mapping.json") -> Tuple[Dict[str, np.ndarray], Dict[str, Dict], Dict[str, Any]]:
    """
    预处理数据
    
    Args:
        data: 从load_dataset返回的字典数据
        mapping_file: 映射文件路径，包含xy_axis和chart_type的编码映射
        
    Returns:
        处理后的数据，包含特征数组和标签数组，以及映射编码器
    """
    # 加载映射文件
    if mapping_file and os.path.exists(mapping_file):
        with open(mapping_file, 'r') as f:
            mappings = json.load(f)
        xy_mapping = mappings['xy_axis']
        trace_mapping = mappings['chart_type']
    else:
        raise FileNotFoundError(f"映射文件 {mapping_file} 不存在")
    
    # 创建映射编码器
    encoder_xy = MappingEncoder(xy_mapping)
    encoder_trace = MappingEncoder(trace_mapping)
    
    processed_data = {}
    for id_val, item in data.items():
        # 处理特征数据：将字典列表转换为特征向量
        features = pd.DataFrame(item['x_data'])
        
        # 处理标签数据
        labels = {}
        if 'y_data' in item:
            if 'is_x_or_y' in item['y_data']:
                xy_labels = [str(label).lower() for label in item['y_data']['is_x_or_y']]
                xy_encoded = encoder_xy.transform(xy_labels)
                xy_one_hot = np.zeros((len(xy_encoded), len(encoder_xy.classes_)))
                for i, label_idx in enumerate(xy_encoded):
                    if label_idx >= 0:  # 只有在映射中存在的标签才设置
                        xy_one_hot[i, label_idx] = 1
                labels['xy'] = xy_one_hot
            
            if 'trace_type' in item['y_data']:
                trace_labels = [str(label) for label in item['y_data']['trace_type']]
                trace_encoded = encoder_trace.transform(trace_labels)
                trace_one_hot = np.zeros((len(trace_encoded), len(encoder_trace.classes_)))
                for i, label_idx in enumerate(trace_encoded):
                    if label_idx >= 0:  # 只有在映射中存在的标签才设置
                        trace_one_hot[i, label_idx] = 1
                labels['trace'] = trace_one_hot
        
        processed_data[id_val] = {
            'features': features,
            'labels': labels
        }
    
    # 返回编码器，便于之后反向映射
    encoders = {
        'is_x_or_y': encoder_xy,
        'trace_type': encoder_trace
    }
    
    return processed_data, encoders

class AttentionDataset(Dataset):
    """图表数据集，用于PyTorch模型训练"""
    
    def __init__(self, data: Dict[str, Dict], task: str = 'both'):
        """
        初始化数据集
        
        Args:
            data: 预处理后的数据，包含特征和标签
            task: 任务类型，'xy'(预测是X还是Y)、'trace'(预测图表类型)或'both'(两者都预测)
        """
        self.data = data
        self.task = task
        self.ids = list(data.keys())
        
        # 将特征转换为numpy数组列表
        self.features = []
        self.xy_labels = []
        self.trace_labels = []
        
        for id_val in self.ids:
            item = data[id_val]
            self.features.append(item['features'].values.astype(np.float32))
            
            if 'labels' in item:
                if 'xy' in item['labels']:
                    self.xy_labels.append(item['labels']['xy'].astype(np.float32))
                if 'trace' in item['labels']:
                    self.trace_labels.append(item['labels']['trace'].astype(np.float32))
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.ids)
    
    def __getitem__(self, idx):
        """获取数据集中的一个样本"""
        X = torch.tensor(self.features[idx], dtype=torch.float32)
        seq_len = X.shape[0]  # 记录序列的实际长度
        
        if not self.xy_labels and not self.trace_labels:
            return X, seq_len
        
        if self.task == 'xy' and self.xy_labels:
            y = torch.tensor(self.xy_labels[idx], dtype=torch.float32)
            return X, seq_len, y
        elif self.task == 'trace' and self.trace_labels:
            y = torch.tensor(self.trace_labels[idx], dtype=torch.float32)
            return X, seq_len, y
        else:  # both
            if self.xy_labels and self.trace_labels:
                y_xy = torch.tensor(self.xy_labels[idx], dtype=torch.float32)
                y_trace = torch.tensor(self.trace_labels[idx], dtype=torch.float32)
                return X, seq_len, (y_xy, y_trace)
            elif self.xy_labels:
                y_xy = torch.tensor(self.xy_labels[idx], dtype=torch.float32)
                return X, seq_len, y_xy
            else:
                y_trace = torch.tensor(self.trace_labels[idx], dtype=torch.float32)
                return X, seq_len, y_trace

def get_data_loaders(
    data_path: str,
    batch_size: int = 32,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    task: str = 'both',
    num_workers: int = 4,
    random_seed: int = 42,
    mapping_file: str = None
) -> Dict[str, Any]:
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
        mapping_file: 映射文件路径，默认为None则使用"chart_classification/output/mapping.json"
        
    Returns:
        包含训练、验证和测试数据加载器的字典
    """
    # 设置默认映射文件
    if mapping_file is None:
        mapping_file = os.path.join("chart_classification", "output", "mapping.json")
    
    # 设置随机种子
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    # 加载完整数据集
    data = load_dataset(data_path, mode='train')
    processed_data, encoders = preprocess_data(data, mapping_file)
    
    # 随机分割数据集
    ids = list(processed_data.keys())
    np.random.shuffle(ids)
    
    train_end = int(train_ratio * len(ids))
    val_end = int((train_ratio + val_ratio) * len(ids))
    
    train_ids = ids[:train_end]
    val_ids = ids[train_end:val_end]
    test_ids = ids[val_end:]
    
    # 创建子数据集
    train_data = {id_val: processed_data[id_val] for id_val in train_ids}
    val_data = {id_val: processed_data[id_val] for id_val in val_ids}
    test_data = {id_val: processed_data[id_val] for id_val in test_ids}
    
    # 应用cut_off处理和标准化
    # 加载feature_list
    feature_list_path = os.path.join("feature_extraction", "feature_list_float_bool.pkl")
    with open(feature_list_path, 'rb') as f:
        feature_list = pickle.load(f)
    
    def cut_off(x, q1, q3):
        try:
            if x >= q1 and x <= q3:
                return x
            elif x < q1:
                return q1
            else:
                return q3
        except Exception as e:
            print(f"Error in cut_off: {e}")
            return q3
    
    # 收集所有训练数据的浮点特征
    all_train_features = {}
    for id_val, item in train_data.items():
        for col in feature_list['float']:
            if col in item['features'].columns:
                if col not in all_train_features:
                    all_train_features[col] = []
                all_train_features[col].extend(item['features'][col].values)
    
    # 计算每个特征的分位数
    dict_cut_off = {}
    for col in feature_list['float']:
        if col in all_train_features:
            values = np.array(all_train_features[col], dtype=np.float32)
            values = values[np.isfinite(values)]
            quantile_1 = np.quantile(values, 0.05) if len(values) > 0 else 0
            quantile_3 = np.quantile(values, 0.95) if len(values) > 0 else 1
            dict_cut_off[col] = (quantile_1, quantile_3)
    
    # 应用cut_off和标准化
    scaler = StandardScaler()
    
    # 准备用于拟合标准化器的数据
    train_float_features = []
    for id_val, item in train_data.items():
        for col in feature_list['float']:
            if col in item['features'].columns:
                # 应用cut_off
                item['features'][col] = item['features'][col].apply(
                    lambda x: cut_off(x, dict_cut_off[col][0], dict_cut_off[col][1]) 
                )
                
        # 处理布尔特征
        for col in feature_list['bool']:
            if col in item['features'].columns:
                item['features'][col] = item['features'][col].fillna(False)
                # 将布尔值转换为浮点型的0和1
                item['features'][col] = item['features'][col].astype(float)
        
        # 收集浮点特征用于拟合标准化器
        if len(feature_list['float']) > 0:
            common_cols = set(item['features'].columns) & set(feature_list['float'])
            if common_cols:
                train_float_features.append(item['features'][list(common_cols)])
    
    # 如果有浮点特征，拟合标准化器
    if train_float_features:
        train_float_df = pd.concat(train_float_features, ignore_index=True)
        scaler.fit(train_float_df)
    
    # 应用标准化器
    for dataset in [train_data, val_data, test_data]:
        for id_val, item in dataset.items():
            for col in feature_list['float']:
                if col in item['features'].columns:
                    if dataset == train_data:  # 训练集已应用cut_off
                        pass
                    else:  # 验证集和测试集应用cut_off
                        item['features'][col] = item['features'][col].apply(
                            lambda x: cut_off(x, dict_cut_off[col][0], dict_cut_off[col][1])
                        )
            
            # 处理布尔特征
            for col in feature_list['bool']:
                if col in item['features'].columns:
                    item['features'][col] = item['features'][col].fillna(False)
                    # 将布尔值转换为浮点型的0和1
                    item['features'][col] = item['features'][col].astype(float)
            
            # 应用标准化
            if len(feature_list['float']) > 0:
                common_cols = set(item['features'].columns) & set(feature_list['float'])
                if common_cols:
                    item['features'][list(common_cols)] = scaler.transform(item['features'][list(common_cols)])
    
    # 创建Dataset对象
    train_dataset = AttentionDataset(train_data, task=task)
    val_dataset = AttentionDataset(val_data, task=task)
    test_dataset = AttentionDataset(test_data, task=task)
    
    # 创建DataLoader对象
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=attention_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=attention_collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=attention_collate_fn
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
    data = load_dataset(data_path, mode='predict')
    processed_data, _ = preprocess_data(data)
    
    # 创建数据集和加载器
    predict_dataset = AttentionDataset(processed_data, task='both')
    
    predict_loader = DataLoader(
        predict_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=attention_collate_fn
    )
    
    return predict_loader, processed_data

def attention_collate_fn(batch):
    """
    自定义的collate函数，用于处理不同长度的输入序列和标签
    
    Args:
        batch: 一批数据样本的列表
        
    Returns:
        填充后的批次数据
    """
    # 分离特征和标签
    has_labels = len(batch[0]) > 2
    
    if has_labels:
        # 提取特征、长度和标签
        features = [item[0] for item in batch]
        lengths = [item[1] for item in batch]
        labels = [item[2] for item in batch]
    else:
        # 预测模式，只有特征和长度
        features = [item[0] for item in batch]
        lengths = [item[1] for item in batch]
    
    # 找出这个批次中最长的序列长度
    max_len = max(lengths)
    
    # 获取特征维度
    feat_dim = features[0].shape[1]
    
    # 创建填充后的特征张量
    padded_features = torch.zeros(len(batch), max_len, feat_dim)
    
    # 创建注意力掩码
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    
    # 填充特征并设置掩码
    for i, (feat, length) in enumerate(zip(features, lengths)):
        padded_features[i, :length] = feat
        attention_mask[i, :length] = True  # True表示有效位置
    
    if not has_labels:
        return padded_features, attention_mask, lengths
    
    # 处理标签
    if isinstance(labels[0], tuple):
        # 对于多任务情况
        # 为xy标签创建填充张量
        xy_labels = [label[0] for label in labels]
        xy_max_len = max([label.shape[0] for label in xy_labels])
        xy_dim = xy_labels[0].shape[1]
        padded_xy = torch.zeros(len(batch), xy_max_len, xy_dim)
        xy_mask = torch.zeros(len(batch), xy_max_len, dtype=torch.bool)
        
        for i, label in enumerate(xy_labels):
            label_len = label.shape[0]
            padded_xy[i, :label_len] = label
            xy_mask[i, :label_len] = True
        
        # 为trace标签创建填充张量
        trace_labels = [label[1] for label in labels]
        trace_max_len = max([label.shape[0] for label in trace_labels])
        trace_dim = trace_labels[0].shape[1]
        padded_trace = torch.zeros(len(batch), trace_max_len, trace_dim)
        trace_mask = torch.zeros(len(batch), trace_max_len, dtype=torch.bool)
        
        for i, label in enumerate(trace_labels):
            label_len = label.shape[0]
            padded_trace[i, :label_len] = label
            trace_mask[i, :label_len] = True
        
        return padded_features, attention_mask, lengths, (padded_xy, xy_mask, padded_trace, trace_mask)
    else:
        # 单任务情况
        max_label_len = max([label.shape[0] for label in labels])
        label_dim = labels[0].shape[1]
        padded_labels = torch.zeros(len(batch), max_label_len, label_dim)
        label_mask = torch.zeros(len(batch), max_label_len, dtype=torch.bool)
        
        for i, label in enumerate(labels):
            label_len = label.shape[0]
            padded_labels[i, :label_len] = label
            label_mask[i, :label_len] = True
            
        return padded_features, attention_mask, lengths, (padded_labels, label_mask)