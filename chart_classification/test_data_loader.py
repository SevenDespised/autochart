import os
import pandas as pd
import numpy as np
import torch
import sys
# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nn.attention_data_loader import (
    load_dataset, 
    preprocess_data, 
    AttentionDataset, 
    get_data_loaders,
)

def test_load_dataset(data_path: str):
    """测试数据集加载函数"""
    print("\n=== 测试数据集加载函数 ===")
    
    # 测试训练模式
    train_data = load_dataset(data_path, mode='train')
    # 取前10条数据进行测试
    train_data = {k: train_data[k] for k in list(train_data)[:10]}
    print(f"训练模式 - 加载的记录数: {len(train_data)}")
    
    # 打印第一个数据条目的结构
    first_id = list(train_data.keys())[0]
    print(f"数据结构示例 (ID={first_id}):")
    print(f"  x_data的长度: {len(train_data[first_id]['x_data'])}")
    print(f"  包含的键: {list(train_data[first_id].keys())}")
    
    # 如果有y_data，打印其结构
    if 'y_data' in train_data[first_id]:
        print(f"  y_data包含的标签: {list(train_data[first_id]['y_data'].keys())}")
        for label_type, labels in train_data[first_id]['y_data'].items():
            print(f"  {label_type}标签数量: {len(labels)}")

    
    return train_data

def test_preprocess_data(data: dict):
    """测试数据预处理功能"""
    print("\n=== 测试数据预处理 ===")
    
    processed_data, encoders = preprocess_data(data)
    print(f"预处理后的记录数: {len(processed_data)}")
    
    # 打印第一个数据条目的结构
    first_id = list(processed_data.keys())[0]
    print(f"处理后的数据结构示例 (ID={first_id}):")
    print(f"  特征形状: {processed_data[first_id]['features'].shape if 'features' in processed_data[first_id] else None}")
    # 如果有标签，打印其结构
    if 'labels' in processed_data[first_id]:
        print(f"  包含的标签类型: {list(processed_data[first_id]['labels'].keys())}")
        for label_type, label_data in processed_data[first_id]['labels'].items():
            print(f"  {label_type}标签形状: {label_data.shape}")
    
    # 打印编码器信息
    print(f"编码器: {list(encoders.keys())}")
    for name, encoder in encoders.items():
        print(f"{name} 标签映射: {dict(zip(encoder.classes_[:3], range(3)))}...")
    
    return processed_data, encoders

def test_chart_dataset(processed_data: dict):
    """测试AttentionDataset类"""
    print("\n=== 测试AttentionDataset类 ===")
    
    # 测试xy任务
    dataset_xy = AttentionDataset(processed_data, task='xy')
    print(f"XY任务 - 数据集大小: {len(dataset_xy)}")
    if len(dataset_xy) > 0:
        sample = dataset_xy[0]
        if isinstance(sample, tuple) and len(sample) == 2:
            X_sample, y_sample = sample
            print(f"样本: X形状 {X_sample.shape}, y类型 {type(y_sample)}")
            if isinstance(y_sample, torch.Tensor):
                print(f"y形状 {y_sample.shape}")
    
    # 测试trace任务
    dataset_trace = AttentionDataset(processed_data, task='trace')
    print(f"Trace任务 - 数据集大小: {len(dataset_trace)}")
    if len(dataset_trace) > 0:
        sample = dataset_trace[0]
        if isinstance(sample, tuple) and len(sample) == 2:
            X_sample, y_sample = sample
            print(f"样本: X形状 {X_sample.shape}, y类型 {type(y_sample)}")
            if isinstance(y_sample, torch.Tensor):
                print(f"y形状 {y_sample.shape}")
    
    # 测试both任务
    dataset_both = AttentionDataset(processed_data, task='both')
    print(f"Both任务 - 数据集大小: {len(dataset_both)}")
    if len(dataset_both) > 0:
        sample = dataset_both[0]
        if isinstance(sample, tuple) and len(sample) == 2:
            X_sample, y_sample = sample
            print(f"样本: X形状 {X_sample.shape}, y类型 {type(y_sample)}")
            if isinstance(y_sample, tuple):
                print(f"y_xy形状 {y_sample[0].shape}, y_trace形状 {y_sample[1].shape}")
            elif isinstance(y_sample, torch.Tensor):
                print(f"y形状 {y_sample.shape}")

def test_data_loaders(data_path: str):
    """测试数据加载器"""
    print("\n=== 测试数据加载器 ===")
    loaders = get_data_loaders(
        data_path,
        batch_size=16,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        task='both',
        num_workers=0  # Windows上测试时用0
    )
    
    print(f"创建的数据加载器: {list(loaders.keys())}")
    
    # 测试训练加载器
    train_loader = loaders['train']
    print(f"训练加载器 - 批次数: {len(train_loader)}")
    
    # 取一个批次
    for batch in train_loader:
        X_batch = batch[0]
        y_batch = batch[1]
        print(f"训练批次 - X形状: {X_batch.shape}")
        if isinstance(y_batch, tuple):
            print(f"y_xy形状: {y_batch[0].shape}, y_trace形状: {y_batch[1].shape}")
        else:
            print(f"y形状: {y_batch.shape}")
        break  # 只查看第一个批次


def main():
    """主测试函数"""
    # 使用模拟数据或真实数据
    test_data_path = "features/fix_nl2chart_feature.csv"

    
    # 测试数据加载
    #data = test_load_dataset(test_data_path)
    
    # 测试数据预处理
    #processed_data, _ = test_preprocess_data(data)
    
    # 测试数据集类
    #test_chart_dataset(processed_data)
    
    # 测试数据加载器
    test_data_loaders(test_data_path)
    
    print("\n✓ 所有测试完成!")

if __name__ == "__main__":
    main()