import os
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

# 导入自定义模块
from nn.fc import FcNN
from nn.dataloader import get_data_loaders

import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
mpl.rcParams['axes.unicode_minus'] = False

class EarlyStopping:
    """早停机制，避免过拟合"""
    def __init__(self, patience=10, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, val_loss, model):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            
    def save_checkpoint(self, val_loss, model):
        """保存最佳模型"""
        torch.save(model.state_dict(), self.path)
        print(f'验证损失减少 ({self.best_score:.6f} --> {-val_loss:.6f}). 保存模型...')
        
def train_epoch(model, train_loader, criterion, optimizer, device, task='both'):
    """训练一个epoch"""
    model.train()
    train_loss = 0
    
    # 使用tqdm添加进度条
    progress_bar = tqdm(train_loader, desc="Training")
    
    for data, target in progress_bar:
        # 将数据移动到GPU（如果可用）
        data = data.to(device)
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播
        output = model(data)
        
        # 根据任务类型计算损失
        if task == 'both':
            target_xy, target_trace = target
            target_xy = target_xy.to(device)
            target_trace = target_trace.to(device)
            
            if isinstance(criterion, tuple):
                criterion_xy, criterion_trace = criterion
                loss_xy = criterion_xy(output[0], target_xy)
                loss_trace = criterion_trace(output[1], target_trace)
                loss = loss_xy + loss_trace
            else:
                raise ValueError("当任务为'both'时，criterion应为(criterion_xy, criterion_trace)元组")
        else:
            target = target.to(device)
            loss = criterion(output, target)
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        
        # 累加batch损失
        train_loss += loss.item() * data.size(0)
        
        # 更新进度条
        progress_bar.set_postfix({"batch_loss": loss.item()})
    
    # 计算整个epoch的平均损失
    train_loss = train_loss / len(train_loader.dataset)
    
    return train_loss

def validate(model, val_loader, criterion, device, task='both'):
    """验证模型"""
    model.eval()
    val_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in val_loader:
            data = data.to(device)
            
            # 前向传播
            output = model(data)
            
            # 根据任务类型计算损失
            if task == 'both':
                target_xy, target_trace = target
                target_xy = target_xy.to(device)
                target_trace = target_trace.to(device)
                
                if isinstance(criterion, tuple):
                    criterion_xy, criterion_trace = criterion
                    loss_xy = criterion_xy(output[0], target_xy)
                    loss_trace = criterion_trace(output[1], target_trace)
                    loss = loss_xy + loss_trace
                    
                    # 记录预测结果 - 不需要应用argmax
                    all_preds.append((output[0].cpu(), output[1].cpu()))
                    all_targets.append((target_xy.cpu(), target_trace.cpu()))
                else:
                    raise ValueError("当任务为'both'时，criterion应为(criterion_xy, criterion_trace)元组")
            else:
                target = target.to(device)
                loss = criterion(output, target)
                
                # 记录预测结果 - 不需要应用argmax
                all_preds.append(output.cpu())
                all_targets.append(target.cpu())
            
            val_loss += loss.item() * data.size(0)
    
    # 计算平均损失
    val_loss = val_loss / len(val_loader.dataset)
    
    # 计算准确率
    accuracy = calc_accuracy(all_preds, all_targets, task)
    
    return val_loss, accuracy

def calc_accuracy(preds, targets, task='both'):
    """计算准确率"""
    if task == 'both':
        # 合并所有batch的预测和标签
        all_pred_xy = torch.cat([p[0] for p in preds])
        all_pred_trace = torch.cat([p[1] for p in preds])
        all_target_xy = torch.cat([t[0] for t in targets])
        all_target_trace = torch.cat([t[1] for t in targets])
        
        # 找到独热编码中最大值的索引
        pred_xy_indices = torch.argmax(all_pred_xy, dim=1)
        pred_trace_indices = torch.argmax(all_pred_trace, dim=1)
        target_xy_indices = torch.argmax(all_target_xy, dim=1)
        target_trace_indices = torch.argmax(all_target_trace, dim=1)
        
        # 计算两个任务的准确率
        acc_xy = (pred_xy_indices == target_xy_indices).float().mean().item()
        acc_trace = (pred_trace_indices == target_trace_indices).float().mean().item()
        
        return {'xy': acc_xy, 'trace': acc_trace, 'avg': (acc_xy + acc_trace) / 2}
    else:
        # 合并所有batch的预测和标签
        all_preds = torch.cat(preds)
        all_targets = torch.cat(targets)
        
        # 找到独热编码中最大值的索引
        pred_indices = torch.argmax(all_preds, dim=1)
        target_indices = torch.argmax(all_targets, dim=1)
        
        # 计算准确率
        accuracy = (pred_indices == target_indices).float().mean().item()
        return accuracy

def train_model(config):
    """训练模型的主函数
    
    Args:
        config: 包含所有训练参数的字典
    """
    # 设置随机种子
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    # 选择设备
    device = torch.device('cuda' if torch.cuda.is_available() and config['cuda'] else 'cpu')
    print(f"使用设备: {device}")
    
    # 获取数据加载器
    data_loaders = get_data_loaders(
        config['data_path'],
        batch_size=config['batch_size'],
        train_ratio=config['train_ratio'],
        val_ratio=config['val_ratio'],
        test_ratio=config['test_ratio'],
        task=config['task'],
        num_workers=config['num_workers'],
        random_seed=config['seed']
    )
    
    train_loader = data_loaders['train']
    val_loader = data_loaders['val']
    test_loader = data_loaders['test']
    encoders = data_loaders['encoders']
    
    # 确定输入维度和输出维度
    sample = next(iter(train_loader))
    input_dim = sample[0].shape[1]

    if config['task'] == 'xy':
        # 获取xy独热编码的维度
        xy_cols = [col for col in train_loader.dataset.labels.columns if col.startswith('xy_')]
        output_dim = len(xy_cols)
    elif config['task'] == 'trace':
        # 获取trace独热编码的维度
        trace_cols = [col for col in train_loader.dataset.labels.columns if col.startswith('trace_')]
        output_dim = len(trace_cols)
    else:  # 'both'
        xy_cols = [col for col in train_loader.dataset.labels.columns if col.startswith('xy_')]
        trace_cols = [col for col in train_loader.dataset.labels.columns if col.startswith('trace_')]
        output_dim_xy = len(xy_cols)
        output_dim_trace = len(trace_cols)
    
    # 创建模型
    if config['task'] == 'both':
        # 多任务模型
        class MultiTaskModel(nn.Module):
            def __init__(self, input_dim, hidden_dims, output_dim_xy, output_dim_trace, dropout_rate=0.2):
                super(MultiTaskModel, self).__init__()
                # 共享特征提取器
                self.feature_extractor = FcNN(input_dim, hidden_dims[:-1], hidden_dims[-1], dropout_rate)
                # 特定任务的头部
                self.xy_head = nn.Linear(hidden_dims[-1], output_dim_xy)
                self.trace_head = nn.Linear(hidden_dims[-1], output_dim_trace)
                
            def forward(self, x):
                # 特征提取
                features = self.feature_extractor(x)
                # 任务特定输出
                xy_out = self.xy_head(features)
                trace_out = self.trace_head(features)
                return xy_out, trace_out
        
        model = MultiTaskModel(input_dim, config['hidden_dims'], output_dim_xy, output_dim_trace, config['dropout_rate'])
    else:
        # 单任务模型
        model = FcNN(input_dim, config['hidden_dims'], output_dim, config['dropout_rate'])
    
    model.to(device)
    print(f"模型结构:\n{model}")
    
    # 设置损失函数和优化器
    if config['task'] == 'both':
        criterion = (nn.BCEWithLogitsLoss(), nn.BCEWithLogitsLoss())
    else:
        criterion = nn.BCEWithLogitsLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # 早停机制
    early_stopping = EarlyStopping(patience=config['early_stopping_patience'], 
                                  path=os.path.join(config['output_dir'], 'best_model.pt'))
    
    # 创建输出目录
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # 训练日志
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    # 训练循环
    for epoch in range(config['epochs']):
        start_time = time.time()
        
        # 训练一个epoch
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, config['task'])
        
        # 在验证集上评估
        val_loss, val_accuracy = validate(model, val_loader, criterion, device, config['task'])
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 记录日志
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        # 输出当前epoch的结果
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{config['epochs']} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | ", end='')
        
        if config['task'] == 'both':
            print(f"Val Acc (XY): {val_accuracy['xy']:.4f} | Val Acc (Trace): {val_accuracy['trace']:.4f} | "
                  f"Time: {epoch_time:.2f}s")
        else:
            print(f"Val Acc: {val_accuracy:.4f} | Time: {epoch_time:.2f}s")
        
        # 早停检查
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("早停! 训练终止.")
            break
    
    # 加载最佳模型
    model.load_state_dict(torch.load(os.path.join(config['output_dir'], 'best_model.pt')))
    
    # 在测试集上评估最终模型
    test_loss, test_accuracy = validate(model, test_loader, criterion, device, config['task'])
    print("\n测试结果:")
    if config['task'] == 'both':
        print(f"Test Loss: {test_loss:.4f} | Test Acc (XY): {test_accuracy['xy']:.4f} | "
              f"Test Acc (Trace): {test_accuracy['trace']:.4f}")
    else:
        print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_accuracy:.4f}")
    
    # 保存训练历史
    history = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'val_accuracy': val_accuracies,
        'test_loss': test_loss,
        'test_accuracy': test_accuracy
    }
    
    # 绘制训练曲线
    plt.figure(figsize=(12, 5))
    
    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('训练和验证损失')
    
    # 准确率曲线
    plt.subplot(1, 2, 2)
    if config['task'] == 'both':
        xy_acc = [acc['xy'] for acc in val_accuracies]
        trace_acc = [acc['trace'] for acc in val_accuracies]
        avg_acc = [acc['avg'] for acc in val_accuracies]
        
        plt.plot(xy_acc, label='XY准确率')
        plt.plot(trace_acc, label='Trace准确率')
        plt.plot(avg_acc, label='平均准确率')
    else:
        plt.plot(val_accuracies, label='验证准确率')
    
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('验证准确率')
    
    plt.tight_layout()
    plt.savefig(os.path.join(config['output_dir'], 'training_curves.png'))
    
    print(f"训练已完成。最佳模型保存至: {os.path.join(config['output_dir'], 'best_model.pt')}")
    
    return model, history

# 默认配置
DEFAULT_CONFIG = {
    # 数据参数
    'data_path': None,  # 必须提供
    'output_dir': './output',
    'task': 'both',  # 'xy', 'trace', 'both'
    
    # 训练参数
    'batch_size': 64,
    'epochs': 100,
    'lr': 0.001,
    'weight_decay': 1e-5,
    'early_stopping_patience': 10,
    
    # 模型参数
    'hidden_dims': [1000,1000,1000],
    'dropout_rate': 0.2,
    
    # 数据集划分参数
    'train_ratio': 0.7,
    'val_ratio': 0.15,
    'test_ratio': 0.15,
    
    # 其他参数
    'num_workers': 0,
    'seed': 42,
    'cuda': False
}

def run_training(config_override=None):
    """运行训练，允许覆盖默认配置
    
    Args:
        config_override: 包含要覆盖的配置项的字典
        
    Returns:
        训练好的模型和训练历史
    """
    # 合并默认配置和自定义配置
    config = DEFAULT_CONFIG.copy()
    if config_override:
        config.update(config_override)
    
    # 检查必要参数
    if config['data_path'] is None:
        raise ValueError("必须提供'data_path'参数")
    
    # 验证任务类型
    if config['task'] not in ['xy', 'trace', 'both']:
        raise ValueError("任务类型必须是'xy'、'trace'或'both'")
    
    # 训练模型
    return train_model(config)

if __name__ == '__main__':
    # 使用示例
    config = {
        'data_path': "features/fix_nl2chart_feature.csv",
        'output_dir': 'chart_classifi_models/nn_model',
        'task': 'trace',  # 'xy', 'trace', 'both'
        
        # 训练参数
        'batch_size': 32,
        'epochs': 100,
        'lr': 0.01,
        'weight_decay': 0,
        'early_stopping_patience': 50,
        
        # 模型参数
        'hidden_dims': [1000,1000,1000],
        'dropout_rate': 0.00,
        
        # 数据集划分参数
        'train_ratio': 0.7,
        'val_ratio': 0.15,
        'test_ratio': 0.15,
        
        # 其他参数
        'num_workers': 0,
        'seed': 42,
        'cuda': False
    }
    
    model, history = run_training(config)