import os
import time
import json
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

from nn.self_attention import AttentionClassifier
from nn.attention_data_loader import get_data_loaders


def train_attention_model(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    训练自注意力分类模型
    
    Args:
        config: 配置字典，包含训练参数
    
    Returns:
        字典，包含训练结果和最佳模型路径
    """
    # 提取配置参数
    data_path = config.get("data_path", "data/train.csv")
    output_dir = config.get("output_dir", "chart_classification/models")
    mapping_file = config.get("mapping_file", "chart_classification/output/mapping.json")
    feature_dim = config.get("feature_dim", 64)
    hidden_dims = config.get("hidden_dims", [128, 64])
    batch_size = config.get("batch_size", 32)
    num_epochs = config.get("num_epochs", 50)
    learning_rate = config.get("learning_rate", 0.001)
    weight_decay = config.get("weight_decay", 1e-4)
    patience = config.get("patience", 10)  # 早停耐心值
    num_heads = config.get("num_heads", 4)
    dropout_rate = config.get("dropout_rate", 0.2)
    num_workers = config.get("num_workers", 4)
    checkpoint_path = config.get("checkpoint_path", None)
    
    # 固定任务为trace
    task = "trace"
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 获取数据加载器
    data_loaders = get_data_loaders(
        data_path=data_path,
        batch_size=batch_size,
        task=task,
        num_workers=num_workers,
        mapping_file=mapping_file
    )
    
    train_loader = data_loaders["train"]
    val_loader = data_loaders["val"]
    test_loader = data_loaders["test"]
    encoders = data_loaders["encoders"]
    
    # 获取输出类别数量
    num_classes_trace = len(encoders["trace_type"].classes_)
    
    # 初始化模型
    model = AttentionClassifier(
        feature_dim=feature_dim,
        hidden_dims=hidden_dims,
        num_classes=num_classes_trace,
        num_heads=num_heads,
        dropout_rate=dropout_rate
    ).to(device)
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode="max", 
        factor=0.5, 
        patience=5, 
    )
    
    # 加载检查点（如果有）
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if "trace_model" in checkpoint:
            model.load_state_dict(checkpoint["trace_model"])
            print(f"加载trace模型检查点：{checkpoint_path}")
    
    # 定义损失函数（二元交叉熵，因为每个类别是独立的多标签问题）
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    
    # 训练历史记录，添加新的指标
    history = {
        "train_loss": [], "val_loss": [],
        "train_accuracy": [], "val_accuracy": [],
        "train_precision": [], "val_precision": [],
        "train_recall": [], "val_recall": [],
        "train_f1": [], "val_f1": []
    }
    
    # 初始化早停变量
    best_val_f1 = 0.0
    best_epoch = 0
    epochs_no_improve = 0
    best_model_path = None
    
    # 训练循环
    for epoch in range(num_epochs):
        start_time = time.time()
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # 训练阶段
        train_metrics = train_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device
        )
        
        # 验证阶段
        val_metrics = validate_epoch(
            model=model,
            data_loader=val_loader,
            criterion=criterion,
            device=device
        )
        
        # 更新学习率
        scheduler.step(val_metrics["f1"])
        
        # 记录训练历史，添加所有指标
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_accuracy"].append(train_metrics["accuracy"])
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["train_precision"].append(train_metrics["precision"])
        history["val_precision"].append(val_metrics["precision"])
        history["train_recall"].append(train_metrics["recall"])
        history["val_recall"].append(val_metrics["recall"])
        history["train_f1"].append(train_metrics["f1"])
        history["val_f1"].append(val_metrics["f1"])
        
        # 打印当前训练状态，显示所有指标
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{num_epochs} - {epoch_time:.2f}s")
        print(f"Train Loss: {train_metrics['loss']:.4f}, Val Loss: {val_metrics['loss']:.4f}")
        print(f"Train Metrics - Acc: {train_metrics['accuracy']:.4f}, Prec: {train_metrics['precision']:.4f}, Rec: {train_metrics['recall']:.4f}, F1: {train_metrics['f1']:.4f}")
        print(f"Val Metrics   - Acc: {val_metrics['accuracy']:.4f}, Prec: {val_metrics['precision']:.4f}, Rec: {val_metrics['recall']:.4f}, F1: {val_metrics['f1']:.4f}")
        
        # 检查是否需要保存模型
        current_val_f1 = val_metrics["f1"]
        
        if current_val_f1 > best_val_f1:
            best_val_f1 = current_val_f1
            best_epoch = epoch
            epochs_no_improve = 0
            
            # 保存最佳模型
            best_model_path = os.path.join(output_dir, "best_trace_model.pt")
            torch.save(model.state_dict(), best_model_path)
            print(f"已保存最佳trace模型，F1: {current_val_f1:.4f}")
        else:
            epochs_no_improve += 1
        
        # 检查是否需要早停
        if epochs_no_improve >= patience:
            print(f"早停：{patience}个epoch没有改善，停止训练")
            break
    
    # 保存训练历史
    history_path = os.path.join(output_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f)
    
    # 返回训练结果
    result = {
        "best_model_path": best_model_path,
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "history_path": history_path,
        "encoders": encoders
    }
    
    return result


def train_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """
    训练一个epoch
    
    Args:
        model: 模型
        data_loader: 训练数据加载器
        optimizer: 优化器
        criterion: 损失函数
        device: 计算设备
        
    Returns:
        包含训练指标的字典
    """
    # 设置训练模式
    model.train()
    
    # 初始化指标
    metrics = {
        "loss": 0.0,
        "samples_count": 0,
        "correct": 0,
        "total": 0,
        "true": [],
        "pred": []
    }
    
    # 遍历批次数据
    pbar = tqdm(data_loader, desc="Training")
    for batch_idx, batch_data in enumerate(pbar):
        # 解析批次数据
        features, attention_mask, lengths = batch_data[0], batch_data[1], batch_data[2]
        features = features.to(device)
        attention_mask = attention_mask.to(device)
        
        batch_size = features.size(0)
        metrics["samples_count"] += batch_size
        
        # 如果有标签，提取标签数据
        if len(batch_data) > 3:
            labels = batch_data[3]
            
            # 提取标签和掩码 - 只保留每个样本的第一个有效标签
            task_labels, task_mask = labels[0].to(device), labels[1].to(device)
            
            # 只保留第一个序列的标签 [batch_size, class_num]
            task_labels = task_labels[:, 0, :]
            
            # 训练模型
            optimizer.zero_grad()
            logits, _ = model(features, attention_mask)
            
            # 计算损失
            loss = compute_loss(logits, task_labels, criterion)
            loss.backward()
            optimizer.step()
            
            metrics["loss"] += loss.item() * batch_size
            
            # 收集预测结果用于计算指标
            update_prediction_metrics(
                metrics, logits, task_labels
            )
        
        # 更新进度条
        pbar.set_description(f"Train Loss: {metrics['loss']/metrics['samples_count']:.4f}")
    
    # 计算平均损失和各项指标
    metrics["loss"] = metrics["loss"] / metrics["samples_count"]
    
    # 计算准确率
    metrics["accuracy"] = metrics["correct"] / metrics["total"] if metrics["total"] > 0 else 0.0
    
    # 计算其他指标
    metrics["f1"] = calculate_f1_score(metrics["true"], metrics["pred"])
    metrics["precision"] = precision_score(
        metrics["true"], metrics["pred"], average="macro", zero_division=0
    )
    metrics["recall"] = recall_score(
        metrics["true"], metrics["pred"], average="macro", zero_division=0
    )
    
    return metrics


def validate_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """
    验证一个epoch
    
    Args:
        model: 模型
        data_loader: 验证数据加载器
        criterion: 损失函数
        device: 计算设备
        
    Returns:
        包含验证指标的字典
    """
    # 设置评估模式
    model.eval()
    
    # 初始化指标
    metrics = {
        "loss": 0.0,
        "samples_count": 0,
        "correct": 0,
        "total": 0,
        "true": [],
        "pred": []
    }
    
    # 禁用梯度计算以加速评估
    with torch.no_grad():
        for batch_data in tqdm(data_loader, desc="Validating"):
            # 解析批次数据
            features, attention_mask, lengths = batch_data[0], batch_data[1], batch_data[2]
            features = features.to(device)
            attention_mask = attention_mask.to(device)
            
            batch_size = features.size(0)
            metrics["samples_count"] += batch_size
            
            # 如果有标签，提取标签数据
            if len(batch_data) > 3:
                labels = batch_data[3]
                
                # 提取标签和掩码
                task_labels, task_mask = labels[0].to(device), labels[1].to(device)
                
                # [batch_size, class_num]
                task_labels = task_labels[:, 0, :]
                
                # 评估模型
                logits, _ = model(features, attention_mask)
                
                # 计算损失
                loss = compute_loss(logits, task_labels, criterion)
                
                metrics["loss"] += loss.item() * batch_size
                
                # 收集预测结果用于计算指标
                update_prediction_metrics(
                    metrics, logits, task_labels
                )
    
    # 计算平均损失和各项指标
    metrics["loss"] = metrics["loss"] / metrics["samples_count"]
    
    # 计算准确率
    metrics["accuracy"] = metrics["correct"] / metrics["total"] if metrics["total"] > 0 else 0.0
    
    # 计算其他指标
    metrics["f1"] = calculate_f1_score(metrics["true"], metrics["pred"])
    metrics["precision"] = precision_score(
        metrics["true"], metrics["pred"], average="macro", zero_division=0
    )
    metrics["recall"] = recall_score(
        metrics["true"], metrics["pred"], average="macro", zero_division=0
    )
    
    return metrics


def compute_loss(logits, labels, criterion):
    """
    计算损失
    
    Args:
        logits: 模型输出的logits，形状为[batch_size, class_num]
        labels: 标签，形状为[batch_size, class_num]
        criterion: 损失函数
        
    Returns:
        平均损失
    """
    # 直接计算每个样本的损失
    loss = criterion(logits, labels)  # [batch_size, class_num]
    
    # 对类别维度求平均，得到每个样本的平均损失
    loss = loss.mean(dim=-1)  # [batch_size]
    
    # 计算所有样本的平均损失
    loss = loss.mean()
    
    return loss


def update_prediction_metrics(metrics, logits, labels):
    """
    更新预测指标
    
    Args:
        metrics: 指标字典
        logits: 模型输出的logits，形状为[batch_size, class_num]
        labels: 真实标签，形状为[batch_size, class_num]
    """
    # 对logits应用sigmoid获取概率
    probs = torch.sigmoid(logits)  # [batch_size, class_num]
    
    # 预测类别（概率大于0.5认为是正类）
    preds = (probs > 0.5).float()  # [batch_size, class_num]
    
    # 将预测和标签转换为numpy数组以计算指标
    true_labels = labels.cpu().numpy()
    pred_labels = preds.cpu().numpy()
    
    # 对于多标签分类，需要将每个样本的所有类别展平
    true_labels_flat = true_labels.reshape(-1)
    pred_labels_flat = pred_labels.reshape(-1)
    
    # 更新指标
    metrics["true"].extend(true_labels_flat.tolist())
    metrics["pred"].extend(pred_labels_flat.tolist())
    
    # 计算正确预测的数量
    correct = (preds == labels).float().sum().item()
    total = labels.numel()  # 总元素数量
    
    metrics["correct"] += correct
    metrics["total"] += total


def calculate_f1_score(y_true, y_pred):
    """
    计算F1分数
    
    Args:
        y_true: 真实标签列表
        y_pred: 预测标签列表
        
    Returns:
        F1分数
    """
    if len(y_true) == 0:
        return 0.0
    
    try:
        return f1_score(y_true, y_pred, average="macro")
    except:
        return 0.0


def evaluate_model_on_test(
    model_path: str, 
    config: Dict[str, Any], 
    data_loaders: Optional[Dict[str, Any]] = None
) -> Dict[str, float]:
    """
    在测试集上评估模型性能
    
    Args:
        model_path: 模型路径
        config: 配置字典
        data_loaders: 可选的数据加载器字典，如果为None则重新创建
        
    Returns:
        包含评估指标的字典
    """
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 如果没有提供数据加载器，则创建
    if data_loaders is None:
        data_loaders = get_data_loaders(
            data_path=config.get("data_path"),
            batch_size=config.get("batch_size", 32),
            task="trace",
            num_workers=config.get("num_workers", 4)
        )
    
    test_loader = data_loaders["test"]
    encoders = data_loaders["encoders"]
    
    # 获取输出类别数量
    num_classes_trace = len(encoders["trace_type"].classes_)
    
    # 初始化模型
    model = AttentionClassifier(
        feature_dim=config.get("feature_dim", 64),
        hidden_dims=config.get("hidden_dims", [128, 64]),
        num_classes=num_classes_trace,
        num_heads=config.get("num_heads", 4),
        dropout_rate=config.get("dropout_rate", 0.2)
    ).to(device)
    
    # 加载模型
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"已加载模型: {model_path}")
    
    # 评估模型
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    metrics = validate_epoch(
        model=model, 
        data_loader=test_loader, 
        criterion=criterion, 
        device=device
    )
    
    # 打印评估结果
    print("\n测试集评估结果:")
    print(f"Loss: {metrics['loss']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    
    # 计算并打印类别分布
    class_distribution = calculate_class_distribution(metrics["true"], encoders["trace_type"])
    print("\n类别分布:")
    for class_name, count in class_distribution.items():
        print(f"{class_name}: {count} 样本")
    
    return metrics


def calculate_class_distribution(y_true, encoder):
    """
    计算测试数据中各类别的分布
    
    Args:
        y_true: 真实标签列表
        encoder: 标签编码器
        
    Returns:
        各类别的样本数量统计
    """
    # 确定类别数量
    num_classes = len(encoder.classes_)
    
    # 将一维列表重构为二维数组
    n_samples = len(y_true) // num_classes
    y_true_reshaped = np.array(y_true).reshape(n_samples, num_classes)
    
    # 统计每个类别的样本数
    distribution = {}
    for i, class_name in enumerate(encoder.classes_):
        count = np.sum(y_true_reshaped[:, i])
        distribution[class_name] = int(count)
    
    return distribution


if __name__ == "__main__":
    config = {
        #"data_path": "features/fix_nl2chart_feature.csv",
        #"output_dir": "chart_classification/models",
        #"mapping_file": "chart_classification/output/mapping.json",
        "data_path": "features/vizml_feature.csv",
        "output_dir": "chart_classification/models_vizml",
        "mapping_file": "chart_classification/output/mapping_vizml.json",
        "feature_dim": 81,
        "hidden_dims": [1024, 1024],
        "batch_size": 16,
        "num_epochs": 1000,
        "learning_rate": 5e-4,
        "weight_decay": 1e-4,
        "patience": 20,
        "num_heads": 8,
        "dropout_rate": 0.1,
        "num_workers": 0
    }
    
    # 训练模型
    results = train_attention_model(config)
    print("训练完成，最佳模型路径:", results["best_model_path"])
    
    # 在测试集上评估最佳模型
    print("\n在测试集上评估最佳模型...")
    evaluate_model_on_test(results["best_model_path"], config)