import torch
import torch.nn as nn
import torch.nn.functional as F

class FcNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate=0.2):
        """
        可变隐藏层的全连接神经网络
        
        参数:
            input_dim (int): 输入特征的维度
            hidden_dims (list): 隐藏层维度的列表，列表长度决定隐藏层数量
            output_dim (int): 输出的维度（如分类任务中的类别数）
            dropout_rate (float): Dropout比率，用于防止过拟合
        """
        super(FcNN, self).__init__()
        
        # 保存网络配置
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        
        # 动态创建网络层
        self.layers = nn.ModuleList()
        
        # 输入层 -> 第一个隐藏层
        layer_dims = [input_dim] + hidden_dims
        
        # 创建所有隐藏层
        for i in range(len(layer_dims) - 1):
            layer_block = nn.Sequential(
                nn.Linear(layer_dims[i], layer_dims[i+1]),
                nn.BatchNorm1d(layer_dims[i+1]),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            )
            self.layers.append(layer_block)
        
        # 输出层
        self.output = nn.Linear(hidden_dims[-1], output_dim)
        
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入张量，形状为 [batch_size, input_dim]
            
        返回:
            输出张量，形状为 [batch_size, output_dim]
        """
        # 通过所有隐藏层
        for layer in self.layers:
            x = layer(x)
        
        # 输出层
        x = self.output(x)
        
        return x
    
    def __repr__(self):
        """返回更详细的模型结构描述"""
        return f"FlexibleNN(input_dim={self.input_dim}, hidden_dims={self.hidden_dims}, output_dim={self.output_dim}, dropout_rate={self.dropout_rate})"
