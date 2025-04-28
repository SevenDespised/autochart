import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionClassifier(nn.Module):
    def __init__(self, feature_dim, hidden_dims, num_classes, num_heads=4, dropout_rate=0.2):
        """
        基于自注意力机制的数据表分类模型
        
        参数:
            feature_dim (int): 每个特征向量的维度
            hidden_dims (list): 全连接层的隐藏层维度列表
            num_classes (int): 分类类别数量
            num_heads (int): 多头注意力的头数
            dropout_rate (float): Dropout比率
        """
        super(AttentionClassifier, self).__init__()
        
        # 保存配置
        self.feature_dim = feature_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        
        # 特征投影
        power = 1
        while power < feature_dim or power % num_heads != 0:
            power *= 2
        self.projected_dim = power
        
        if self.projected_dim != feature_dim:
            print(f"特征维度从 {feature_dim} 投影到 {self.projected_dim}")
        
        # 投影层
        self.input_projection = nn.Linear(feature_dim, self.projected_dim) if self.projected_dim != feature_dim else nn.Identity()
        
        # 自注意力层
        self.self_attention = nn.MultiheadAttention(
            embed_dim=self.projected_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(self.projected_dim)
        
        # 移除输出投影层，直接使用projected_dim作为分类网络的输入维度
        
        # 特征提取网络
        self.feature_extractor = nn.ModuleList()
        
        # 构建全连接分类器
        layer_dims = [self.projected_dim] + hidden_dims
        for i in range(len(layer_dims) - 1):
            layer_block = nn.Sequential(
                nn.Linear(layer_dims[i], layer_dims[i+1]),
                nn.BatchNorm1d(layer_dims[i+1]),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            )
            self.feature_extractor.append(layer_block)
        
        # 输出层
        self.classifier = nn.Linear(hidden_dims[-1], num_classes)
    
    def forward(self, x, attention_mask=None):
        """
        前向传播
        
        参数:
            x: 输入张量，形状为 [batch_size, num_columns, feature_dim]，
               其中num_columns是数据列数量（已填充到批次内最大长度）
            attention_mask: 注意力掩码，形状为 [batch_size, num_columns]，
                          True表示实际数据，False表示填充位置
            
        返回:
            logits: 分类logits，形状为 [batch_size, num_classes]
            attn_weights: 注意力权重，形状为 [batch_size, num_columns, num_columns]
        """
        batch_size, seq_len, _ = x.shape
        
        # 应用输入投影，调整特征维度以适应多头注意力
        if self.projected_dim != self.feature_dim:
            # 重塑张量以便应用线性层
            x_reshaped = x.reshape(-1, self.feature_dim)
            x_projected = self.input_projection(x_reshaped)
            x = x_projected.reshape(batch_size, seq_len, self.projected_dim)
        
        # 自注意力机制，计算不同数据列之间的关系
        if attention_mask is not None:
            # 创建自注意力层需要的key_padding_mask（形状为[batch_size, seq_len]）
            # 注意：key_padding_mask中，True表示要mask的位置，与attention_mask相反
            key_padding_mask = ~attention_mask
            
            # PyTorch的MultiheadAttention需要key_padding_mask
            attn_output, attn_weights = self.self_attention(
                x, x, x,
                key_padding_mask=key_padding_mask
            )
        else:
            attn_output, attn_weights = self.self_attention(x, x, x)
        
        # 如果attn_weights是4D的[batch_size, num_heads, seq_len, seq_len]，则取平均值
        if len(attn_weights.shape) == 4:
            attn_weights = attn_weights.mean(dim=1)  # [batch_size, seq_len, seq_len]
            
        # 残差连接和层归一化
        attn_output = self.layer_norm(attn_output + x)
        
        # 汇总所有数据列的特征
        if attention_mask is not None:
            # 创建掩码来屏蔽填充位置的注意力权重
            mask = attention_mask.unsqueeze(2) & attention_mask.unsqueeze(1)  # [batch_size, seq_len, seq_len]
            
            # 将填充位置的注意力权重置为0
            attn_weights = attn_weights.masked_fill(~mask, 0.0)
            
            # 计算有效位置的注意力权重均值
            column_weights = attn_weights.sum(dim=1, keepdim=True)  # [batch_size, 1, seq_len]
            
            # 将权重归一化，避免零除问题
            column_weights = column_weights / (column_weights.sum(dim=2, keepdim=True) + 1e-10)
        else:
            # 如果没有掩码，使用普通的均值
            column_weights = attn_weights.mean(dim=1, keepdim=True)  # [batch_size, 1, num_columns]
        
        weighted_features = torch.bmm(column_weights, attn_output)  # [batch_size, 1, projected_dim]
        features = weighted_features.squeeze(1)  # [batch_size, projected_dim]
        
        # 通过特征提取网络
        for layer in self.feature_extractor:
            features = layer(features)
        
        # 最终分类
        logits = self.classifier(features)
        
        return logits, attn_weights
    
    def __repr__(self):
        """返回模型结构描述"""
        return f"AttentionClassifier(feature_dim={self.feature_dim}, projected_dim={self.projected_dim}, hidden_dims={self.hidden_dims}, num_classes={self.num_classes}, num_heads={self.num_heads})"