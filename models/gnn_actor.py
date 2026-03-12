# -*- coding: utf-8 -*-
"""
文件名称：gnn_actor.py
作者：denny
创建日期：2026-03-12
描述：GNN Actor-Critic 模型，使用 PyTorch 和 PyTorch Geometric
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.data import Data, Batch
import numpy as np


class GNNLayer(MessagePassing):
    """自定义 GNN 层，支持边权重"""
    
    def __init__(self, in_channels, out_channels, edge_dim=1):
        super(GNNLayer, self).__init__(aggr='mean')
        self.mlp = nn.Sequential(
            nn.Linear(in_channels * 2 + edge_dim, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(in_channels + out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )

    def forward(self, x, edge_index, edge_attr=None):
        """
        Args:
            x: [N, in_channels] 节点特征
            edge_index: [2, E] 边索引
            edge_attr: [E, edge_dim] 边特征
        """
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        """构建消息"""
        # x_i: [E, in_channels] 目标节点特征
        # x_j: [E, in_channels] 源节点特征
        # edge_attr: [E, edge_dim] 边特征
        if edge_attr is None:
            tmp = torch.cat([x_i, x_j], dim=1)
        else:
            tmp = torch.cat([x_i, x_j, edge_attr], dim=1)
        return self.mlp(tmp)

    def update(self, aggr_out, x):
        """更新节点特征"""
        tmp = torch.cat([x, aggr_out], dim=1)
        return self.update_mlp(tmp)


class GNNFeatureExtractor(nn.Module):
    """
    GNN 特征提取器
    处理图结构输入，输出图级别嵌入
    """
    
    def __init__(self, node_dim=5, edge_dim=1, hidden_dim=64, output_dim=128, num_layers=3):
        super(GNNFeatureExtractor, self).__init__()
        
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # 输入投影层
        self.node_encoder = nn.Linear(node_dim, hidden_dim)
        
        # GNN 层
        self.gnn_layers = nn.ModuleList([
            GNNLayer(hidden_dim, hidden_dim, edge_dim) 
            for _ in range(num_layers)
        ])
        
        # 层归一化
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) 
            for _ in range(num_layers)
        ])
        
        # 输出投影层
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, node_feat, edge_index, edge_attr, batch=None):
        """
        Args:
            node_feat: [N, node_dim] 或 [B, N_max, node_dim] (带填充)
            edge_index: [2, E] 或 [B, 2, E_max] (带填充)
            edge_attr: [E, edge_dim] 或 [B, E_max, edge_dim] (带填充)
            batch: [N] 每个节点所属的图索引 (单图时为 None)
            
        Returns:
            graph_embedding: [B, output_dim] 图级别嵌入
        """
        # 处理带填充的批次输入
        if node_feat.dim() == 3:
            # node_feat: [B, N_max, node_dim]
            # edge_index: [B, 2, E_max]
            # edge_attr: [B, E_max, edge_dim]
            return self._forward_batch(node_feat, edge_index, edge_attr)
        
        # 单图处理
        x = self.node_encoder(node_feat)
        
        for gnn_layer, layer_norm in zip(self.gnn_layers, self.layer_norms):
            x_new = gnn_layer(x, edge_index, edge_attr)
            x_new = layer_norm(x_new)
            x = F.relu(x_new)
        
        # 全局平均池化
        if batch is None:
            # 单图，直接对所有节点取平均
            graph_embedding = x.mean(dim=0, keepdim=True)
        else:
            graph_embedding = global_mean_pool(x, batch)
        
        graph_embedding = self.output_mlp(graph_embedding)
        return graph_embedding

    def _forward_batch(self, node_feat, edge_index, edge_attr):
        """处理批次输入 (带填充的情况)"""
        batch_size = node_feat.shape[0]
        device = node_feat.device
        
        # 构建 PyG 的 Batch 对象
        data_list = []
        for i in range(batch_size):
            # 获取实际的节点数和边数
            n_nodes = (node_feat[i].abs().sum(dim=1) > 0).sum().item()
            
            # 对于边，需要找到有效的边 (非零边索引)
            valid_edges = (edge_index[i][0] < n_nodes) & (edge_index[i][1] < n_nodes) & \
                         (edge_index[i][0] >= 0) & (edge_index[i][1] >= 0)
            n_edges = valid_edges.sum().item()
            
            if n_nodes == 0:
                n_nodes = 1  # 至少一个节点避免错误
            if n_edges == 0:
                # 创建自环边
                edge_idx = torch.tensor([[0], [0]], dtype=torch.long, device=device)
                edge_w = torch.zeros((1, self.edge_dim), device=device)
            else:
                edge_idx = edge_index[i][:, valid_edges]
                edge_w = edge_attr[i][valid_edges]
            
            data = Data(
                x=node_feat[i, :n_nodes],
                edge_index=edge_idx,
                edge_attr=edge_w
            )
            data_list.append(data)
        
        batch = Batch.from_data_list(data_list)
        
        # 前向传播
        x = self.node_encoder(batch.x)
        for gnn_layer, layer_norm in zip(self.gnn_layers, self.layer_norms):
            x_new = gnn_layer(x, batch.edge_index, batch.edge_attr)
            x_new = layer_norm(x_new)
            x = F.relu(x_new)
        
        # 全局池化
        graph_embedding = global_mean_pool(x, batch.batch)
        graph_embedding = self.output_mlp(graph_embedding)
        
        return graph_embedding


class GNNActorCritic(nn.Module):
    """
    GNN Actor-Critic 网络
    用于 PPO 算法
    """
    
    def __init__(self, node_dim=5, edge_dim=1, hidden_dim=64, action_dim=2):
        super(GNNActorCritic, self).__init__()
        
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.action_dim = action_dim
        
        # 特征提取器
        self.feature_extractor = GNNFeatureExtractor(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=3
        )
        
        # Actor 头 (输出动作的均值和方差)
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # 输出在 [-1, 1] 范围内
        )
        
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Critic 头 (输出状态价值)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, node_feat, edge_index, edge_attr):
        """
        前向传播
        
        Args:
            node_feat: [B, N, node_dim] 或 [N, node_dim]
            edge_index: [B, 2, E] 或 [2, E]
            edge_attr: [B, E, edge_dim] 或 [E, edge_dim]
            
        Returns:
            action_mean: [B, action_dim]
            action_std: [B, action_dim]
            value: [B, 1]
        """
        # 提取图特征
        features = self.feature_extractor(node_feat, edge_index, edge_attr)
        
        # Actor 输出
        action_mean = self.actor_mean(features)
        action_std = torch.exp(self.actor_log_std).expand_as(action_mean)
        
        # Critic 输出
        value = self.critic(features)
        
        return action_mean, action_std, value

    def get_value(self, node_feat, edge_index, edge_attr):
        """获取状态价值"""
        features = self.feature_extractor(node_feat, edge_index, edge_attr)
        return self.critic(features)

    def evaluate_actions(self, node_feat, edge_index, edge_attr, actions):
        """
        评估动作 (用于 PPO 更新)
        
        Args:
            actions: [B, action_dim]
            
        Returns:
            log_probs: [B]
            entropy: [B]
            value: [B, 1]
        """
        action_mean, action_std, value = self.forward(node_feat, edge_index, edge_attr)
        
        # 计算对数概率
        dist = torch.distributions.Normal(action_mean, action_std)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return log_probs, entropy, value

    def act(self, node_feat, edge_index, edge_attr, deterministic=False):
        """
        采样动作
        
        Args:
            deterministic: 是否使用确定性策略 (使用均值)
            
        Returns:
            action: [action_dim]
            value: [1]
            log_prob: [1]
        """
        action_mean, action_std, value = self.forward(node_feat, edge_index, edge_attr)
        
        if deterministic:
            action = action_mean
        else:
            dist = torch.distributions.Normal(action_mean, action_std)
            action = dist.sample()
        
        dist = torch.distributions.Normal(action_mean, action_std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action.squeeze(0), value.squeeze(0), log_prob.squeeze(0)


class GNNFeatureExtractorSB3(nn.Module):
    """
    与 Stable-Baselines3 兼容的 GNN 特征提取器
    """
    
    def __init__(self, observation_space, features_dim=256, hidden_dim=64):
        super(GNNFeatureExtractorSB3, self).__init__()
        
        self.features_dim = features_dim
        self.hidden_dim = hidden_dim
        
        # GNN 特征提取器
        self.gnn = GNNFeatureExtractor(
            node_dim=5,
            edge_dim=1,
            hidden_dim=hidden_dim,
            output_dim=features_dim,
            num_layers=3
        )

    def forward(self, observations):
        """
        处理观察值
        
        Args:
            observations: dict 包含:
                - node_feat: [B, N, 5]
                - edge_index: [B, 2, E]
                - edge_attr: [B, E, 1]
                
        Returns:
            features: [B, features_dim]
        """
        node_feat = observations['node_feat']
        edge_index = observations['edge_index']
        edge_attr = observations['edge_attr']
        
        return self.gnn(node_feat, edge_index, edge_attr)


def test_gnn_model():
    """测试 GNN 模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建模型
    model = GNNActorCritic(node_dim=5, edge_dim=1, hidden_dim=64, action_dim=2).to(device)
    
    # 创建测试数据
    batch_size = 4
    n_nodes = 50
    n_edges = 150
    
    node_feat = torch.randn(batch_size, n_nodes, 5, device=device)
    edge_index = torch.randint(0, n_nodes, (batch_size, 2, n_edges), device=device)
    edge_attr = torch.randn(batch_size, n_edges, 1, device=device)
    
    # 前向传播
    action_mean, action_std, value = model(node_feat, edge_index, edge_attr)
    
    print(f"Action mean shape: {action_mean.shape}")
    print(f"Action std shape: {action_std.shape}")
    print(f"Value shape: {value.shape}")
    
    # 测试动作采样
    action, val, log_prob = model.act(node_feat, edge_index, edge_attr, deterministic=False)
    print(f"Sampled action shape: {action.shape}")
    print(f"Value: {val}")
    print(f"Log prob: {log_prob}")
    
    print("\nModel test passed!")
    
    return model


if __name__ == "__main__":
    test_gnn_model()
