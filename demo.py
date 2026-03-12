# -*- coding: utf-8 -*-
"""
文件名称：demo.py
作者：denny
创建日期：2026-03-12
描述：快速演示脚本，展示如何使用 CoverWorld 环境
"""

import numpy as np
import torch

from cover_world import CoverWorldEnv
from models import GNNActorCritic


def random_agent_demo():
    """随机智能体演示"""
    print("=" * 60)
    print("Random Agent Demo")
    print("=" * 60)
    
    # 创建环境
    env = CoverWorldEnv({
        'width': 10,
        'height': 10,
        'max_episodes_length': 100,
        'seed': 42,
        'render_mode': None
    })
    
    obs, info = env.reset()
    total_reward = 0
    steps = 0
    
    print(f"\nInitial coverage: {info['coverage_rate']:.2%}")
    
    for step in range(100):
        # 随机动作
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        steps += 1
        
        if step % 10 == 0:
            print(f"Step {step}: coverage={info['coverage_rate']:.2%}, reward={reward:.2f}")
        
        if terminated or truncated:
            break
    
    print(f"\nEpisode finished!")
    print(f"  Total steps: {steps}")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Final coverage: {info['coverage_rate']:.2%}")


def gnn_model_demo():
    """GNN 模型演示"""
    print("\n" + "=" * 60)
    print("GNN Model Demo")
    print("=" * 60)
    
    # 创建环境
    env = CoverWorldEnv({
        'width': 10,
        'height': 10,
        'max_episodes_length': 100,
        'seed': 42,
        'render_mode': None
    })
    
    # 创建模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GNNActorCritic(node_dim=5, edge_dim=1, hidden_dim=64, action_dim=2).to(device)
    
    print(f"\nDevice: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    obs, info = env.reset()
    total_reward = 0
    steps = 0
    
    print(f"\nInitial coverage: {info['coverage_rate']:.2%}")
    
    for step in range(100):
        # 准备输入
        node_feat = torch.tensor(obs['node_feat'], dtype=torch.float32).unsqueeze(0).to(device)
        edge_index = torch.tensor(obs['edge_index'], dtype=torch.int64).unsqueeze(0).to(device)
        edge_attr = torch.tensor(obs['edge_attr'], dtype=torch.float32).unsqueeze(0).to(device)
        
        # 模型推理
        with torch.no_grad():
            action, value, log_prob = model.act(node_feat, edge_index, edge_attr)
        
        # 转换为角度
        action_np = action.cpu().numpy()
        angle = np.arctan2(action_np[1], action_np[0]) % (2 * np.pi)
        
        # 执行动作
        obs, reward, terminated, truncated, info = env.step(np.array([angle]))
        
        total_reward += reward
        steps += 1
        
        if step % 10 == 0:
            print(f"Step {step}: coverage={info['coverage_rate']:.2%}, "
                  f"reward={reward:.2f}, value={value.item():.2f}")
        
        if terminated or truncated:
            break
    
    print(f"\nEpisode finished!")
    print(f"  Total steps: {steps}")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Final coverage: {info['coverage_rate']:.2%}")


def batch_inference_demo():
    """批量推理演示"""
    print("\n" + "=" * 60)
    print("Batch Inference Demo")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GNNActorCritic(node_dim=5, edge_dim=1, hidden_dim=64, action_dim=2).to(device)
    
    # 创建批量数据
    batch_size = 4
    n_nodes = 50
    n_edges = 100
    
    node_feat = torch.randn(batch_size, n_nodes, 5, device=device)
    edge_index = torch.randint(0, n_nodes, (batch_size, 2, n_edges), device=device)
    edge_attr = torch.randn(batch_size, n_edges, 1, device=device)
    
    print(f"\nBatch size: {batch_size}")
    print(f"Nodes per graph: {n_nodes}")
    print(f"Edges per graph: {n_edges}")
    
    # 前向传播
    with torch.no_grad():
        action_mean, action_std, value = model(node_feat, edge_index, edge_attr)
    
    print(f"\nOutput shapes:")
    print(f"  Action mean: {action_mean.shape}")
    print(f"  Action std: {action_std.shape}")
    print(f"  Value: {value.shape}")
    
    print("\n✓ Batch inference successful!")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("CoverWorld Demo")
    print("=" * 60)
    
    # 运行演示
    random_agent_demo()
    gnn_model_demo()
    batch_inference_demo()
    
    print("\n" + "=" * 60)
    print("All demos completed!")
    print("=" * 60)
