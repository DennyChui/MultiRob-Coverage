# -*- coding: utf-8 -*-
"""
文件名称：train.py
作者：denny
创建日期：2026-03-12
描述：PPO 训练脚本，使用 GNN Actor-Critic 模型
"""

import os
import argparse
import random
from collections import deque
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from cover_world import CoverWorldEnv
from models import GNNActorCritic


class RolloutBuffer:
    """PPO 经验回放缓冲区"""
    
    def __init__(self):
        self.node_feats = []
        self.edge_indices = []
        self.edge_attrs = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def add(self, node_feat, edge_index, edge_attr, action, log_prob, reward, value, done):
        self.node_feats.append(node_feat)
        self.edge_indices.append(edge_index)
        self.edge_attrs.append(edge_attr)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def clear(self):
        self.node_feats.clear()
        self.edge_indices.clear()
        self.edge_attrs.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()

    def get(self):
        return {
            'node_feats': self.node_feats,
            'edge_indices': self.edge_indices,
            'edge_attrs': self.edge_attrs,
            'actions': self.actions,
            'log_probs': self.log_probs,
            'rewards': self.rewards,
            'values': self.values,
            'dones': self.dones
        }


class PPOTrainer:
    """PPO 训练器"""
    
    def __init__(
        self,
        env,
        model,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_coef=0.2,
        vf_coef=0.5,
        ent_coef=0.01,
        max_grad_norm=0.5,
        device='cpu'
    ):
        self.env = env
        self.model = model.to(device)
        self.device = device
        
        # PPO 超参数
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        
        # 优化器
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # 回放缓冲区
        self.buffer = RolloutBuffer()

    def compute_gae(self, rewards, values, dones, next_value):
        """计算 GAE (Generalized Advantage Estimation)"""
        advantages = []
        gae = 0
        
        values = values + [next_value]
        
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step + 1] * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[step]) * gae
            advantages.insert(0, gae)
        
        returns = [adv + val for adv, val in zip(advantages, values[:-1])]
        
        return torch.tensor(advantages, dtype=torch.float32, device=self.device), \
               torch.tensor(returns, dtype=torch.float32, device=self.device)

    def update(self, batch_size=64, epochs=10):
        """更新策略"""
        buffer_data = self.buffer.get()
        
        # 准备数据
        node_feats = torch.stack([torch.tensor(nf, dtype=torch.float32) for nf in buffer_data['node_feats']]).to(self.device)
        edge_indices = torch.stack([torch.tensor(ei, dtype=torch.int64) for ei in buffer_data['edge_indices']]).to(self.device)
        edge_attrs = torch.stack([torch.tensor(ea, dtype=torch.float32) for ea in buffer_data['edge_attrs']]).to(self.device)
        old_actions = torch.stack([torch.tensor(a, dtype=torch.float32) for a in buffer_data['actions']]).to(self.device)
        old_log_probs = torch.stack([torch.tensor(lp, dtype=torch.float32) for lp in buffer_data['log_probs']]).to(self.device)
        
        # 计算优势和回报
        with torch.no_grad():
            # 获取下一个状态的价值
            obs, _ = self.env.reset()
            next_node_feat = torch.tensor(obs['node_feat'], dtype=torch.float32).unsqueeze(0).to(self.device)
            next_edge_index = torch.tensor(obs['edge_index'], dtype=torch.int64).unsqueeze(0).to(self.device)
            next_edge_attr = torch.tensor(obs['edge_attr'], dtype=torch.float32).unsqueeze(0).to(self.device)
            next_value = self.model.get_value(next_node_feat, next_edge_index, next_edge_attr).squeeze().item()
        
        advantages, returns = self.compute_gae(
            buffer_data['rewards'],
            [v.item() for v in buffer_data['values']],
            buffer_data['dones'],
            next_value
        )
        
        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 多次迭代更新
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        
        n_samples = len(buffer_data['rewards'])
        
        for epoch in range(epochs):
            # 随机采样
            indices = np.random.permutation(n_samples)
            
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]
                
                batch_node_feats = node_feats[batch_indices]
                batch_edge_indices = edge_indices[batch_indices]
                batch_edge_attrs = edge_attrs[batch_indices]
                batch_old_actions = old_actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # 评估动作
                log_probs, entropy, values = self.model.evaluate_actions(
                    batch_node_feats, batch_edge_indices, batch_edge_attrs, batch_old_actions
                )
                
                # 策略损失 (PPO-Clip)
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 价值损失
                value_loss = F.mse_loss(values.squeeze(), batch_returns)
                
                # 总损失
                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy.mean()
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
        
        n_updates = epochs * (n_samples // batch_size + 1)
        
        return {
            'loss': total_loss / n_updates,
            'policy_loss': total_policy_loss / n_updates,
            'value_loss': total_value_loss / n_updates,
            'entropy': total_entropy / n_updates
        }

    def collect_rollouts(self, n_steps=2048):
        """收集经验"""
        self.buffer.clear()
        obs, _ = self.env.reset()
        
        for step in range(n_steps):
            # 准备输入
            node_feat = torch.tensor(obs['node_feat'], dtype=torch.float32).unsqueeze(0).to(self.device)
            edge_index = torch.tensor(obs['edge_index'], dtype=torch.int64).unsqueeze(0).to(self.device)
            edge_attr = torch.tensor(obs['edge_attr'], dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # 采样动作
            with torch.no_grad():
                action, value, log_prob = self.model.act(node_feat, edge_index, edge_attr)
            
            # 执行动作 (模型输出是方向向量，需要转换为角度)
            action_np = action.cpu().numpy()
            angle = np.arctan2(action_np[1], action_np[0])
            angle = angle % (2 * np.pi)  # 归一化到 [0, 2π]
            
            next_obs, reward, terminated, truncated, info = self.env.step(np.array([angle]))
            done = terminated or truncated
            
            # 存储经验
            self.buffer.add(
                obs['node_feat'],
                obs['edge_index'],
                obs['edge_attr'],
                action.cpu().numpy(),
                log_prob.cpu().numpy(),
                reward,
                value.cpu().numpy(),
                float(done)
            )
            
            obs = next_obs
            
            if done:
                obs, _ = self.env.reset()
        
        return self.buffer


def train(
    total_timesteps=1000000,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    lr=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_coef=0.2,
    vf_coef=0.5,
    ent_coef=0.01,
    max_grad_norm=0.5,
    width=10,
    height=10,
    max_episodes_length=200,
    seed=42,
    device='cpu',
    save_dir='./checkpoints',
    log_dir='./logs'
):
    """
    训练 PPO 模型
    
    Args:
        total_timesteps: 总训练步数
        n_steps: 每次更新的步数
        batch_size: 批次大小
        n_epochs: 每次数据迭代次数
        lr: 学习率
        gamma: 折扣因子
        gae_lambda: GAE lambda
        clip_coef: PPO clip 系数
        vf_coef: 价值函数系数
        ent_coef: 熵系数
        max_grad_norm: 梯度裁剪
        width: 地图宽度
        height: 地图高度
        max_episodes_length: 最大回合长度
        seed: 随机种子
        device: 训练设备
        save_dir: 模型保存目录
        log_dir: 日志目录
    """
    # 设置随机种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # 创建目录
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建环境
    env = CoverWorldEnv({
        'width': width,
        'height': height,
        'max_episodes_length': max_episodes_length,
        'seed': seed,
        'render_mode': None
    })
    
    # 创建模型
    model = GNNActorCritic(
        node_dim=5,
        edge_dim=1,
        hidden_dim=64,
        action_dim=2  # 输出2D方向向量
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # 创建训练器
    trainer = PPOTrainer(
        env=env,
        model=model,
        lr=lr,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_coef=clip_coef,
        vf_coef=vf_coef,
        ent_coef=ent_coef,
        max_grad_norm=max_grad_norm,
        device=device
    )
    
    # 创建 TensorBoard writer
    writer = SummaryWriter(log_dir=log_dir)
    
    # 训练循环
    timestep = 0
    episode = 0
    best_reward = -float('inf')
    
    print(f"Starting training on {device}...")
    print(f"Total timesteps: {total_timesteps}")
    
    while timestep < total_timesteps:
        # 收集经验
        trainer.collect_rollouts(n_steps=n_steps)
        timestep += n_steps
        
        # 更新策略
        update_info = trainer.update(batch_size=batch_size, epochs=n_epochs)
        
        # 记录日志
        writer.add_scalar('train/loss', update_info['loss'], timestep)
        writer.add_scalar('train/policy_loss', update_info['policy_loss'], timestep)
        writer.add_scalar('train/value_loss', update_info['value_loss'], timestep)
        writer.add_scalar('train/entropy', update_info['entropy'], timestep)
        
        # 定期评估
        if timestep % (n_steps * 5) == 0:
            eval_reward = evaluate(env, model, device, n_episodes=5)
            writer.add_scalar('eval/mean_reward', eval_reward, timestep)
            
            print(f"Timestep: {timestep}, Eval Reward: {eval_reward:.2f}, "
                  f"Loss: {update_info['loss']:.4f}, "
                  f"Entropy: {update_info['entropy']:.4f}")
            
            # 保存最佳模型
            if eval_reward > best_reward:
                best_reward = eval_reward
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                    'timestep': timestep,
                    'best_reward': best_reward
                }, os.path.join(save_dir, 'best_model.pt'))
        
        # 定期保存检查点
        if timestep % (n_steps * 10) == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'timestep': timestep,
            }, os.path.join(save_dir, f'checkpoint_{timestep}.pt'))
    
    # 保存最终模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
        'timestep': timestep,
    }, os.path.join(save_dir, 'final_model.pt'))
    
    writer.close()
    print(f"Training completed! Final model saved to {save_dir}/final_model.pt")
    
    return model


def evaluate(env, model, device, n_episodes=10):
    """评估模型"""
    model.eval()
    total_rewards = []
    
    for _ in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            node_feat = torch.tensor(obs['node_feat'], dtype=torch.float32).unsqueeze(0).to(device)
            edge_index = torch.tensor(obs['edge_index'], dtype=torch.int64).unsqueeze(0).to(device)
            edge_attr = torch.tensor(obs['edge_attr'], dtype=torch.float32).unsqueeze(0).to(device)
            
            with torch.no_grad():
                action, _, _ = model.act(node_feat, edge_index, edge_attr, deterministic=True)
            
            action_np = action.cpu().numpy()
            angle = np.arctan2(action_np[1], action_np[0]) % (2 * np.pi)
            
            obs, reward, terminated, truncated, _ = env.step(np.array([angle]))
            episode_reward += reward
            done = terminated or truncated
        
        total_rewards.append(episode_reward)
    
    model.train()
    return np.mean(total_rewards)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train PPO with GNN on CoverWorld')
    parser.add_argument('--timesteps', type=int, default=1000000, help='Total training timesteps')
    parser.add_argument('--width', type=int, default=10, help='Map width')
    parser.add_argument('--height', type=int, default=10, help='Map height')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='auto', help='Device (cpu/cuda/auto)')
    parser.add_argument('--save-dir', type=str, default='./checkpoints', help='Checkpoint directory')
    parser.add_argument('--log-dir', type=str, default='./logs', help='Log directory')
    
    args = parser.parse_args()
    
    # 自动选择设备
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Training configuration:")
    print(f"  Total timesteps: {args.timesteps}")
    print(f"  Map size: {args.width}x{args.height}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Seed: {args.seed}")
    print(f"  Device: {device}")
    
    train(
        total_timesteps=args.timesteps,
        width=args.width,
        height=args.height,
        seed=args.seed,
        device=device,
        save_dir=args.save_dir,
        log_dir=args.log_dir
    )
