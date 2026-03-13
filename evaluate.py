# -*- coding: utf-8 -*-
"""
文件名称：evaluate.py
作者：denny
创建日期：2026-03-12
描述：对比评估训练好的模型和随机模型的性能
"""

import os
import argparse
import random
import numpy as np
import torch
from collections import defaultdict
from datetime import datetime

from cover_world import CoverWorldEnv
from models import GNNActorCritic


class RandomAgent:
    """随机动作代理"""
    
    def __init__(self, action_dim=2):
        self.action_dim = action_dim
    
    def act(self, node_feat, edge_index, edge_attr, deterministic=False):
        """返回随机动作"""
        # 随机生成方向向量
        angle = random.uniform(0, 2 * np.pi)
        action = np.array([np.cos(angle), np.sin(angle)])
        return action, 0.0, 0.0


def evaluate_agent(env, agent, device, n_episodes=20, deterministic=True, agent_name="Agent"):
    """
    评估代理性能
    
    Args:
        env: 环境实例
        agent: 代理模型（训练好的模型或随机代理）
        device: 计算设备
        n_episodes: 评估回合数
        deterministic: 是否使用确定性策略
        agent_name: 代理名称（用于日志）
    
    Returns:
        dict: 包含各种评估指标的字典
    """
    metrics = {
        'episode_rewards': [],
        'episode_lengths': [],
        'final_coverage_rates': [],
        'coverages_50_percent': [],  # 达到50%覆盖率的回合
        'coverages_70_percent': [],  # 达到70%覆盖率的回合
        'coverages_90_percent': [],  # 达到90%覆盖率的回合
    }
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        steps = 0
        done = False
        max_coverage = 0.0
        
        while not done:
            # 根据代理类型执行动作
            if isinstance(agent, GNNActorCritic):
                # 训练好的模型
                node_feat = torch.tensor(obs['node_feat'], dtype=torch.float32).unsqueeze(0).to(device)
                edge_index = torch.tensor(obs['edge_index'], dtype=torch.int64).unsqueeze(0).to(device)
                edge_attr = torch.tensor(obs['edge_attr'], dtype=torch.float32).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    action, _, _ = agent.act(node_feat, edge_index, edge_attr, deterministic=deterministic)
                
                action_np = action.cpu().numpy()
                angle = np.arctan2(action_np[1], action_np[0]) % (2 * np.pi)
            else:
                # 随机代理
                action, _, _ = agent.act(None, None, None)
                angle = np.arctan2(action[1], action[0]) % (2 * np.pi)
            
            # 执行动作
            obs, reward, terminated, truncated, info = env.step(np.array([angle]))
            episode_reward += reward
            steps += 1
            done = terminated or truncated
            
            # 记录最大覆盖率
            coverage = info.get('coverage_rate', 0.0)
            max_coverage = max(max_coverage, coverage)
        
        # 记录本回合的指标
        metrics['episode_rewards'].append(episode_reward)
        metrics['episode_lengths'].append(steps)
        metrics['final_coverage_rates'].append(info.get('coverage_rate', 0.0))
        
        # 记录是否达到覆盖率阈值
        final_coverage = info.get('coverage_rate', 0.0)
        metrics['coverages_50_percent'].append(1 if final_coverage >= 0.5 else 0)
        metrics['coverages_70_percent'].append(1 if final_coverage >= 0.7 else 0)
        metrics['coverages_90_percent'].append(1 if final_coverage >= 0.9 else 0)
        
        if (episode + 1) % 5 == 0:
            print(f"  {agent_name} - Episode {episode + 1}/{n_episodes} completed")
    
    # 计算统计指标
    results = {
        'mean_reward': np.mean(metrics['episode_rewards']),
        'std_reward': np.std(metrics['episode_rewards']),
        'mean_length': np.mean(metrics['episode_lengths']),
        'std_length': np.std(metrics['episode_lengths']),
        'mean_coverage': np.mean(metrics['final_coverage_rates']),
        'std_coverage': np.std(metrics['final_coverage_rates']),
        'coverage_50_rate': np.mean(metrics['coverages_50_percent']) * 100,
        'coverage_70_rate': np.mean(metrics['coverages_70_percent']) * 100,
        'coverage_90_rate': np.mean(metrics['coverages_90_percent']) * 100,
        'max_reward': np.max(metrics['episode_rewards']),
        'min_reward': np.min(metrics['episode_rewards']),
        'max_coverage': np.max(metrics['final_coverage_rates']),
        'min_coverage': np.min(metrics['final_coverage_rates']),
    }
    
    return results, metrics


def print_comparison(trained_results, random_results):
    """打印对比结果"""
    print("\n" + "=" * 80)
    print("模型性能对比结果")
    print("=" * 80)
    
    # 表头
    print(f"{'指标':<30} {'训练模型':>20} {'随机模型':>20} {'提升':>10}")
    print("-" * 80)
    
    # 各项指标
    metrics_to_compare = [
        ('平均回合奖励', 'mean_reward', '.2f'),
        ('回合奖励标准差', 'std_reward', '.2f'),
        ('最大回合奖励', 'max_reward', '.2f'),
        ('最小回合奖励', 'min_reward', '.2f'),
        ('平均回合长度', 'mean_length', '.1f'),
        ('平均覆盖率', 'mean_coverage', '.2%'),
        ('覆盖率标准差', 'std_coverage', '.2%'),
        ('最大覆盖率', 'max_coverage', '.2%'),
        ('最小覆盖率', 'min_coverage', '.2%'),
        ('达到50%覆盖率的回合比例', 'coverage_50_rate', '.1f%'),
        ('达到70%覆盖率的回合比例', 'coverage_70_rate', '.1f%'),
        ('达到90%覆盖率的回合比例', 'coverage_90_rate', '.1f%'),
    ]
    
    for name, key, fmt in metrics_to_compare:
        trained_val = trained_results[key]
        random_val = random_results[key]
        
        # 计算提升百分比（对于可以计算提升的指标）
        if '覆盖率' in name or '奖励' in name:
            if random_val != 0:
                improvement = ((trained_val - random_val) / abs(random_val)) * 100
                improvement_str = f"{improvement:+.1f}%"
            else:
                improvement_str = "N/A"
        else:
            improvement_str = "-"
        
        if '%' in fmt:
            trained_str = f"{trained_val:.1f}%"
            random_str = f"{random_val:.1f}%"
        else:
            trained_str = f"{trained_val:{fmt.replace('%', '')}}"
            random_str = f"{random_val:{fmt.replace('%', '')}}"
        
        print(f"{name:<30} {trained_str:>20} {random_str:>20} {improvement_str:>10}")
    
    print("=" * 80)


def save_results(trained_results, random_results, output_file='evaluation_results.txt'):
    """保存评估结果到文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("模型性能对比评估报告\n")
        f.write(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        # 训练模型结果
        f.write("【训练模型结果】\n")
        f.write("-" * 40 + "\n")
        for key, value in trained_results.items():
            if 'coverage' in key or 'rate' in key:
                f.write(f"{key}: {value:.2%}\n")
            else:
                f.write(f"{key}: {value:.4f}\n")
        
        f.write("\n")
        
        # 随机模型结果
        f.write("【随机模型结果】\n")
        f.write("-" * 40 + "\n")
        for key, value in random_results.items():
            if 'coverage' in key or 'rate' in key:
                f.write(f"{key}: {value:.2%}\n")
            else:
                f.write(f"{key}: {value:.4f}\n")
        
        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write("评估完成\n")
    
    print(f"\n评估结果已保存到: {output_file}")


def main(
    model_path='./checkpoints/best_model.pt',
    n_episodes=20,
    width=10,
    height=10,
    max_episodes_length=200,
    seed=42,
    device='cpu',
    save_results_file=True,
    deterministic=True
):
    """
    主评估函数
    
    Args:
        model_path: 训练好的模型路径
        n_episodes: 评估回合数
        width: 地图宽度
        height: 地图高度
        max_episodes_length: 最大回合长度
        seed: 随机种子
        device: 计算设备
        save_results_file: 是否保存结果到文件
        deterministic: 是否使用确定性策略
    """
    # 设置随机种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    print("=" * 80)
    print("开始模型性能对比评估")
    print("=" * 80)
    print(f"评估配置:")
    print(f"  模型路径: {model_path}")
    print(f"  评估回合数: {n_episodes}")
    print(f"  地图大小: {width}x{height}")
    print(f"  最大回合长度: {max_episodes_length}")
    print(f"  随机种子: {seed}")
    print(f"  设备: {device}")
    print(f"  确定性策略: {deterministic}")
    print("=" * 80)
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        print(f"可用模型文件:")
        checkpoint_dir = './checkpoints'
        if os.path.exists(checkpoint_dir):
            for f in os.listdir(checkpoint_dir):
                if f.endswith('.pt'):
                    print(f"  - {os.path.join(checkpoint_dir, f)}")
        return
    
    # 创建环境
    env = CoverWorldEnv({
        'width': width,
        'height': height,
        'max_episodes_length': max_episodes_length,
        'seed': seed,
        'render_mode': None
    })
    
    # 加载训练好的模型
    print("\n正在加载训练好的模型...")
    trained_model = GNNActorCritic(
        node_dim=5,
        edge_dim=1,
        hidden_dim=64,
        action_dim=2
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    trained_model.load_state_dict(checkpoint['model_state_dict'])
    trained_model.eval()
    print(f"成功加载模型: {model_path}")
    
    # 创建随机代理
    random_agent = RandomAgent(action_dim=2)
    
    # 评估训练好的模型
    print(f"\n正在评估训练好的模型 ({n_episodes} 回合)...")
    trained_results, trained_metrics = evaluate_agent(
        env, trained_model, device, n_episodes, deterministic, "训练模型"
    )
    
    # 评估随机模型
    print(f"\n正在评估随机模型 ({n_episodes} 回合)...")
    random_results, random_metrics = evaluate_agent(
        env, random_agent, device, n_episodes, False, "随机模型"
    )
    
    # 打印对比结果
    print_comparison(trained_results, random_results)
    
    # 保存结果
    if save_results_file:
        output_file = f'evaluation_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
        save_results(trained_results, random_results, output_file)
    
    print("\n评估完成!")
    
    return trained_results, random_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='对比评估训练模型和随机模型')
    parser.add_argument('--model-path', type=str, default='./checkpoints/best_model.pt',
                        help='训练好的模型路径')
    parser.add_argument('--n-episodes', type=int, default=20,
                        help='评估回合数')
    parser.add_argument('--width', type=int, default=10,
                        help='地图宽度')
    parser.add_argument('--height', type=int, default=10,
                        help='地图高度')
    parser.add_argument('--max-episodes-length', type=int, default=200,
                        help='最大回合长度')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--device', type=str, default='auto',
                        help='计算设备 (cpu/cuda/auto)')
    parser.add_argument('--no-save', action='store_true',
                        help='不保存结果到文件')
    parser.add_argument('--stochastic', action='store_true',
                        help='使用随机策略（默认确定性）')
    
    args = parser.parse_args()
    
    # 自动选择设备
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    main(
        model_path=args.model_path,
        n_episodes=args.n_episodes,
        width=args.width,
        height=args.height,
        max_episodes_length=args.max_episodes_length,
        seed=args.seed,
        device=device,
        save_results_file=not args.no_save,
        deterministic=not args.stochastic
    )
