# -*- coding: utf-8 -*-
"""
文件名称：visualize.py
作者：denny
创建日期：2026-03-12
描述：模型评估与决策路径可视化
"""

import os
import argparse
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from datetime import datetime
from collections import defaultdict

from cover_world import CoverWorldEnv
from models import GNNActorCritic

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False


def set_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def run_episode(env, model, device, deterministic=True, record_frames=False):
    """
    运行一个回合，记录轨迹和决策信息
    
    Returns:
        dict: 包含轨迹、决策、覆盖率等信息
    """
    obs, info = env.reset()
    
    trajectory = []  # 机器人位置轨迹 [(x, y), ...]
    actions = []  # 动作序列 [(cos, sin), ...]
    coverages = []  # 覆盖率序列
    rewards = []  # 奖励序列
    frames = []  # 渲染帧（用于动画）
    node_nums = []  # 节点数量
    values = []  # 价值函数估计
    
    episode_reward = 0
    steps = 0
    done = False
    
    while not done:
        # 记录当前状态
        grid_graph = env.grid_graph
        trajectory.append((grid_graph.rob_pos[0], grid_graph.rob_pos[1]))
        coverages.append(info['coverage_rate'])
        
        # 准备模型输入
        node_feat = torch.tensor(obs['node_feat'], dtype=torch.float32).unsqueeze(0).to(device)
        edge_index = torch.tensor(obs['edge_index'], dtype=torch.int64).unsqueeze(0).to(device)
        edge_attr = torch.tensor(obs['edge_attr'], dtype=torch.float32).unsqueeze(0).to(device)
        
        # 模型推理
        if isinstance(model, GNNActorCritic):
            # 训练好的模型
            with torch.no_grad():
                action_mean, action_std, value = model(node_feat, edge_index, edge_attr)
                action, _, _ = model.act(node_feat, edge_index, edge_attr, deterministic=deterministic)
            
            # 记录决策信息
            action_np = action.cpu().numpy()
            values.append(value.item())
        else:
            # 随机代理
            action, _, _ = model.act(None, None, None)
            action_np = np.array(action)
            values.append(0.0)
        
        actions.append((action_np[0], action_np[1]))
        node_nums.append(obs['node_num'][0])
        
        # 转换为角度
        angle = np.arctan2(action_np[1], action_np[0]) % (2 * np.pi)
        
        # 执行动作
        obs, reward, terminated, truncated, info = env.step(np.array([angle]))
        
        rewards.append(reward)
        episode_reward += reward
        steps += 1
        done = terminated or truncated
        
        # 记录帧
        if record_frames:
            frames.append(get_frame_data(grid_graph))
    
    # 添加最终位置
    trajectory.append((grid_graph.rob_pos[0], grid_graph.rob_pos[1]))
    coverages.append(info['coverage_rate'])
    
    return {
        'trajectory': np.array(trajectory),
        'actions': np.array(actions),
        'coverages': np.array(coverages),
        'rewards': np.array(rewards),
        'values': np.array(values),
        'node_nums': np.array(node_nums),
        'frames': frames,
        'episode_reward': episode_reward,
        'steps': steps,
        'final_coverage': info['coverage_rate'],
        'grid_graph': grid_graph
    }


def get_frame_data(grid_graph):
    """获取当前帧的可视化数据"""
    # 获取地图边界
    width = grid_graph.width
    height = grid_graph.height
    
    # 获取节点位置
    node_positions = {}
    node_covered = {}
    for node in grid_graph.graph.nodes(data=True):
        node_id = node[0]
        attr = node[1]['attr']
        node_positions[node_id] = attr['position']
        node_covered[node_id] = attr['covered']
    
    # 机器人位置
    rob_pos = grid_graph.rob_pos.copy()
    
    return {
        'node_positions': node_positions,
        'node_covered': node_covered,
        'rob_pos': rob_pos,
        'covered_num': grid_graph.covered_num,
        'max_num_points': grid_graph.max_num_points
    }


def visualize_trajectory(result, save_path=None, show=True):
    """
    可视化决策路径（静态图）
    """
    trajectory = result['trajectory']
    grid_graph = result['grid_graph']
    coverages = result['coverages']
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # 左图：轨迹地图
    ax1 = axes[0]
    
    # 绘制地图网格点
    covered_nodes = []
    uncovered_nodes = []
    
    for node in grid_graph.graph.nodes(data=True):
        attr = node[1]['attr']
        if attr['id'] == grid_graph.rob_id:
            continue
        pos = attr['position']
        if attr['covered'] == 1:
            covered_nodes.append(pos)
        elif attr['covered'] == 0:
            uncovered_nodes.append(pos)
    
    if uncovered_nodes:
        uncovered_nodes = np.array(uncovered_nodes)
        ax1.scatter(uncovered_nodes[:, 0], uncovered_nodes[:, 1], 
                   c='lightblue', s=20, alpha=0.5, label='未覆盖节点')
    
    if covered_nodes:
        covered_nodes = np.array(covered_nodes)
        ax1.scatter(covered_nodes[:, 0], covered_nodes[:, 1], 
                   c='lightgreen', s=20, alpha=0.5, label='已覆盖节点')
    
    # 绘制轨迹
    ax1.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2, alpha=0.6, label='移动轨迹')
    ax1.plot(trajectory[0, 0], trajectory[0, 1], 'go', markersize=15, label='起点')
    ax1.plot(trajectory[-1, 0], trajectory[-1, 1], 'r*', markersize=20, label='终点')
    
    # 绘制覆盖范围圆
    from cover_world.grid_graph import COVER_R
    for pos in trajectory[::10]:  # 每10步绘制一个覆盖范围
        circle = plt.Circle(pos, COVER_R, fill=False, linestyle='--', 
                           color='orange', alpha=0.3, linewidth=0.5)
        ax1.add_patch(circle)
    
    ax1.set_xlim(-grid_graph.width/2 - 1, grid_graph.width/2 + 1)
    ax1.set_ylim(-grid_graph.height/2 - 1, grid_graph.height/2 + 1)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('X 坐标')
    ax1.set_ylabel('Y 坐标')
    ax1.set_title(f'决策路径可视化\n总步数: {result["steps"]}, 最终覆盖率: {result["final_coverage"]:.2%}')
    ax1.legend(loc='upper right')
    
    # 右图：覆盖率变化曲线
    ax2 = axes[1]
    steps = range(len(coverages))
    ax2.plot(steps, coverages * 100, 'b-', linewidth=2)
    ax2.fill_between(steps, 0, coverages * 100, alpha=0.3)
    ax2.axhline(y=100, color='r', linestyle='--', label='目标覆盖率 (100%)')
    ax2.axhline(y=50, color='orange', linestyle='--', alpha=0.5, label='50% 覆盖率')
    ax2.set_xlabel('步数')
    ax2.set_ylabel('覆盖率 (%)')
    ax2.set_title('覆盖率变化曲线')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_ylim(0, 110)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"轨迹图已保存到: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def visualize_action_distribution(result, save_path=None, show=True):
    """
    可视化动作分布（极坐标图）
    """
    actions = result['actions']
    
    # 计算角度
    angles = np.arctan2(actions[:, 1], actions[:, 0])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左图：动作方向的时间序列
    ax1 = axes[0]
    steps = range(len(angles))
    ax1.plot(steps, angles, 'b-', linewidth=1, alpha=0.7)
    ax1.scatter(steps, angles, c=steps, cmap='viridis', s=20, alpha=0.6)
    ax1.set_xlabel('步数')
    ax1.set_ylabel('动作角度 (弧度)')
    ax1.set_title('动作方向变化')
    ax1.grid(True, alpha=0.3)
    
    # 右图：动作分布玫瑰图
    ax2 = axes[1]
    ax2 = plt.subplot(122, projection='polar')
    
    # 创建直方图
    n_bins = 24
    hist, bin_edges = np.histogram(angles, bins=n_bins, range=(-np.pi, np.pi))
    theta = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    bars = ax2.bar(theta, hist, width=2*np.pi/n_bins, bottom=0.0, alpha=0.7)
    
    # 根据频次设置颜色
    for bar, count in zip(bars, hist):
        bar.set_facecolor(plt.cm.viridis(count / max(hist) if max(hist) > 0 else 0))
    
    ax2.set_theta_zero_location('E')
    ax2.set_theta_direction(-1)
    ax2.set_title('动作方向分布')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"动作分布图已保存到: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def visualize_value_function(result, save_path=None, show=True):
    """
    可视化价值函数估计
    """
    values = result['values']
    rewards = result['rewards']
    coverages = result['coverages'][:-1]
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # 上图：价值函数变化
    ax1 = axes[0]
    steps = range(len(values))
    ax1.plot(steps, values, 'g-', linewidth=2, label='价值估计')
    ax1.set_xlabel('步数')
    ax1.set_ylabel('价值 V(s)')
    ax1.set_title('价值函数估计变化')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 下图：奖励和价值对比
    ax2 = axes[1]
    ax2.plot(steps, rewards, 'r-', linewidth=1, alpha=0.7, label='即时奖励')
    ax2.plot(steps, values, 'g-', linewidth=2, alpha=0.8, label='价值估计')
    ax2.set_xlabel('步数')
    ax2.set_ylabel('数值')
    ax2.set_title('奖励 vs 价值')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"价值函数图已保存到: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def create_animation(result, save_path=None, show=True):
    """
    创建决策过程动画
    """
    trajectory = result['trajectory']
    grid_graph = result['grid_graph']
    coverages = result['coverages']
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 获取地图范围
    width = grid_graph.width
    height = grid_graph.height
    
    # 获取所有节点位置
    node_positions = []
    for node in grid_graph.graph.nodes(data=True):
        attr = node[1]['attr']
        if attr['id'] != grid_graph.rob_id:
            node_positions.append(attr['position'])
    
    node_positions = np.array(node_positions)
    
    # 绘制静态背景
    ax.scatter(node_positions[:, 0], node_positions[:, 1], 
              c='lightgray', s=10, alpha=0.5, label='地图节点')
    
    # 初始化动态元素
    path_line, = ax.plot([], [], 'b-', linewidth=2, alpha=0.6, label='轨迹')
    current_pos, = ax.plot([], [], 'ro', markersize=15, label='机器人')
    start_pos, = ax.plot([], [], 'go', markersize=15, label='起点')
    
    # 添加覆盖率文本
    coverage_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                           fontsize=12, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 添加覆盖范围圆
    from cover_world.grid_graph import COVER_R
    cover_circle = plt.Circle((0, 0), COVER_R, fill=False, linestyle='--', 
                              color='orange', alpha=0.6, linewidth=2)
    ax.add_patch(cover_circle)
    
    ax.set_xlim(-width/2 - 1, width/2 + 1)
    ax.set_ylim(-height/2 - 1, height/2 + 1)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    def init():
        path_line.set_data([], [])
        current_pos.set_data([], [])
        start_pos.set_data([trajectory[0, 0]], [trajectory[0, 1]])
        coverage_text.set_text(f'覆盖率: 0.0%')
        cover_circle.center = (trajectory[0, 0], trajectory[0, 1])
        return path_line, current_pos, start_pos, coverage_text, cover_circle
    
    def update(frame):
        path_line.set_data(trajectory[:frame+1, 0], trajectory[:frame+1, 1])
        current_pos.set_data([trajectory[frame, 0]], [trajectory[frame, 1]])
        coverage_text.set_text(f'覆盖率: {coverages[frame]:.1%} | 步数: {frame}')
        cover_circle.center = (trajectory[frame, 0], trajectory[frame, 1])
        return path_line, current_pos, start_pos, coverage_text, cover_circle
    
    anim = FuncAnimation(fig, update, frames=len(trajectory)-1,
                        init_func=init, blit=True, interval=50)
    
    if save_path:
        anim.save(save_path, writer='pillow', fps=20, dpi=100)
        print(f"动画已保存到: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return anim


def visualize_comparison(trained_result, random_result, save_path=None, show=True):
    """
    对比训练模型和随机模型的决策路径
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    for idx, (result, title) in enumerate([(trained_result, '训练模型'), (random_result, '随机模型')]):
        ax = axes[idx]
        trajectory = result['trajectory']
        grid_graph = result['grid_graph']
        
        # 绘制地图节点
        covered_nodes = []
        uncovered_nodes = []
        
        for node in grid_graph.graph.nodes(data=True):
            attr = node[1]['attr']
            if attr['id'] == grid_graph.rob_id:
                continue
            pos = attr['position']
            if attr['covered'] == 1:
                covered_nodes.append(pos)
            elif attr['covered'] == 0:
                uncovered_nodes.append(pos)
        
        if uncovered_nodes:
            uncovered_nodes = np.array(uncovered_nodes)
            ax.scatter(uncovered_nodes[:, 0], uncovered_nodes[:, 1], 
                      c='lightblue', s=15, alpha=0.4)
        
        if covered_nodes:
            covered_nodes = np.array(covered_nodes)
            ax.scatter(covered_nodes[:, 0], covered_nodes[:, 1], 
                      c='lightgreen', s=15, alpha=0.4)
        
        # 绘制轨迹
        ax.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2, alpha=0.6)
        ax.plot(trajectory[0, 0], trajectory[0, 1], 'go', markersize=12, label='起点')
        ax.plot(trajectory[-1, 0], trajectory[-1, 1], 'r*', markersize=15, label='终点')
        
        ax.set_xlim(-grid_graph.width/2 - 1, grid_graph.width/2 + 1)
        ax.set_ylim(-grid_graph.height/2 - 1, grid_graph.height/2 + 1)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X 坐标')
        ax.set_ylabel('Y 坐标')
        ax.set_title(f'{title}\n步数: {result["steps"]}, 覆盖率: {result["final_coverage"]:.1%}, '
                    f'奖励: {result["episode_reward"]:.1f}')
        ax.legend(loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"对比图已保存到: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def main(
    model_path='./checkpoints/best_model.pt',
    n_episodes=3,
    width=10,
    height=10,
    max_episodes_length=200,
    seed=42,
    device='cpu',
    output_dir='./visualizations',
    no_show=False,
    create_anim=True
):
    """
    主可视化函数
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # 设置种子
    set_seed(seed)
    
    print("=" * 80)
    print("模型决策路径可视化")
    print("=" * 80)
    print(f"模型路径: {model_path}")
    print(f"评估回合: {n_episodes}")
    print(f"地图大小: {width}x{height}")
    print(f"输出目录: {run_dir}")
    print("=" * 80)
    
    # 检查模型文件
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        return
    
    # 创建环境
    env = CoverWorldEnv({
        'width': width,
        'height': height,
        'max_episodes_length': max_episodes_length,
        'seed': seed,
        'render_mode': None
    })
    
    # 加载模型
    print("\n加载模型...")
    model = GNNActorCritic(
        node_dim=5,
        edge_dim=1,
        hidden_dim=64,
        action_dim=2
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"✓ 模型加载成功")
    
    # 运行评估回合
    results = []
    for ep in range(n_episodes):
        print(f"\n运行回合 {ep+1}/{n_episodes}...")
        
        # 使用不同种子
        ep_seed = seed + ep
        set_seed(ep_seed)
        env.seed = ep_seed
        
        result = run_episode(env, model, device, deterministic=True, 
                           record_frames=create_anim and ep == 0)
        results.append(result)
        
        print(f"  步数: {result['steps']}")
        print(f"  最终覆盖率: {result['final_coverage']:.2%}")
        print(f"  总奖励: {result['episode_reward']:.2f}")
        
        # 生成可视化
        ep_dir = os.path.join(run_dir, f"episode_{ep+1}")
        os.makedirs(ep_dir, exist_ok=True)
        
        show = not no_show and ep == 0  # 只显示第一个回合
        
        # 轨迹图
        visualize_trajectory(
            result, 
            save_path=os.path.join(ep_dir, "trajectory.png"),
            show=show
        )
        
        # 动作分布
        visualize_action_distribution(
            result,
            save_path=os.path.join(ep_dir, "action_distribution.png"),
            show=show
        )
        
        # 价值函数
        visualize_value_function(
            result,
            save_path=os.path.join(ep_dir, "value_function.png"),
            show=show
        )
        
        # 动画（只生成第一个回合）
        if create_anim and ep == 0:
            print("  生成动画...")
            create_animation(
                result,
                save_path=os.path.join(ep_dir, "animation.gif"),
                show=False
            )
    
    # 运行随机模型进行对比
    print("\n运行随机模型对比...")
    set_seed(seed)
    env.seed = seed
    
    # 使用与evaluate.py相同的随机代理
    class RandomAgent:
        def act(self, node_feat, edge_index, edge_attr, deterministic=False):
            angle = random.uniform(0, 2 * np.pi)
            action = np.array([np.cos(angle), np.sin(angle)])
            return action, 0.0, 0.0
    
    random_agent = RandomAgent()
    random_result = run_episode(env, random_agent, device, deterministic=False)
    
    print(f"  随机模型 - 步数: {random_result['steps']}, 覆盖率: {random_result['final_coverage']:.2%}")
    
    # 对比可视化
    visualize_comparison(
        results[0],
        random_result,
        save_path=os.path.join(run_dir, "comparison.png"),
        show=not no_show
    )
    
    # 生成汇总报告
    report_path = os.path.join(run_dir, "report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("模型决策路径可视化报告\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("【训练模型评估结果】\n")
        f.write("-" * 40 + "\n")
        for i, result in enumerate(results):
            f.write(f"回合 {i+1}:\n")
            f.write(f"  步数: {result['steps']}\n")
            f.write(f"  最终覆盖率: {result['final_coverage']:.2%}\n")
            f.write(f"  总奖励: {result['episode_reward']:.2f}\n")
            f.write(f"  平均节点数: {np.mean(result['node_nums']):.1f}\n\n")
        
        f.write("【随机模型对比结果】\n")
        f.write("-" * 40 + "\n")
        f.write(f"步数: {random_result['steps']}\n")
        f.write(f"最终覆盖率: {random_result['final_coverage']:.2%}\n")
        f.write(f"总奖励: {random_result['episode_reward']:.2f}\n")
    
    print(f"\n✓ 可视化完成！结果保存在: {run_dir}")
    print(f"  报告文件: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='模型决策路径可视化')
    parser.add_argument('--model-path', type=str, default='./checkpoints/best_model.pt',
                       help='模型路径')
    parser.add_argument('--n-episodes', type=int, default=3,
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
                       help='计算设备')
    parser.add_argument('--output-dir', type=str, default='./visualizations',
                       help='输出目录')
    parser.add_argument('--no-show', action='store_true',
                       help='不显示图像（只保存）')
    parser.add_argument('--no-anim', action='store_true',
                       help='不生成动画')
    
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
        output_dir=args.output_dir,
        no_show=args.no_show,
        create_anim=not args.no_anim
    )
