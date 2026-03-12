# -*- coding: utf-8 -*-
"""
文件名称：env.py
作者：denny
创建日期：2023-10-09
描述：Gymnasium 环境包装器
"""

import math
import random
from collections import OrderedDict

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .grid_graph import GridGraph, CONSTRUCT_DIS, MAX_EPISODE_LENGTH


class CoverWorldEnv(gym.Env):
    """覆盖世界环境"""
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, env_config=None):
        """
        初始化环境
        
        Args:
            env_config: 环境配置字典，包含:
                - width: 地图宽度
                - height: 地图高度
                - render_mode: 渲染模式
                - max_episodes_length: 最大回合长度
                - seed: 随机种子
        """
        if env_config is None:
            env_config = {}
            
        self.width = env_config.get('width', 10)
        self.height = env_config.get('height', 10)
        self.max_episodes_length = env_config.get('max_episodes_length', MAX_EPISODE_LENGTH)
        self.seed = env_config.get('seed', None)
        render_mode = env_config.get('render_mode', None)
        
        # 初始化网格图
        self.grid_graph = GridGraph(
            self.width, 
            self.height, 
            self.max_episodes_length,
            self.seed if self.seed else 1
        )
        
        # 定义观察空间 (使用 PyTorch Geometric 的格式)
        # 节点特征: [N, 5] (x, y, covered, direction_x, direction_y)
        # 边索引: [2, E] (source, destination)
        # 边特征: [E, 1] (distance)
        max_nodes = self.grid_graph.node_padding_len
        max_edges = self.grid_graph.edge_padding_len
        
        self.observation_space = spaces.Dict({
            'node_feat': spaces.Box(
                low=np.array([-self.width / 2, -self.height / 2, -2, -1, -1] * max_nodes).reshape([-1, 5]),
                high=np.array([self.width / 2, self.height / 2, 2, 1, 1] * max_nodes).reshape([-1, 5]),
                dtype=np.float32
            ),
            'edge_index': spaces.Box(
                low=0,
                high=max_nodes,
                shape=(2, max_edges),
                dtype=np.int64
            ),
            'edge_attr': spaces.Box(
                low=0,
                high=10,
                shape=(max_edges, 1),
                dtype=np.float32
            ),
            'node_num': spaces.Box(low=0, high=max_nodes, shape=(1,), dtype=np.int64),
            'edge_num': spaces.Box(low=0, high=max_edges, shape=(1,), dtype=np.int64),
        })
        
        # 动作空间: 连续角度 [0, 2π]
        self.action_space = spaces.Box(low=0, high=2 * np.pi, shape=(1,), dtype=np.float32)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None

    def _get_obs(self):
        """获取当前观察"""
        # 填充到固定大小
        node_feat = np.zeros((self.grid_graph.node_padding_len, 5), dtype=np.float32)
        edge_index = np.zeros((2, self.grid_graph.edge_padding_len), dtype=np.int64)
        edge_attr = np.zeros((self.grid_graph.edge_padding_len, 1), dtype=np.float32)
        
        n_nodes = min(self.grid_graph.node_num, self.grid_graph.node_padding_len)
        n_edges = min(self.grid_graph.edge_num, self.grid_graph.edge_padding_len)
        
        if n_nodes > 0:
            node_feat[:n_nodes] = self.grid_graph.node_feat[:n_nodes]
        if n_edges > 0:
            edge_index[:, :n_edges] = self.grid_graph.edge_index[:, :n_edges]
            edge_attr[:n_edges] = self.grid_graph.edge_attr[:n_edges]
        
        return OrderedDict({
            'node_feat': node_feat,
            'edge_index': edge_index,
            'edge_attr': edge_attr,
            'node_num': np.array([n_nodes], dtype=np.int64),
            'edge_num': np.array([n_edges], dtype=np.int64),
        })

    def _get_info(self):
        """获取额外信息"""
        return {
            "reward": self.grid_graph.rob_last_reward,
            "covered_num": self.grid_graph.covered_num,
            "max_num_points": self.grid_graph.max_num_points,
            "coverage_rate": self.grid_graph.covered_num / self.grid_graph.max_num_points
        }

    def reset(self, seed=None, options=None):
        """重置环境"""
        super().reset(seed=seed)
        
        if seed is not None:
            self.seed = seed
            
        self.grid_graph.reset(seed=self.seed, random_init_pos=False)
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info

    def step(self, action=None):
        """执行动作"""
        if action is None:
            action = random.random() * 2 * np.pi
        else:
            # 将角度转换为方向向量
            action = np.array([math.cos(action[0]), math.sin(action[0])])
            
        reward, terminated = self.grid_graph.move(action)
        observation = self._get_obs()
        info = self._get_info()
        
        # Gymnasium API: obs, reward, terminated, truncated, info
        return observation, reward, terminated, False, info

    def render(self):
        """渲染环境"""
        if self.render_mode == "human":
            return self._render_frame()
        elif self.render_mode == "rgb_array":
            # TODO: 实现 RGB 数组渲染
            pass

    def _render_frame(self):
        """渲染一帧"""
        self.grid_graph.drawgraph()


def create_env(width=10, height=10, max_episodes_length=200, seed=None, render_mode=None):
    """
    创建环境的便捷函数
    
    Args:
        width: 地图宽度
        height: 地图高度
        max_episodes_length: 最大回合长度
        seed: 随机种子
        render_mode: 渲染模式
        
    Returns:
        CoverWorldEnv 实例
    """
    env_config = {
        'width': width,
        'height': height,
        'max_episodes_length': max_episodes_length,
        'seed': seed,
        'render_mode': render_mode
    }
    return CoverWorldEnv(env_config)
