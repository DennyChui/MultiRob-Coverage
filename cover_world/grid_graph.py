# -*- coding: utf-8 -*-
"""
文件名称：grid_graph.py
作者：denny
创建日期：2023-10-09
描述：表示离散的地图和机器人的图结构
"""

import math
import random
from collections import defaultdict

import numpy as np
import networkx as nx

# 用于连接图上下左右的边
AROUND = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=int)
# 用于找到rob周围的格点
NEAR_BY = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=int)

# 常量定义
COVER_R = 1.8  # 覆盖半径
STEP = 1.42  # 机器人步长
BUFFER_SIZE = 2000
MAX_EPISODE_LENGTH = 200
COVER_MODERATE = 1.0  # 覆盖率阈值
COVER_RECORD_LENGTH = 10
CONSTRUCT_DIS = 6  # 用于控制构建神经网络输入的子图的大小


class GridGraph:
    """网格地图图结构"""
    
    def __init__(self, width=100, height=100, max_episode_steps=1000, seed=1):
        self.seed = seed
        self.max_episode_steps = max_episode_steps
        self.width = width
        self.height = height
        self.edge_padding_len = self.height * self.width * 4
        self.node_padding_len = self.height * self.width
        self.max_num_points = int(0.6 * self.width * self.height)
        
        self.num_step = 0
        self.num_point = 0  # 当前顶点数量
        self.rob_id = 0
        self.rob_pos = np.array([0., 0.], dtype="float32")
        self.rob_old_direct = np.array([0., 0.], dtype="float32")
        self.rob_old_angle = 0.
        self.rob_last_reward = 0
        
        self.graph = nx.Graph()
        self.rob_obs_graph = None  # nx图
        self.rob_obs_dict = defaultdict()  # 返回给gym的dict
        
        self.terminated = False
        self.covered_num = 0
        self.covered_record = []
        self.grid_points = defaultdict()  # 存储边界顶点
        self.grid_map = np.ones([self.width, self.height], dtype=np.int32) * -1  # 存储顶点id
        self.grid_covered_map = np.ones([self.width, self.height], dtype=np.int8) * -1  # 覆盖标记
        
        # 特征数据 (用于神经网络输入)
        self.node_feat = None  # [N, 5] (x, y, covered_status, direction_x, direction_y)
        self.edge_feat = None  # [E, 3] (source_id, dest_id, distance)
        self.node_num = 0
        self.edge_num = 0
        
        self._seed(seed)

    def reset(self, seed=None, random_init_pos=False):
        """重置环境"""
        self.max_num_points = int(0.6 * self.width * self.height)
        self.graph.clear()
        self.num_point = 0
        self.num_step = 0
        self.rob_id = 0
        self.rob_pos = np.array([0., 0.])
        self.rob_old_angle = np.random.uniform([2 * np.pi])
        self.rob_old_direct = np.array([
            np.cos(np.random.uniform(self.rob_old_angle)),
            np.sin(np.random.uniform(self.rob_old_angle))
        ]).reshape([-1])
        self.rob_last_reward = 0
        self.terminated = False
        self.covered_num = 0
        self.covered_record.clear()
        self.rob_obs_graph = None
        self.rob_obs_dict.clear()
        self.grid_points.clear()
        self.grid_map = np.ones([self.width, self.height], dtype=float) * -1
        self.grid_covered_map = np.ones([self.width, self.height], dtype=float) * -1
        
        if seed:
            self.seed = seed
        self._seed(seed)
        self.generate(random_init_pos=random_init_pos)
        self._update_observation()

    @staticmethod
    def _seed(seed):
        """设置随机种子"""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def _is_grid(self, x, y):
        """检查坐标是否在网格范围内"""
        x, y = int(x), int(y)
        return 0 <= x < self.width and 0 <= y < self.height

    def _is_in_map(self, x, y):
        """检查坐标是否在地图范围内"""
        return 0 <= x < self.width and 0 <= y < self.height

    def _is_connected(self, x, y):
        """判断rob是否和地图的顶点有连接"""
        x += self.width / 2
        y += self.height / 2
        neighbors = [
            (math.floor(x), math.floor(y)),
            (math.floor(x + 1), math.floor(y)),
            (math.floor(x), math.floor(y + 1)),
            (math.floor(x + 1), math.floor(y + 1))
        ]
        for neighbor in neighbors:
            if not self._is_in_map(neighbor[0], neighbor[1]):
                continue
            elif self.grid_covered_map[neighbor[0], neighbor[1]] == -1:
                continue
            else:
                return True
        return False

    def _add_grid_node(self, x, y):
        """将坐标x,y的点加入图中"""
        if not (0 <= x < self.width) or not (0 <= y < self.height) or self.grid_map[x][y] != -1:
            return False

        # 添加结点并将以中心点作为原点的坐标作为属性输入图中
        self.grid_covered_map[x, y] = 0
        self.graph.add_node(
            self.num_point,
            attr={
                "id": self.num_point,
                "position": (x - int(self.width / 2), y - int(self.height / 2)),
                "covered": 0,
                "draw_pos": (x, y)
            }
        )
        
        # 添加度统计量，并对边界上的点进行相应的减小
        self.grid_points[(x, y)] = 4
        if x == 0 or x == self.width - 1:
            self.grid_points[(x, y)] -= 1
        if y == 0 or y == self.height - 1:
            self.grid_points[(x, y)] -= 1
        self.grid_map[x][y] = self.num_point
        
        # 对该顶点添加边
        for i, j in AROUND:
            x0, y0 = x - i, y - j
            if (0 <= x0 < self.width) and (0 <= y0 < self.height) and self.grid_map[x0][y0] != -1:
                self.graph.add_edge(self.num_point, self.grid_map[x0][y0], attr={"distance": 1.0})
                self.grid_points[(x0, y0)] -= 1
                self.grid_points[(x, y)] -= 1
                
                if self.grid_points[(x0, y0)] == 0:
                    self.grid_points.pop((x0, y0))
                if self.grid_points[(x, y)] == 0:
                    self.grid_points.pop((x, y))

        self.num_point += 1
        return True

    def _grid_generate(self):
        """生成地图网格的矩阵表示"""
        for i in range(self.max_num_points):
            if len(list(self.grid_points.keys())) == 0:
                x = int(self.width / 2)
                y = int(self.height / 2)
                self._add_grid_node(x, y)
            else:
                idx = random.choice(list(self.grid_points.keys()))
                for idx_cord in random.sample(range(4), 4):
                    cord = [AROUND[idx_cord][0] + idx[0], AROUND[idx_cord][1] + idx[1]]
                    if not 0 <= cord[0] < self.width or not 0 <= cord[1] < self.height:
                        continue
                    if self.grid_map[cord[0], cord[1]] != -1:
                        continue
                    if not self._add_grid_node(cord[0], cord[1]):
                        print("Not Added.")
                    break

    def _add_rob_node(self, x, y):
        """根据坐标将机器人添加到地图中"""
        x -= self.width / 2
        y -= self.height / 2
        self.rob_pos = [x, y]

        assert -self.width / 2 <= self.rob_pos[0] <= self.width / 2 and \
               -self.height / 2 <= self.rob_pos[1] <= self.height / 2, \
               f"ERROR in _add_rob_node: Rob pos is not in map! pos: {self.rob_pos}"

        neighbors = [
            (math.floor(x), math.floor(y)),
            (math.floor(x + 1), math.floor(y)),
            (math.floor(x), math.floor(y + 1)),
            (math.floor(x + 1), math.floor(y + 1))
        ]
        
        a = x - math.floor(x)
        b = y - math.floor(y)
        l1 = (a ** 2 + b ** 2) ** 0.5
        l2 = ((1 - a) ** 2 + b ** 2) ** 0.5
        l3 = (a ** 2 + (1 - b) ** 2) ** 0.5
        l4 = ((1 - a) ** 2 + (1 - b) ** 2) ** 0.5
        ls = [l1, l2, l3, l4]
        distances = defaultdict(list)
        for i in range(4):
            distances[neighbors[i]] = ls[i]

        nodes = self.graph.nodes(data=True)
        ids = defaultdict(bool)
        for node in nodes:
            node_attr = node[1]['attr']
            if node_attr['position'] in neighbors:
                ids[node_attr['position']] = node_attr['id']

        self.rob_id = self.num_point
        self.graph.add_node(
            self.num_point,
            attr={
                "id": self.rob_id,
                "position": (x, y),
                "covered": -2,
                "draw_pos": (x + self.width / 2, y + self.height / 2)
            }
        )
        self.num_point += 1
        
        for neighbor in neighbors:
            node_id = ids[neighbor]
            if node_id is False:
                continue
            self.graph.add_edge(self.rob_id, ids[neighbor], attr={"distance": distances[neighbor]})
        
        return self._rob_cover(x, y)

    def _remove_rob_node(self):
        """删除rob在图中的顶点"""
        self.graph.remove_node(self.rob_id)
        self.num_point -= 1
        self.rob_id = -1
        self.rob_pos = np.array([0., 0.])

    def _rob_move(self, direct, step):
        """机器人移动"""
        hit = False
        pos = self.graph.nodes(data=True)[self.rob_id]['attr']['position']
        pos = np.array(pos, dtype=np.float32)
        direct = np.array(direct)
        self.rob_old_angle = math.atan2(direct[1], direct[0]) % (2 * np.pi)

        cos_change_angle = np.sum(self.rob_old_direct * direct) / \
                          (np.sum(self.rob_old_direct * self.rob_old_direct) ** 0.5 * 
                           np.sum(direct * direct) ** 0.5)
        print(f"iter{self.num_step} Cos:", cos_change_angle, self.rob_old_direct, direct)

        pos = pos + direct * step
        pos = pos + np.array([self.width / 2, self.height / 2])
        
        if not self._is_in_map(pos[0], pos[1]) or \
           not self._is_connected(pos[0] - self.width / 2, pos[1] - self.height / 2):
            cover_num = 0
            hit = True
        else:
            self.rob_old_direct = direct
            self._remove_rob_node()
            cover_num = self._add_rob_node(*pos)

        return cos_change_angle, cover_num, hit

    def _rob_cover(self, x, y):
        """覆盖范围内的网格点"""
        cover_num = 0
        x = x + self.width / 2
        y = y + self.height / 2
        points = []
        r = COVER_R
        
        for i in range(int(x - r), int(x + r + 1)):
            for j in range(int(y - r), int(y + r + 1)):
                if (i - x) ** 2 + (j - y) ** 2 <= r ** 2:
                    point = (i, j)
                    if self._is_grid(*point) and self.grid_map[i, j] != -1:
                        points.append(self.grid_map[i, j])
                        self.grid_covered_map[i, j] = 1

        for idx in points:
            if self.graph.nodes(data=True)[int(idx)]['attr']['covered'] == 0:
                cover_num += 1
                self.graph.nodes(data=True)[int(idx)]['attr']['covered'] = 1
        
        self.covered_num += cover_num
        return cover_num

    def _reward(self, cos_change_angle=-1, hit=False):
        """计算奖励"""
        reward = 0
        if hit:
            reward = -self.max_num_points + self.covered_num
        if cos_change_angle > 0.9 or cos_change_angle < -0.9:
            reward = -10
        return reward

    def move(self, direct, step=STEP):
        """执行动作移动机器人"""
        self.num_step += 1

        if isinstance(direct, float):
            direct = np.array([math.cos(direct), math.sin(direct)])
        else:
            direct = np.array(direct)
            
        cos_change_angle, cover_num, hit = self._rob_move(direct, step)
        if hit:
            print("hit")

        self.covered_record.append(cover_num)
        if len(self.covered_record) > COVER_RECORD_LENGTH:
            self.covered_record.pop(0)

        reward = self._reward(cos_change_angle, hit)
        self.rob_last_reward = reward
        
        self.terminated = (
            self.covered_num >= self.max_num_points * COVER_MODERATE or 
            self.num_step >= self.max_episode_steps or hit
        )
        
        if self.terminated and not self.covered_num >= self.max_num_points * COVER_MODERATE:
            reward -= (self.max_num_points - self.covered_num) / self.max_num_points * 100
            
        self._update_observation()
        return reward, self.terminated

    def generate(self, random_init_pos=False):
        """生成随机网格地图"""
        self._grid_generate()
        if not random_init_pos:
            self._add_rob_node(self.width / 2 + 0.5, self.height / 2 + 0.5)
        else:
            self._add_rob_node(random.random() * self.width, random.random() * self.height)

    def _custom_pos(self, pos):
        """计算位置到rob_pos的距离"""
        pos = np.array(pos)
        pos_rob = np.array([math.floor(self.rob_pos[0]), math.floor(self.rob_pos[1])]) + 0.5
        return np.sum((pos_rob - pos) ** 2) ** 0.5

    def _update_observation(self, local=True):
        """更新观察值（用于神经网络输入）"""
        node_feat = []  # [x, y, covered, old_direct_x, old_direct_y]
        edge_index = []  # [source, destination]
        edge_attr = []  # [distance]

        # 从整个地图构建节点和边
        node_id_map = {}  # 原始节点id -> 新id
        new_id = 0
        
        for n in self.graph.nodes(data=True):
            orig_id = n[0]
            n_attr = n[1]['attr']
            
            if local:
                dis = self._custom_pos(n_attr['position'])
                if dis >= CONSTRUCT_DIS:
                    continue
                    
            node_feat.append([
                n_attr['position'][0],
                n_attr['position'][1],
                n_attr['covered'],
                self.rob_old_direct[0],
                self.rob_old_direct[1]
            ])
            node_id_map[orig_id] = new_id
            new_id += 1

        for edge in self.graph.edges(data=True):
            x, y = edge[0], edge[1]
            if local and (x not in node_id_map or y not in node_id_map):
                continue
            edge_attr_val = edge[2]['attr']
            edge_index.append([node_id_map[x], node_id_map[y]])
            edge_attr.append([edge_attr_val['distance']])

        self.node_feat = np.array(node_feat, dtype=np.float32)
        self.edge_index = np.array(edge_index, dtype=np.int64).T if edge_index else np.zeros((2, 0), dtype=np.int64)
        self.edge_attr = np.array(edge_attr, dtype=np.float32) if edge_attr else np.zeros((0, 1), dtype=np.float32)
        self.node_num = len(node_feat)
        self.edge_num = len(edge_index)

    def drawgraph(self, local=False):
        """绘制图"""
        import matplotlib.pyplot as plt
        
        color = []
        g = self.rob_obs_graph if local else self.graph

        for node in g.nodes(data=True):
            if node[1]['attr']['id'] == self.rob_id:
                color.append('red')
            elif node[1]['attr']['covered'] == 0:
                color.append('blue')
            elif node[1]['attr']['covered'] == 1:
                color.append('green')
            else:
                color.append('red')

        pos = {}
        for node in g.nodes(data=True):
            pos[node[0]] = node[1]['attr']['position']

        nx.draw(g, pos, node_color=color, node_size=40)
        plt.show()

    def test(self):
        """测试函数"""
        self.reset(seed=2)
        self.drawgraph()
        print("TEST COVER:", self.rob_obs_dict.get('map', 'N/A'))
