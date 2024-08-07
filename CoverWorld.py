# 文件名称：CoverWorld.py
# 作者：denny
# 创建日期：2023-10-09
# 描述：表示离散的地图和无人机的图
import math
import time
import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
import gymnasium as gym
from gym import ActionWrapper, ObservationWrapper, RewardWrapper, Wrapper
import pygame
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import random
import dgl
from gymnasium import spaces
from gymnasium.spaces import Sequence, Box, Discrete, Tuple
from ray.rllib.utils.spaces.repeated import Repeated
from collections import defaultdict, OrderedDict
from pylab import show
from graph_nets import utils_tf
from graph_nets import utils_np
from ray.rllib.env.env_context import EnvContext

# for gpu in tf.config.experimental.list_physical_devices("GPU"):
#     tf.config.experimental.set_virtual_device_configuration(gpu,
#     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])


# matplotlib.use('TKAgg')
AROUND = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=int)  # 用于连接图上下左右的边
NEAR_BY = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=int)  # 用于找到rob周围的格点
COVER_R = 1.8
STEP = 1.42
BUFFER_SIZE = 2000
ANGLE_SCALE = 0.
ANGLE_BIAS = 0.0
COVER_SCALE = 0.
REWARD_BIAS = 0.
INPUT_FIELDS = ['nodes', 'edges', 'receivers', 'senders', 'globals', 'n_node', 'n_edge']
INPUT_DIMS = []
CONSTRUCT_DIS = 6  # 用于控制构建神经网络输入的子图的大小
ENV_CONFIG = {
    "width": 10,
    "height": 10,
    "render_mode": "human",
    "max_episodes_length": 200
}
MAX_EPISODE_LENGTH = 200
COVER_MODERATE = 1.0
COVER_RECORD_LENGTH = 10


class GridGraph:
    # 先生成一个任意的矩形图
    def __init__(self,
                 width=100,
                 height=100,
                 max_episode_steps=1000,
                 seed=1):
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
        # self.out_of_map = 0  # 控制训练不会长时间停留在外部
        self.covered_num = 0
        self.covered_record = []  #
        self.grid_points = defaultdict()  # 存储边界顶点
        self.grid_map = np.ones([self.width, self.height], dtype=np.int32) * -1  # 存储顶点id
        self.grid_covered_map = np.ones([self.width, self.height], dtype=np.int8) * -1  # 覆盖标记
        self._seed(seed)

    def reset(self, seed=None, random_init_pos=False):
        self.max_num_points = int(0.6 * self.width * self.height)
        self.graph.clear()
        self.num_point = 0  # 当前顶点数量
        self.num_step = 0
        self.rob_id = 0
        self.rob_pos = np.array([0., 0.])
        self.rob_old_angle = np.random.uniform([2 * np.pi])
        self.rob_old_direct = np.array([np.cos(np.random.uniform(self.rob_old_angle)), np.sin(np.random.uniform(self.rob_old_angle))]).reshape([-1])  # 初始朝向随机
        self.rob_last_reward = 0
        self.terminated = False
        self.covered_num = 0
        self.covered_record.clear()
        self.rob_obs_graph = None  # nx图
        self.rob_obs_dict.clear()  # 返回给gym的dict
        self.grid_points.clear()
        self.grid_map = np.ones([self.width, self.height], dtype=float) * -1  # 存储顶点id
        self.grid_covered_map = np.ones([self.width, self.height], dtype=float) * -1  # 覆盖标记
        if seed:
            self.seed = seed
        self._seed(seed)
        self.generate(random_init_pos=random_init_pos)
        self.graph_to_input()
        self.generate_dgl_graph()

    @staticmethod
    def _seed(seed):
        random.seed(seed)
        np.random.seed(seed)

    def _is_grid(self, x, y):
        x = int(x)
        y = int(y)
        if (0 <= x < self.width) and (0 <= y < self.height):
            return True
        else:
            return False

    def _match_test(self):
        # 确保图的覆盖信息和矩阵的覆盖信息一致
        print(self.graph.nodes[1])
        for i in range(self.width):
            for j in range(self.height):
                if self.grid_map[i, j] != -1:
                    if self.graph.nodes[self.grid_map[i, j]]['attr']['covered'] != self.grid_covered_map[i, j]:
                        print(i, j)
                        return False
        return True

    def _add_grid_node(self, x, y):
        # 将坐标x,y的点加入图中，并添加邻边，更新边界表，和网格图
        # 若x，y在范围外或者已有顶点，则返回
        if not (0 <= x < self.width) or not (0 <= y < self.height) or self.grid_map[x][y] != -1:
            return False

        # 添加结点并将以中心点（width/2， height/2）作为原点的坐标作为属性输入图中
        self.grid_covered_map[x, y] = 0  # 添加的点的标记为未覆盖
        self.graph.add_node(self.num_point, attr={"id": self.num_point, "position": (x - int(self.width / 2), y - int(self.height / 2)), "covered": 0, "draw_pos": (x, y)})
        # 添加度统计量，并对边界上的点进行相应的减小
        self.grid_points[(x, y)] = 4
        if x == 0 or x == self.width - 1:
            self.grid_points[(x, y)] -= 1
        if y == 0 or y == self.height - 1:
            self.grid_points[(x, y)] -= 1
        self.grid_map[x][y] = self.num_point
        # 对该顶点添加边
        for i, j in AROUND:
            x0 = x - i
            y0 = y - j
            if (0 <= x0 < self.width) and (0 <= y0 < self.height) and self.grid_map[x0][y0] != -1:
                self.graph.add_edge(self.num_point, self.grid_map[x0][y0], attr={"distance": 1.})
                # 对添加边的顶点减少1，当为0时则其无法添加邻点，pop出来
                self.grid_points[(x0, y0)] -= 1
                self.grid_points[(x, y)] -= 1
                # 添加之后需要检测边两侧的点的度
                if self.grid_points[(x0, y0)] == 0:
                    self.grid_points.pop((x0, y0))
                if self.grid_points[(x, y)] == 0:
                    self.grid_points.pop((x, y))

        self.num_point += 1
        return True

    def _grid_generate(self):
        # 生成地图网格的矩阵表示
        for i in range(self.max_num_points):
            if len(list(self.grid_points.keys())) == 0:
                x = int(self.width / 2)
                y = int(self.height / 2)
                self._add_grid_node(x, y)
            else:
                idx = random.choice(list(self.grid_points.keys()))
                for idx_cord in random.sample(range(4), 4):
                    cord = [AROUND[idx_cord][0], AROUND[idx_cord][1]]
                    cord[0] = cord[0] + idx[0]
                    cord[1] = cord[1] + idx[1]
                    if not 0 <= cord[0] < self.width or not 0 <= cord[1] < self.height:
                        continue
                    if self.grid_map[cord[0], cord[1]] != -1:
                        continue
                    if not self._add_grid_node(cord[0], cord[1]):
                        print("Not Added.")
                    break

    def _is_in_map(self, x, y):
        if not (0 <= x < self.width) or not (0 <= y < self.height):
            return False
        return True

    def _is_connected(self, x, y):
        # 判断rob是否和地图的顶点有连接
        x += self.width / 2
        y += self.height / 2
        neighbors = [(math.floor(x), math.floor(y)), (math.floor(x+1), math.floor(y)), (math.floor(x), math.floor(y+1)), (math.floor(x+1), math.floor(y+1))]
        for neighbor in neighbors:
            if not self._is_in_map(neighbor[0], neighbor[1]):
                continue
            elif self.grid_covered_map[neighbor[0], neighbor[1]] == -1:
                continue
            else:
                return True
        return False


    def _add_rob_node(self, x, y):
        # 根据坐标将无人机添加到地图中，x，y为对矩阵的0，0的坐标，不是对中心点坐标，输入图的属性是对中心点的坐标，注意！
        # 周围的四个点的坐标
        x -= self.width / 2
        y -= self.height / 2
        self.rob_pos = [x, y]

        assert -self.width / 2 <= self.rob_pos[0] <= self.width / 2 and -self.height / 2 <= self.rob_pos[1] <= self.height / 2, f"ERROR in _add_rob_node:Rob pos is not in map! pos: {self.rob_pos}"

        neighbors = [(math.floor(x), math.floor(y)), (math.floor(x+1), math.floor(y)), (math.floor(x), math.floor(y+1)), (math.floor(x+1), math.floor(y+1))]  # 把int改成math.floor
        # grids = [(a, b) for a, b in grids if (0 <= a < self.width) and (0 <= b < self.height)]  # 除去在地图上的点
        a = x - math.floor(x)
        b = y - math.floor(y)
        l1 = (a ** 2 + b ** 2) ** 0.5
        l2 = ((1-a) ** 2 + b ** 2) ** 0.5
        l3 = (a ** 2 + (1-b) ** 2) ** 0.5
        l4 = ((1-a) ** 2 + (1-b) ** 2) ** 0.5
        ls = [l1, l2, l3, l4]
        distances = defaultdict(list)  # {(x,y): distance}
        for i in range(4):
            distances[neighbors[i]] = ls[i]

        nodes = self.graph.nodes(data=True)
        ids = defaultdict(bool)  # {(x,y): neighbor_id}
        # 通过坐标找到邻居的id
        for node in nodes:
            node = node[1]['attr']
            if node['position'] in neighbors:
                ids[node['position']] = node['id']

        # 将rob顶点添加到图中
        # print("Test neighbors:", (x, y), neighbors, ls, ids)
        self.rob_id = self.num_point
        self.graph.add_node(self.num_point, attr={"id": self.rob_id, "position": (x, y), "covered": -2, "draw_pos": ( x + self.width / 2, y + self.height / 2)})
        self.num_point += 1
        # 和周围的点添加边
        for neighbor in neighbors:
            # print("Test edge:", idx)
            id = ids[neighbor]
            # print(neighbor, id)
            if id is False:
                continue
            # print(self.rob_id, ids[neighbor], distances[neighbor])
            self.graph.add_edge(self.rob_id, ids[neighbor], attr={"distance": distances[neighbor]})
        return self._rob_cover(x, y)  # 覆盖顶点

    def _remove_rob_node(self):
        # 删除rob在图中的顶点
        self.graph.remove_node(self.rob_id)
        self.num_point -= 1
        self.rob_id = -1
        self.rob_pos = np.array([0., 0.])

    def _rob_move(self, direct, step):
        # direct is a 2d array, step is a constant
        hit = False
        pos = self.graph.nodes(data=True)[self.rob_id]['attr']['position']  # 元组
        pos = np.array(pos, dtype=np.float32)  # 转换成array
        direct = np.array(direct)
        self.rob_old_angle = math.atan2(direct[1], direct[0]) % (2*np.pi)  # 计算角度

        cos_change_angle = np.sum(self.rob_old_direct * direct)/ (np.sum(self.rob_old_direct * self.rob_old_direct) ** 0.5 * np.sum(direct * direct) ** 0.5)
        print(f"iter{self.num_step} Cos:", cos_change_angle, self.rob_old_direct, direct)
        # 计算夹脚的余弦值（-1，1）用于计算奖励

        pos = pos + direct * step
        pos = pos + np.array([self.width / 2, self.height / 2])
        if not self._is_in_map(pos[0], pos[1]) or not self._is_connected(pos[0] - self.width /2, pos[1] - self.height / 2):
            cover_num = 0  # No postion change when rob is out of map or not connected
            hit = True
        else:
            # Update Env state
            self.rob_old_direct = direct
            self._remove_rob_node()
            cover_num = self._add_rob_node(*pos)

        return cos_change_angle, cover_num, hit

    def _rob_cover(self, x, y):
        # cover the grid point within COVER_R, x, y 为以地图中心为原点的坐标 
        cover_num = 0  # 统计新cover的顶点，用于计算奖励
        x = x + self.width / 2
        y = y + self.height / 2
        points = []
        r = COVER_R
        for i in range(int(x-r), int(x+r+1)):
            for j in range(int(y - r), int(y + r + 1)):
                #  for j in range(int(y-r), int(y+r+3)):
                if (i-x) ** 2 + (j-y) ** 2 <= r ** 2:
                    point = (i, j)  # 不知道为什么但是需要加一
                    if self._is_grid(*point) and self.grid_map[i, j] != -1:
                        # 记录在范围内的合法grid的id
                        points.append(self.grid_map[i, j])
                        self.grid_covered_map[i, j] = 1  # 对覆盖顶点标记1
        # 将对应的点的覆盖标记设置为1
        # print(self.graph.nodes(data=True))
        for idx in points:
            # print(self.graph.nodes(data=True)[int(idx)])
            if self.graph.nodes(data=True)[int(idx)]['attr']['covered'] == 0:
                cover_num += 1
                self.graph.nodes(data=True)[int(idx)]['attr']['covered'] = 1  # 放在if里因为-1的点不是需要覆盖的区域
        self.covered_num += cover_num
        return cover_num


    def _reward(self, cos_change_angle=-1, hit=False):
        # 计算奖励
        cos = np.array(cos_change_angle)  # （-1，1）
        # assert -1 <= cos <= 1 and cos_change_angle != -2, f"ERROR in _reward: cos_change_angle is not in [-1, 1]! cos_change_angle: {cos_change_angle}"
        reward = 0
        if hit:
            reward = - self.max_num_points + self.covered_num
        if cos > 0.9 or cos < -0.9:
            reward = -10
        return reward


    def move(self, direct, step=STEP):
        # Direct 既可以是向量，也可以是角度
        self.num_step += 1

        if isinstance(direct, float):
            direct = np.array([math.cos(direct), math.sin(direct)])
        else:
            direct = np.array(direct)
        # print("Test fore rob pos:", self.rob_pos, self.graph.nodes(data=True)[self.rob_id]['attr']['position'])
        cos_change_angle, cover_num, hit = self._rob_move(direct, step)
        if hit:
            print("hit")

        # Record the covered num
        self.covered_record.append(cover_num)
        if len(self.covered_record) > COVER_RECORD_LENGTH:
            self.covered_record.pop(0)
        # # print("Test rob pos:", self.rob_pos, self.graph.nodes(data=True)[self.rob_id]['attr']['position'])

        # 若rob与地图的顶点没有连接，则返回一个负的奖励，同时结束episode
        # if cos_change_angle == -2:
        #     return self._reward() - (self.max_num_points - self.covered_num) * 10, True

        reward = self._reward(cos_change_angle, hit)
        self.rob_last_reward = reward
        self.terminated = (self.covered_num >= self.max_num_points * COVER_MODERATE) or (self.num_step >= self.max_episode_steps) or hit
        if self.terminated and not self.covered_num >= self.max_num_points * COVER_MODERATE:
            reward -= (self.max_num_points - self.covered_num) / self.max_num_points * 100  # 没有完成任务的惩罚
        self.graph_to_input()
        self.generate_dgl_graph()
        return reward, self.terminated


    def generate(self, random_init_pos=False):
        # 生成随机网格地图
        self._grid_generate()
        # 添加随机robot
        if not random_init_pos:
            self._add_rob_node(self.width / 2 + 0.5, self.height / 2 + 0.5)
        else:
            self._add_rob_node(random.random() * self.width, random.random() * self.height)


    def _custom_pos(self, pos):
        # 定义pos到rob_pos的距离函数
        pos = np.array(pos)
        pos_rob = np.array([math.floor(self.rob_pos[0]), math.floor(self.rob_pos[1])]) + 0.5
        dis = np.sum((pos_rob - pos) ** 2) ** 0.5
        # dis = np.sum(np.abs(self.rob_pos - pos))
        return dis


    def graph_to_input(self, local=True, to_dict=False):
        # 将图转换为图神经网络的输入
        def create_features(attr, fields):
            ans = np.hstack([np.array(attr[field]) for field in fields])
            return ans

        node_fields = ['position', 'covered']  # (3)
        edge_fields = ['distance']  # (1)

        input_graph = nx.MultiDiGraph()  # Graph_nets要求为多有向图，更一般化
        # 对每个点提取拼接特征
        node_num = 0
        node_id = defaultdict(int)
        for node_idx, node_feature in self.graph.nodes(data=True):
            if local:
                dis = self._custom_pos(node_feature['attr']['position'])
                if dis >= CONSTRUCT_DIS:
                    continue
            input_graph.add_node(node_num, features=create_features(node_feature['attr'], node_fields), attr=node_feature['attr'])
            node_id[node_idx] = node_num
            node_num += 1
        # 对每条边提取特征
        for receiver, sender, edge_features in self.graph.edges(data=True):
            if local:
                if receiver not in node_id.keys() or sender not in node_id.keys():
                    continue
            r_id = node_id[receiver]
            s_id = node_id[sender]
            input_graph.add_edge(r_id, s_id, features=create_features(edge_features['attr'], edge_fields))
            input_graph.add_edge(s_id, r_id, features=create_features(edge_features['attr'], edge_fields))

        input_graph.graph['features'] = self.rob_old_direct  # 全局顶点
        self.rob_obs_graph = input_graph
        self.rob_obs_dict['graph'] = utils_np.networkx_to_data_dict(input_graph)
        self.rob_obs_dict['map'] = self.grid_covered_map

        # input_dict = utils_np.networkx_to_data_dict(input_graph)  # 转换为字典
        # input_tuple = utils_np.networkxs_to_graphs_tuple([input_graph])  # 转换为graph tuple
        # print(input_tuple)
        # # print("Test:", input_dict.keys(), input_tuple)


    def generate_dgl_graph(self):
        """
        生成actor所需要dgl的图和feature
        :return: node_feat(Num of nodes, 4), edge_feat(source ,destination, distance)
        """
        # print("Graph_Nodes:", self.graph.nodes(data=True))

        # for node in self.graph.nodes(data=True):
        #     print("One node:", node[0], node[1].keys())
        #
        # for edge in self.graph.edges(data=True):
        #     if self.rob_id in [edge[0], edge[1]]:
        #         print("One edge:", edge)

        node_feat = []  # [x, y, covered, old_direct]
        edge_feat = []  # [source, destination, distance]
        node = []  # [x, y]

        # node and edge from the whole map
        for n in self.graph.nodes(data=True):
            n = n[1]['attr']
            node_feat.append(np.array([n['position'][0], n['position'][1], n['covered'], self.rob_old_direct[0], self.rob_old_direct[1]]))
            node.append([n['position'][0], n['position'][1]])

        for edge in self.graph.edges(data=True):
            x, y = edge[0], edge[1]
            edge = edge[2]['attr']
            edge_feat.append(np.array([x, y, edge['distance']]))

        # Padding for repeated space
        for i in range(len(node), self.node_padding_len):
            node_feat.append(np.array([0, 0, 0, 0, 0], dtype="float32"))

        for i in range(len(edge_feat), self.edge_padding_len):
            edge_feat.append(np.array([0, 0, 0], dtype="float32"))

        self.node_feat = np.array(node_feat, dtype="float32")
        self.edge_feat = np.array(edge_feat, dtype="float32")
        self.node_num = np.array([len(node_feat)], dtype="float32")
        self.edge_num = np.array([len(edge_feat)], dtype="float32")

        # print("Feature:", self.node_feat, self.edge_feat)

        return node_feat, edge_feat

    def drawgraph(self, local=False):
        # 把robot标红
        color = []
        if local:
            g = self.rob_obs_graph
        else:
            g = self.graph

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
        labels = {}

        for node in g.nodes(data=True):
            pos[node[0]] = node[1]['attr']['position']
            labels[node[0]] = node[1]['attr']['position']

        # pos = nx.spring_layout(g)
        # plt.subplot(121)
        # nx.draw_spectral(self.graph)
        # plt.subplot(122)
        nx.draw(g, pos, node_color=color, node_size=40)  # 用于边不重叠的作图
        # nx.draw(self.rob_obs_graph)
        # plt.subplot(122)
        # nx.draw(g2)
        show()


    def test(self):
        self.reset(seed=2)
        self.drawgraph()
        print("TEST COVER:", self.rob_obs_dict['map'])


class CoverWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, env_config: EnvContext):
        render_mode = env_config['render_mode']
        self.width = env_config['width']
        self.height = env_config['height']
        self.grid_graph = GridGraph(self.width, self.height, env_config['max_episodes_length'])
        self.observation_space = spaces.Dict({
                "edge_feat": Box(
                    low=np.array([0, 0, 0] * self.grid_graph.edge_padding_len).reshape([-1, 3]),
                    high=np.array([self.width * self.height, self.width * self.height, 2] * self.grid_graph.edge_padding_len).reshape([-1, 3]),
                    dtype=np.float32)
                ,
                "edge_num": Box(low=0, high=self.width * self.height * 4, dtype=np.float32),
                "last_direct": Box(low=-1, high=1, shape=(2,), dtype=np.float32),
                "node_feat": Box(
                    low=np.array([-self.width / 2, -self.height / 2, -100, -1, -1] * self.grid_graph.node_padding_len).reshape([-1, 5]),
                    high=np.array([self.width / 2, self.height / 2, 2, 1, 1] * self.grid_graph.node_padding_len).reshape([-1, 5]),
                    dtype=np.float32)
                ,
                "node_num": Box(low=0, high=self.width * self.height, dtype=np.float32),
            })
        self.action_space = Box(low=0, high=2 * np.pi, shape=(1,), dtype=np.float32)

        self._action_to_direct = lambda action: np.array([math.cos(action), math.sin(action)])

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None

    def _get_obs(self):
        return OrderedDict({
            "edge_feat": self.grid_graph.edge_feat,
            "edge_num": self.grid_graph.edge_num,
            "last_direct": self.grid_graph.rob_old_direct,
            "node_feat": self.grid_graph.node_feat,
            "node_num": self.grid_graph.node_num,
        })

    def _get_info(self):
        return {"reward": self.grid_graph.rob_last_reward}

    def reset(self, seed=None, return_info=True, options=None):
        super().reset(seed=seed)
        self.grid_graph.reset(seed)
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action=None):
        print("Action is taken:", action, "Type: ", type(action))
        if action == None:
            action = random.random() * 2 * np.pi
        else:
            action = np.array([math.cos(action), math.sin(action)])
        reward, terminated = self.grid_graph.move(action)
        observation = self._get_obs()
        info = self._get_info()
        return observation, reward, terminated, terminated, info

    def render(self, mode="human"):
        return self._render_frame()

    def _render_frame(self):
        self.grid_graph.drawgraph()



def cover_env_test():
    env = CoverWorldEnv(env_config=ENV_CONFIG)
    env.reset()
    action = random.uniform(0, 2 * np.pi)
    g = dgl.from_networkx(env.grid_graph.rob_obs_graph)
    for i in range(10):
        obs, r, done, trunc, info = env.step(action)
        print(obs, r, done, trunc)
        print("Obs box shape", obs["node_feat"].shape, obs["edge_feat"].shape)
        print("Step: ", i, " Reward: ", r, "Record: ", env.grid_graph.covered_record)
        if done:
            break
    pos = env.grid_graph.rob_pos
    map = env.grid_graph.grid_covered_map
    env.render()
    # print(env.observation_space.sample())
    return g, pos, map


if __name__ == '__main__':
    # Test GridGraph
    # env = GridGraph(width=10, height=10)
    # env.test()
    # Test CoverWorldEnv
    for i in range(3):
        g, pos, map = cover_env_test()

