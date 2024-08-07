from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import tree
import numpy as np
import math
from tqdm import tqdm
import ray
import ray.rllib.algorithms.ppo as ppo
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
import dgl
from kgcnn.layers.casting import CastEdgeIndicesToDisjointSparseAdjacency
from kgcnn.layers.gather import GatherNodes
from kgcnn.layers.modules import Dense, LazyConcatenate  # ragged support
from kgcnn.layers.aggr import AggregateWeightedLocalEdges
from kgcnn.layers.pooling import PoolingNodes
from kgcnn.layers.norm import GraphLayerNormalization


# for gpu in tf.config.experimental.list_physical_devices("GPU"):
#     tf.config.experimental.set_virtual_device_configuration(gpu,
#     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])

NUM_LAYERS = 2  # Hard-code number of layers in the edge/node/global models.
LATENT_SIZE = 16  # Hard-code latent layer sizes for demos.
ACTION_DIM = 128
BUFFER_SIZE = 10

_model_config = {
        "in_feats": 1,
        "n_hidden": 4,
        "n_classes": 1,
        "n_layers": 1,
        "activation": "relu",
        "dropout": 0.2
}


class Actor(TFModelV2):
    """Custom model for policy gradient algorithms."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(Actor, self).__init__(
            obs_space, action_space, num_outputs, model_config, name
        )
        self.in_feats = _model_config["in_feats"]
        self.n_hidden = _model_config["n_hidden"]
        self.n_classes = _model_config["n_classes"]
        self.n_layers = _model_config["n_layers"]
        self.activation = _model_config["activation"]
        self.dropout = _model_config["dropout"]


        # 输入的占位符
        self.last_direct = tf.keras.layers.Input(shape=(2,), name="last_direct", dtype="float32")
        self.node_num = tf.keras.layers.Input(shape=(), name="node_num", dtype="int32")
        self.edge_num = tf.keras.layers.Input(shape=(), name="edge_num", dtype="int32")
        self.node_attr = tf.keras.layers.Input(shape=(None, self.in_feats), name="node_attr", dtype="float32")
        self.edge_weight = tf.keras.layers.Input(shape=(None, 1), name="edge_attr", dtype="float32")
        self.edge_index = tf.keras.layers.Input(shape=(None, 2), name="edge_index", dtype="int32")
        #
        self.initializer = tf.keras.initializers.GlorotNormal()
        # 模型输入的embedding，图pooling层，action和value的输出头
        self.fc_node_in = Dense(self.n_hidden)
        self.fc_direct_in = Dense(self.n_hidden)
        self.fc_action_out = Dense(self.n_classes * 2)  # 乘2是因为ray用均值和方差来表示action的正态分布
        self.fc_value_out = Dense(1)

        # GNN组件
        self.gather_layers = []
        self.message_fc_layers = []
        self.aggr_layers = []
        self.fc_node_layers = []
        self.fc_uni_layers = []

        for i in range(self.n_layers):
            self.gather_layers.append(GatherNodes())
            self.message_fc_layers.append(Dense(self.n_hidden, activation=self.activation))
            self.aggr_layers.append(AggregateWeightedLocalEdges(is_sorted=False))
            self.fc_node_layers.append(Dense(self.n_hidden, activation=self.activation))
            self.fc_uni_layers.append(Dense(self.n_hidden, activation=self.activation))

        # 模型的forward过程
        node_num = tf.reshape(self.node_num, [-1])
        edge_num = tf.reshape(self.edge_num, [-1])

        to_ragged = tf.keras.layers.Lambda(
            lambda inputs: tf.RaggedTensor.from_tensor(inputs[0], lengths=inputs[1], ragged_rank=1))

        pooling = tf.keras.layers.Lambda(
            lambda inputs: tf.reduce_mean(inputs, axis=1))


        node_attr = to_ragged([self.node_attr, node_num])
        edge_weight = to_ragged([self.edge_weight, edge_num])
        edge_index = to_ragged([self.edge_index, edge_num])
        print("SHAPES:", node_attr.shape, edge_weight.shape, edge_index.shape)

        node = self.fc_node_in(node_attr)
        # norm = GraphLayerNormalization(epsilon=1e-7)
        # node_attr = norm(node_attr)
        direct_embed = self.fc_direct_in(self.last_direct)
        uni_node = None
        # print("SHAPES:", node.shape, node_attr.shape)

        for i in range(self.n_layers):
            node_in_out = self.gather_layers[i]([node, edge_index])  # 这其实是一个edge
            node_message = self.message_fc_layers[i](node_in_out)  # 这是edge收集到的feature进入一个全连接层
            node_update = self.aggr_layers[i]([node, node_message, edge_index, edge_weight])  #
            node = self.fc_node_layers[i](LazyConcatenate(axis=-1)([node, node_update]))
            if i == 0:
                uni_node = pooling(node)
                uni_node = self.fc_uni_layers[i](uni_node)
                continue
            # else:
            # uni_node = self.fc_uni_layers[i](tf.concat([uni_node, mean_pooling(node)]), axis=-1)
            uni_node = self.fc_uni_layers[i](LazyConcatenate(axis=-1)([pooling(node), uni_node]))


        graph_embedding = uni_node
        # direct_embedding = self.fc_direct(self.last_direct)
        # graph_embedding = tf.keras.layers.Concatenate()([graph_embedding, direct_embedding])
        # print("Embedding Shape:", graph_embedding.shape, direct_embedding.shape)

        # action = self.fc_action_out(LazyConcatenate(axis=-1)([graph_embedding, direct_embed]))
        # value = self.fc_value_out(LazyConcatenate(axis=-1)([graph_embedding, direct_embed]))
        action = self.fc_action_out(graph_embedding)
        value = self.fc_value_out(graph_embedding)

        self.base_model = tf.keras.Model([self.node_attr, self.last_direct, self.edge_index, self.edge_weight, self.node_num, self.edge_num], [action, value, node, node_attr])
        self.base_model.summary()


    # def forward_backup(self, input_dict, state, seq_lens):
    #     """
    #     传入的input_dict是的值的第一维是batch维度，先要从batch中提出图，做成图batch送进dgl的keras层。
    #     :param input_dict: {"obs": {"node_feat": [None, node_num, in_dim], "edge_feat": [None, edge_num, 2 + 1]]}
    #     :param state: 循环神经网络的状态，没有用到
    #     :param seq_lens: 没有序列，没有用到
    #     :return: 返回的是GNN的输出的动作信息，和循环神经网络的状态
    #     """
    #     node = input_dict["obs"]["node_feat"]
    #     edge = input_dict["obs"]["edge_feat"][:, :, 2:]
    #     edge_index = input_dict["obs"]["edge_feat"][:, :, :2]
    #
    #     node = tf.RaggedTensor.from_tensor(node)
    #     edge = tf.RaggedTensor.from_tensor(edge)
    #     edge_index = tf.cast(tf.RaggedTensor.from_tensor(edge_index), dtype="int32")
    #
    #     # print("Shape:", node.shape, edge.shape, edge_index.shape)
    #     action, self._value_out = self.base_model([node, edge_index, edge])
    #
    #     return action, state


    def forward(self, input_dict, state, seq_lens):
        """
        传入的input_dict是的值的第一维是batch维度，先要从batch中提出图，做成图batch送进dgl的keras层。
        :param input_dict: {"obs": {"node_feat": [None, node_num, in_dim], "edge_feat": [None, edge_num, 2 + 1]]}
        :param state: 循环神经网络的状态，没有用到
        :param seq_lens: 没有序列，没有用到
        :return: 返回的是GNN的输出的动作信息，和循环神经网络的状态
        """
        # print("Raw Env Observation:\n", input_dict["obs"])
        # node_num = tf.reshape(tf.cast(input_dict["obs"]["node_num"], dtype="int32"), [-1])
        # edge_num = tf.reshape(tf.cast(input_dict["obs"]["edge_num"], dtype="int32"), [-1])
        node_num = tf.cast(input_dict["obs"]["node_num"], dtype="int32")
        edge_num = tf.cast(input_dict["obs"]["edge_num"], dtype="int32")

        print(f"Nums,\n Node:{node_num}, Edge:{edge_num}")
        # last_direct = tf.cast(input_dict["obs"]["last_direct"], dtype="float32")
        node = input_dict["obs"]["node_feat"][:, :, :3] * tf.constant([1., 1., 100.], dtype="float32") - tf.constant([0., 0., 1.], dtype="float32")
        node = node[:, :, 2] * -1
        node = tf.expand_dims(node, axis=-1)

        edge = 1 / (input_dict["obs"]["edge_feat"][:, :, 2:] + 0.001)
        edge_index = input_dict["obs"]["edge_feat"][:, :, :2]
        last_direct = input_dict["obs"]["last_direct"]  # [batch_size, 2]
        print(f"Orgin Shape,\n Node:{node.shape}, Edge:{edge.shape}, Edge_index:{edge_index.shape}, Last_direct:{last_direct.shape}")

        action, self._value_out, node, node_attr = \
            self.base_model([node, last_direct, edge_index, edge, node_num, edge_num])
        print(f"Model output,\n Action:{action.shape}, Value:{self._value_out.shape}, Node:{node.shape}, Node_attr:{node_attr.shape}")
        # print("Row id:", action.value_rowids(), action.values)
        return action, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])

    def metrics(self):
        return {"value": self._value_out}



if __name__ == "__main__":
    pass


