# MultiRob-Coverage

多机器人覆盖任务仿真环境，使用强化学习 (RL) 和图神经网络 (GNN) 进行训练。

## 项目重构说明

> **注意**: 本项目已完成从 TensorFlow 到 PyTorch 的重构。
> 
> - **旧文件** (保留供参考):
>   - `CoverWorld.py` - 原 TensorFlow/DGL 版本环境
>   - `actor.py` - 原 TensorFlow/KGCNN 版本模型
> 
> - **新文件** (推荐使用):
>   - `cover_world/` - PyTorch 版本环境模块
>   - `models/` - PyTorch Geometric 版本 GNN 模型
>   - `train.py` - PyTorch 版本训练脚本

## 项目重构说明

本项目已从 TensorFlow/DGL/KGCNN 迁移到 **PyTorch + PyTorch Geometric**，使用 `uv` 进行依赖管理。

### 主要变更

- **深度学习框架**: TensorFlow → PyTorch
- **图神经网络**: DGL + KGCNN + graph_nets → PyTorch Geometric
- **RL 框架**: Ray RLlib → 自定义 PPO 实现
- **包管理**: pip → [uv](https://github.com/astral-sh/uv)

## 项目结构

```
MultiRob-Coverage/
├── cover_world/          # 环境模块
│   ├── __init__.py
│   ├── grid_graph.py     # 网格图核心逻辑
│   └── env.py            # Gymnasium 环境包装器
├── models/               # 模型模块
│   ├── __init__.py
│   └── gnn_actor.py      # GNN Actor-Critic 模型
├── train.py              # 训练脚本
├── test_env.py           # 环境测试脚本
├── pyproject.toml        # 项目配置 (uv)
├── uv.lock               # 依赖锁定文件
└── README.md             # 项目说明
```

## 环境要求

- Python >= 3.11
- [uv](https://github.com/astral-sh/uv) >= 0.8.0

## 安装

### 安装 uv (如果尚未安装)

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 安装项目依赖

```bash
# 克隆项目后，使用 uv 安装依赖
uv sync

# 包含开发依赖 (pytest, black, ruff)
uv sync --all-groups
```

## 使用

### 运行测试

```bash
# 运行环境测试
uv run python test_env.py
```

### 训练模型

```bash
# 基本训练
uv run python train.py

# 指定参数
uv run python train.py --timesteps 1000000 --width 15 --height 15 --seed 42

# 使用 GPU (如果可用)
uv run python train.py --device cuda
```

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--timesteps` | 1000000 | 总训练步数 |
| `--width` | 10 | 地图宽度 |
| `--height` | 10 | 地图高度 |
| `--lr` | 3e-4 | 学习率 |
| `--seed` | 42 | 随机种子 |
| `--device` | auto | 训练设备 (cpu/cuda/auto) |
| `--save-dir` | ./checkpoints | 模型保存目录 |
| `--log-dir` | ./logs | TensorBoard 日志目录 |

### 查看训练进度

```bash
# 使用 uv 运行 tensorboard
uv run tensorboard --logdir=./logs
```

## 常用 uv 命令

```bash
# 添加新依赖
uv add package-name

# 添加开发依赖
uv add --group dev package-name

# 更新依赖
uv sync --upgrade

# 更新特定包
uv upgrade package-name

# 运行 Python 脚本
uv run python script.py

# 进入虚拟环境 shell
uv run bash

# 锁定依赖版本
uv lock

# 导出 requirements.txt (如需兼容 pip)
uv export --no-dev --format requirements-txt > requirements.txt
```

## 核心概念

### 环境 (CoverWorldEnv)

- **地图生成**: 随机网格增长算法生成不规则地图
- **机器人移动**: 连续 2D 空间移动，带碰撞检测
- **覆盖机制**: 机器人覆盖半径 `COVER_R` (默认 1.8) 内的网格单元
- **终止条件**:
  - 覆盖率达到阈值 (60% 地图)
  - 达到最大回合步数
  - 机器人撞到边界或与地图断开连接

### 观察空间

图结构观察值包含:
- `node_feat`: [N, 5] - 节点特征 (x, y, covered_status, direction_x, direction_y)
- `edge_index`: [2, E] - 边索引
- `edge_attr`: [E, 1] - 边特征 (distance)
- `node_num`: 节点数量
- `edge_num`: 边数量

### 动作空间

- 连续角度值 [0, 2π]，表示移动方向
- 转换为 2D 方向向量 [cos(θ), sin(θ)]

### GNN 模型 (GNNActorCritic)

- **特征提取**: 3层 GNN，使用消息传递机制
- **Actor**: 输出动作的均值和对数标准差
- **Critic**: 估计状态价值函数

### PPO 训练

- **算法**: Proximal Policy Optimization
- **优势估计**: GAE (Generalized Advantage Estimation)
- **更新**: 多批次迭代，梯度裁剪

## 性能优化

- 使用 GPU 加速训练 (如果可用)
- 调整 `n_steps` 和 `batch_size` 优化内存使用
- 增大 `width` 和 `height` 训练更复杂的地图

## GPU 支持 (CUDA)

如需使用 CUDA 版本的 PyTorch，修改 `pyproject.toml` 中的 torch 依赖：

```toml
dependencies = [
    "torch>=2.2.0",
    # ... 其他依赖
]
```

然后在同步时指定 CUDA 版本：

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
uv sync
```

## 开发工具

项目配置了以下开发工具：

- **black**: 代码格式化
- **ruff**: 代码检查
- **pytest**: 测试框架

使用方式：

```bash
# 代码格式化
uv run black .

# 代码检查
uv run ruff check .

# 运行测试
uv run pytest
```

## 常见问题

### uv 安装失败

如果 uv 安装依赖时遇到问题，尝试清除缓存：

```bash
uv clean
uv sync
```

### PyTorch Geometric 安装问题

如果遇到 PyTorch Geometric 相关问题，确保 PyTorch 已正确安装：

```bash
uv run python -c "import torch; print(torch.__version__)"
```

## License

MIT License
