# MultiRob-Coverage - AI Agent Guide

## Project Overview

MultiRob-Coverage is a simulation environment for multi-robot coverage tasks using Reinforcement Learning (RL) with Graph Neural Networks (GNN). The project implements a coverage scenario where a robot/agent navigates a grid-based map to cover as much area as possible.

**Key Characteristics:**
- **Domain**: Reinforcement Learning + Graph Neural Networks
- **Task**: Multi-robot area coverage in a grid-based environment
- **Language**: Python (with Chinese comments in source code)
- **Project Size**: Small (~800 lines of code)
- **Configuration**: No formal dependency management (no requirements.txt, pyproject.toml, etc.)

## Technology Stack

### Core Dependencies
- **TensorFlow/Keras**: Deep learning framework for model implementation
- **Ray RLlib**: Distributed RL training library (uses PPO algorithm)
- **Gymnasium**: OpenAI Gym-compatible environment interface
- **NetworkX**: Graph data structure and algorithms
- **DGL (Deep Graph Library)**: Graph neural network operations
- **KGCNN**: Keras-based graph convolutional neural network layers
- **graph_nets**: DeepMind's library for building graph networks

### Supporting Libraries
- **NumPy**: Numerical computations
- **Pygame**: Rendering/visualization
- **Matplotlib**: Graph visualization and plotting

## Project Structure

```
MultiRob-Coverage/
├── CoverWorld.py      # Environment implementation (~608 lines)
├── actor.py           # RL agent/model definition (~200 lines)
├── README.md          # Brief project description
└── AGENTS.md          # This file
```

### File Descriptions

#### `CoverWorld.py`
Main environment file containing:

1. **`GridGraph` class**: Core environment logic
   - Generates random grid-based maps using a growth algorithm
   - Manages robot position and movement
   - Tracks coverage status of grid cells
   - Provides graph-based observations for RL agents
   - Key constants:
     - `COVER_R = 1.8`: Coverage radius
     - `STEP = 1.42`: Robot step size
     - `CONSTRUCT_DIS = 6`: Subgraph construction distance
     - `MAX_EPISODE_LENGTH = 200`: Maximum steps per episode

2. **`CoverWorldEnv` class**: Gymnasium environment wrapper
   - Implements standard Gymnasium interface (`reset()`, `step()`, `render()`)
   - Defines observation space (node features, edge features, counts, direction)
   - Action space: Single continuous value `[0, 2π]` representing movement direction
   - Configuration via `ENV_CONFIG` dict:
     - `width`, `height`: Map dimensions
     - `render_mode`: "human" or "rgb_array"
     - `max_episodes_length`: Episode termination limit

#### `actor.py`
RL agent implementation containing:

1. **`Actor` class**: Custom TFModelV2 for Ray RLlib
   - Graph Neural Network architecture using KGCNN layers
   - Processes variable-size graph observations
   - Outputs action distribution (mean/std) and value estimate
   - Architecture:
     - Input: Node attributes, edge weights, edge indices, last direction
     - GNN layers with message passing
     - Graph pooling to get global embedding
     - Output heads for action and value

## Key Concepts

### Environment Dynamics
1. **Map Generation**: Random grid growth from center, creates irregular maps
2. **Robot Movement**: Continuous 2D movement with collision detection
3. **Coverage Mechanism**: Robot covers cells within `COVER_R` radius
4. **Termination Conditions**:
   - Coverage threshold reached (60% of map)
   - Maximum episode steps exceeded
   - Robot hits boundary or disconnects from map

### Observation Space
The environment returns graph-structured observations:
- `node_feat`: `[N, 5]` array (x, y, covered_status, direction_x, direction_y)
- `edge_feat`: `[E, 3]` array (source_id, dest_id, distance)
- `node_num`: Scalar, number of nodes
- `edge_num`: Scalar, number of edges  
- `last_direct`: `[2]` array, previous movement direction

### Action Space
- Single continuous value in `[0, 2π]` representing movement angle
- Converted to 2D direction vector `[cos(θ), sin(θ)]`

## Build and Run Instructions

### Prerequisites
Install dependencies manually (no requirements.txt provided):

```bash
pip install tensorflow gymnasium networkx matplotlib numpy pygame ray[rllib] dgl kgcnn graph-nets
```

### Running the Environment
Test the environment standalone:

```bash
python CoverWorld.py
```

This runs `cover_env_test()` which creates an environment and runs 3 episodes.

### Training with RLlib
To train using Ray RLlib, you would need to create a training script:

```python
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from actor import Actor
from CoverWorld import CoverWorldEnv

# Register custom model
ModelCatalog.register_custom_model("actor", Actor)

# Configure and train
config = (
    PPOConfig()
    .environment(CoverWorldEnv, env_config={"width": 10, "height": 10, "render_mode": None, "max_episodes_length": 200})
    .framework("tf2")
    .training(model={"custom_model": "actor"})
)

algo = config.build()
# ... training loop
```

## Code Style Guidelines

### Language
- Source code comments are primarily in **Chinese**
- Variable names use English (snake_case)
- Class names use PascalCase

### Naming Conventions
- Private methods: `_method_name` (single underscore prefix)
- Constants: `UPPER_CASE_WITH_UNDERSCORES`
- Instance variables: `snake_case`

### Code Organization
- Each file has a header comment with filename, author, creation date, and description
- Methods grouped by functionality (movement, coverage, graph operations)

## Development Notes

### Important Constants
Located at top of `CoverWorld.py`:
- `COVER_R`: Coverage radius for the robot
- `STEP`: Movement step size
- `MAX_EPISODE_LENGTH`: Episode length limit
- `COVER_MODERATE = 1.0`: Coverage completion threshold (100%)
- `CONSTRUCT_DIS = 6`: Distance threshold for local graph construction

### Debug Output
The code contains extensive print statements for debugging:
- Action values in `step()`
- Shape information in model `forward()`
- Node/edge counts

### GPU Configuration
Commented out GPU memory limiting code exists in both files:
```python
# for gpu in tf.config.experimental.list_physical_devices("GPU"):
#     tf.config.experimental.set_virtual_device_configuration(gpu,
#     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
```

## Testing

No formal test suite exists. Manual testing via:
- `cover_env_test()` function in `CoverWorld.py`
- Run `python CoverWorld.py` for basic environment test
- Run `python actor.py` (currently does nothing - empty main block)

## Limitations and TODOs

1. **No dependency management**: No requirements.txt or pyproject.toml
2. **Single agent only**: Despite "MultiRob" name, current implementation supports single robot
3. **No trained model checkpoint**: Actor implementation provided but no training script
4. **Hardcoded constants**: Many hyperparameters defined as module-level constants
5. **Debug output**: Extensive print statements should be removed or converted to logging

## Git History

Recent commits:
- `3ddb11a`: Add files via upload
- `93793a9`: Add files via upload  
- `776e527`: Initial commit

## External References

The project appears to be related to research in:
- Multi-robot coverage path planning
- Graph Neural Networks for RL
- GNN-based reinforcement learning for navigation tasks
