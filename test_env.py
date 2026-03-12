# -*- coding: utf-8 -*-
"""
文件名称：test_env.py
作者：denny
创建日期：2026-03-12
描述：环境测试脚本
"""

import random
import numpy as np
import torch

from cover_world import CoverWorldEnv


def test_env_basic():
    """测试环境基本功能"""
    print("=" * 50)
    print("Testing CoverWorldEnv Basic Functionality")
    print("=" * 50)
    
    env_config = {
        'width': 10,
        'height': 10,
        'max_episodes_length': 200,
        'seed': 42,
        'render_mode': None
    }
    
    env = CoverWorldEnv(env_config)
    
    # 测试重置
    print("\n1. Testing reset()")
    obs, info = env.reset()
    print(f"   Observation keys: {obs.keys()}")
    print(f"   Node feat shape: {obs['node_feat'].shape}")
    print(f"   Edge index shape: {obs['edge_index'].shape}")
    print(f"   Edge attr shape: {obs['edge_attr'].shape}")
    print(f"   Node num: {obs['node_num']}")
    print(f"   Edge num: {obs['edge_num']}")
    print(f"   Info: {info}")
    
    # 测试动作空间
    print("\n2. Testing action space")
    print(f"   Action space: {env.action_space}")
    sample_action = env.action_space.sample()
    print(f"   Sample action: {sample_action}")
    
    # 测试观察空间
    print("\n3. Testing observation space")
    print(f"   Observation space keys: {env.observation_space.spaces.keys()}")
    
    # 测试单步
    print("\n4. Testing step()")
    action = np.array([random.uniform(0, 2 * np.pi)])
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"   Reward: {reward}")
    print(f"   Terminated: {terminated}")
    print(f"   Info: {info}")
    
    print("\n✓ Basic tests passed!")
    return True


def test_env_episode():
    """测试完整回合"""
    print("\n" + "=" * 50)
    print("Testing Full Episode")
    print("=" * 50)
    
    env = CoverWorldEnv({
        'width': 10,
        'height': 10,
        'max_episodes_length': 50,
        'seed': 42,
        'render_mode': None
    })
    
    obs, info = env.reset()
    total_reward = 0
    steps = 0
    
    for step in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        
        print(f"Step {step}: reward={reward:.2f}, coverage={info['coverage_rate']:.2%}")
        
        if terminated or truncated:
            print(f"Episode finished after {steps} steps")
            print(f"Total reward: {total_reward:.2f}")
            print(f"Final coverage: {info['coverage_rate']:.2%}")
            break
    
    print("\n✓ Episode test passed!")
    return True


def test_env_multiple_episodes():
    """测试多个回合"""
    print("\n" + "=" * 50)
    print("Testing Multiple Episodes")
    print("=" * 50)
    
    env = CoverWorldEnv({
        'width': 10,
        'height': 10,
        'max_episodes_length': 50,
        'seed': 42,
        'render_mode': None
    })
    
    for episode in range(3):
        obs, info = env.reset()
        total_reward = 0
        steps = 0
        
        while steps < 100:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            if terminated or truncated:
                break
        
        print(f"Episode {episode + 1}: steps={steps}, reward={total_reward:.2f}, coverage={info['coverage_rate']:.2%}")
    
    print("\n✓ Multiple episodes test passed!")
    return True


def test_model_integration():
    """测试模型集成"""
    print("\n" + "=" * 50)
    print("Testing Model Integration")
    print("=" * 50)
    
    from models import GNNActorCritic
    
    # 创建环境和模型
    env = CoverWorldEnv({
        'width': 10,
        'height': 10,
        'max_episodes_length': 50,
        'seed': 42,
        'render_mode': None
    })
    
    model = GNNActorCritic(node_dim=5, edge_dim=1, hidden_dim=64, action_dim=2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"Using device: {device}")
    print(f"Model: {model.__class__.__name__}")
    
    # 运行一个回合
    obs, info = env.reset()
    total_reward = 0
    
    for step in range(50):
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
        
        print(f"Step {step}: action=[{action_np[0]:.2f}, {action_np[1]:.2f}], "
              f"angle={angle:.2f}, reward={reward:.2f}, value={value.item():.2f}")
        
        if terminated or truncated:
            print(f"Episode finished after {step + 1} steps")
            break
    
    print(f"Total reward: {total_reward:.2f}")
    print("\n✓ Model integration test passed!")
    return True


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 70)
    print("Running All Tests")
    print("=" * 70)
    
    tests = [
        ("Basic Functionality", test_env_basic),
        ("Full Episode", test_env_episode),
        ("Multiple Episodes", test_env_multiple_episodes),
        ("Model Integration", test_model_integration),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"\n✗ Test '{name}' failed with error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 70)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 70)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
