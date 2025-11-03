#!/usr/bin/env python3
"""
测试观察数据结构
"""
import os
import sys
from pathlib import Path

# 设置路径
sys.path.insert(0, str(Path(__file__).parent.absolute()))

def test_env_observation():
    """测试环境观察数据格式"""
    print("🧪 测试SimplerEnv观察数据格式")
    print("=" * 60)

    try:
        import simpler_env

        # 创建环境
        env = simpler_env.make("widowx_spoon_on_towel")

        # 重置并获取观察
        obs, reset_info = env.reset()

        print(f"\n✅ 环境创建成功")
        print(f"\n📊 观察数据结构:")
        print(f"  类型: {type(obs)}")

        if isinstance(obs, dict):
            print(f"  顶层键: {list(obs.keys())}")

            # 检查 agent 键
            if 'agent' in obs:
                print(f"  \n  obs['agent'] 类型: {type(obs['agent'])}")
                if isinstance(obs['agent'], dict):
                    print(f"  obs['agent'] 键: {list(obs['agent'].keys())}")

                    # 检查 eef_pos
                    if 'eef_pos' in obs['agent']:
                        print(f"  ✅ obs['agent']['eef_pos'] 存在")
                        print(f"     形状: {obs['agent']['eef_pos'].shape if hasattr(obs['agent']['eef_pos'], 'shape') else 'N/A'}")
                    else:
                        print(f"  ❌ obs['agent'] 中没有 'eef_pos' 键")
                else:
                    print(f"  ❌ obs['agent'] 不是字典")
            else:
                print(f"  ❌ obs 中没有 'agent' 键")
        else:
            print(f"  ❌ obs 不是字典类型")

        env.close()
        print(f"\n✅ 测试完成")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_env_observation()
