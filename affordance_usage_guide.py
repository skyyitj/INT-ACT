#!/usr/bin/env python3
"""
Affordance使用指南和示例代码

这个脚本展示如何在实际的机器人策略训练/测试中使用affordance功能
"""

import numpy as np
import simpler_env
from get_pose_corrected_coordinates import add_affordance_to_observation


def example_1_basic_usage():
    """示例1: 基础使用 - 在环境循环中添加affordance"""
    print("=== 示例1: 基础使用 ===\n")

    task_name = "widowx_carrot_on_plate"
    env = simpler_env.make(task_name)

    try:
        obs, reset_info = env.reset(seed=42)

        for step in range(5):
            print(f"步骤 {step+1}:")

            # 添加affordance到观测
            obs_with_affordance = add_affordance_to_observation(obs, env)

            # 此时 obs_with_affordance 中的图像已经包含了affordance箭头
            # 可以直接传给策略网络

            # 这里使用随机动作作为示例
            action = env.action_space.sample() * 0.1  # 使用小动作

            # 执行动作
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"  奖励: {reward}")

            if terminated or truncated:
                print("  任务终止")
                break

        print("\n✅ 示例1完成")

    finally:
        env.close()


def example_2_custom_affordance():
    """示例2: 自定义affordance样式"""
    print("\n=== 示例2: 自定义Affordance样式 ===\n")

    task_name = "widowx_carrot_on_plate"
    env = simpler_env.make(task_name)

    try:
        obs, reset_info = env.reset(seed=42)

        # 测试不同的affordance样式
        styles = [
            {
                'name': '绿色中等箭头',
                'arrow_color': (0, 255, 0),
                'arrow_thickness': 3,
                'show_point': True
            },
            {
                'name': '红色粗箭头',
                'arrow_color': (0, 0, 255),
                'arrow_thickness': 5,
                'show_point': True
            },
            {
                'name': '蓝色细箭头（无点）',
                'arrow_color': (255, 0, 0),
                'arrow_thickness': 2,
                'show_point': False
            }
        ]

        for style in styles:
            print(f"测试: {style['name']}")

            obs_with_aff = add_affordance_to_observation(
                obs, env,
                arrow_color=style['arrow_color'],
                arrow_thickness=style['arrow_thickness'],
                show_point=style['show_point']
            )

            print(f"  ✓ Affordance已添加")

        print("\n✅ 示例2完成")
        print("💡 建议: 选择一种affordance样式并在整个训练过程中保持一致")

    finally:
        env.close()


def example_3_policy_integration():
    """示例3: 与策略集成的伪代码示例"""
    print("\n=== 示例3: 与策略集成 ===\n")

    print("这是一个伪代码示例，展示如何集成affordance到策略训练/测试中:")
    print("""
class AffordanceWrapper:
    '''环境包装器 - 自动为观测添加affordance'''

    def __init__(self, env, use_affordance=True, **affordance_kwargs):
        self.env = env
        self.use_affordance = use_affordance
        self.affordance_kwargs = affordance_kwargs

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        if self.use_affordance:
            obs = add_affordance_to_observation(obs, self.env, **self.affordance_kwargs)

        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if self.use_affordance:
            obs = add_affordance_to_observation(obs, self.env, **self.affordance_kwargs)

        return obs, reward, terminated, truncated, info

    def __getattr__(self, name):
        return getattr(self.env, name)


# 使用示例:
def train_with_affordance():
    base_env = simpler_env.make("widowx_carrot_on_plate")

    # 对照组: 不使用affordance
    env_baseline = AffordanceWrapper(base_env, use_affordance=False)

    # 实验组: 使用affordance
    env_with_affordance = AffordanceWrapper(
        base_env,
        use_affordance=True,
        arrow_color=(0, 255, 0),
        arrow_thickness=3,
        show_point=True
    )

    # 训练策略...
    # policy.train(env_with_affordance)

    # 对比性能...
    """)

    print("\n✅ 示例3完成")
    print("💡 提示: 你可以基于这个模板创建自己的环境包装器")


def example_4_ablation_study():
    """示例4: 消融实验设计"""
    print("\n=== 示例4: 消融实验设计 ===\n")

    print("建议的消融实验设置:")
    print("""
实验组：
1. Baseline（无affordance）
   - 使用原始观测图像

2. Affordance-Position（只显示位置点）
   - arrow_thickness=0 或只画圆点

3. Affordance-Direction（完整的朝向箭头）
   - arrow_thickness=3, show_point=True

4. Affordance-Thick（更粗的箭头）
   - arrow_thickness=5, show_point=True

5. Affordance-Thin（更细的箭头）
   - arrow_thickness=2, show_point=False

评估指标：
- 成功率
- 收敛速度
- 样本效率
- 泛化能力

建议训练配置：
- 每个配置使用相同的随机种子
- 运行多次取平均
- 记录详细的训练日志
    """)

    print("\n✅ 示例4完成")


def main():
    """运行所有示例"""
    print("=" * 60)
    print("Affordance使用指南和示例代码")
    print("=" * 60)

    print("\n选择要运行的示例:")
    print("1. 基础使用")
    print("2. 自定义样式")
    print("3. 策略集成（伪代码）")
    print("4. 消融实验设计")
    print("5. 运行所有示例")

    try:
        choice = input("\n请选择 (1-5，默认为5): ").strip()
        if not choice:
            choice = "5"
    except:
        choice = "5"

    if choice == "1":
        example_1_basic_usage()
    elif choice == "2":
        example_2_custom_affordance()
    elif choice == "3":
        example_3_policy_integration()
    elif choice == "4":
        example_4_ablation_study()
    else:
        example_1_basic_usage()
        example_2_custom_affordance()
        example_3_policy_integration()
        example_4_ablation_study()

    print("\n" + "=" * 60)
    print("所有示例完成！")
    print("=" * 60)

    print("\n下一步:")
    print("1. 运行 'python get_pose_corrected_coordinates.py --affordance' 测试可视化效果")
    print("2. 运行 'python get_pose_corrected_coordinates.py --affordance-actions' 测试动作序列")
    print("3. 在你的策略代码中集成 add_affordance_to_observation() 函数")
    print("4. 开始对比实验，评估affordance对性能的影响")


if __name__ == "__main__":
    main()

