#!/usr/bin/env python3
"""
Affordance环境包装器

用于在策略训练/测试中自动为观测添加affordance信息
"""

import numpy as np
from get_pose_corrected_coordinates import add_affordance_to_observation


class AffordanceWrapper:
    """
    环境包装器 - 自动为观测添加affordance

    用法:
        env = simpler_env.make("widowx_carrot_on_plate")
        env = AffordanceWrapper(env, use_affordance=True)

        obs, info = env.reset()
        # obs 中的图像已经包含了affordance箭头
    """

    def __init__(self, env, use_affordance=True,
                 arrow_length=0.08,
                 arrow_color=(0, 255, 0),
                 arrow_thickness=3,
                 show_point=True,
                 verbose=False):
        """
        Args:
            env: 基础环境
            use_affordance: 是否启用affordance（用于对照实验）
            arrow_length: 箭头长度（米）
            arrow_color: 箭头颜色 (B, G, R)
            arrow_thickness: 箭头粗细
            show_point: 是否显示夹爪位置点
            verbose: 是否打印详细信息
        """
        self.env = env
        self.use_affordance = use_affordance
        self.arrow_length = arrow_length
        self.arrow_color = arrow_color
        self.arrow_thickness = arrow_thickness
        self.show_point = show_point
        self.verbose = verbose

        # 统计信息
        self.total_steps = 0
        self.affordance_added_count = 0
        self.affordance_failed_count = 0

    def reset(self, **kwargs):
        """重置环境并添加affordance"""
        obs, info = self.env.reset(**kwargs)

        if self.use_affordance:
            try:
                obs = add_affordance_to_observation(
                    obs, self.env,
                    arrow_length=self.arrow_length,
                    arrow_color=self.arrow_color,
                    arrow_thickness=self.arrow_thickness,
                    show_point=self.show_point
                )
                self.affordance_added_count += 1
                if self.verbose:
                    print("✓ Affordance已添加到观测")
            except Exception as e:
                self.affordance_failed_count += 1
                if self.verbose:
                    print(f"⚠ Affordance添加失败: {e}")

        return obs, info

    def step(self, action):
        """执行动作并为新观测添加affordance"""
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.total_steps += 1

        if self.use_affordance:
            try:
                obs = add_affordance_to_observation(
                    obs, self.env,
                    arrow_length=self.arrow_length,
                    arrow_color=self.arrow_color,
                    arrow_thickness=self.arrow_thickness,
                    show_point=self.show_point
                )
                self.affordance_added_count += 1
            except Exception as e:
                self.affordance_failed_count += 1
                if self.verbose:
                    print(f"⚠ Affordance添加失败 (步骤 {self.total_steps}): {e}")

        return obs, reward, terminated, truncated, info

    def get_stats(self):
        """获取统计信息"""
        return {
            'total_steps': self.total_steps,
            'affordance_added': self.affordance_added_count,
            'affordance_failed': self.affordance_failed_count,
            'success_rate': self.affordance_added_count / max(1, self.total_steps + 1)
        }

    def print_stats(self):
        """打印统计信息"""
        stats = self.get_stats()
        print("\n=== Affordance统计信息 ===")
        print(f"总步数: {stats['total_steps']}")
        print(f"成功添加: {stats['affordance_added']}")
        print(f"添加失败: {stats['affordance_failed']}")
        print(f"成功率: {stats['success_rate']*100:.1f}%")

    def close(self):
        """关闭环境"""
        if self.verbose and self.use_affordance:
            self.print_stats()
        return self.env.close()

    def __getattr__(self, name):
        """代理其他属性到基础环境"""
        return getattr(self.env, name)


def test_wrapper():
    """测试AffordanceWrapper"""
    import simpler_env

    print("=== 测试AffordanceWrapper ===\n")

    task_name = "widowx_carrot_on_plate"
    base_env = simpler_env.make(task_name)

    # 创建带affordance的环境
    env = AffordanceWrapper(
        base_env,
        use_affordance=True,
        arrow_color=(0, 255, 0),
        arrow_thickness=3,
        show_point=True,
        verbose=True
    )

    try:
        print("1. 重置环境...")
        obs, info = env.reset(seed=42)
        print("   ✓ 环境已重置\n")

        print("2. 执行5个步骤...")
        for i in range(5):
            action = env.action_space.sample() * 0.05  # 小动作
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"   步骤 {i+1}: 奖励={reward:.3f}")

            if terminated or truncated:
                print("   任务终止，重置环境")
                obs, info = env.reset(seed=42)

        print("\n3. 统计信息:")
        env.print_stats()

        print("\n✅ 测试完成！")

    finally:
        env.close()


if __name__ == "__main__":
    test_wrapper()

