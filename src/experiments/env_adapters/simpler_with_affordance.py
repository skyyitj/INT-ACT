"""
带Affordance功能的SimplerEnv适配器
在原有适配器基础上添加affordance可视化功能
"""

import numpy as np
import cv2
from typing_extensions import override
from PIL import Image

# 导入基础适配器
from .simpler import SimplerAdapter, BridgeSimplerAdapter, BridgeSimplerSpatialVLAAdapter
from ...utils.affordance_utils import add_affordance_to_observation


class SimplerAdapterWithAffordance(SimplerAdapter):
    """带Affordance功能的SimplerAdapter"""

    def __init__(self, config):
        super().__init__(config)

        # Affordance配置
        self.use_affordance = getattr(config.eval_cfg, 'use_affordance', False)
        self.affordance_color = getattr(config.eval_cfg, 'affordance_color', [0, 255, 0])
        self.affordance_thickness = getattr(config.eval_cfg, 'affordance_thickness', 3)
        self.affordance_length = getattr(config.eval_cfg, 'affordance_length', 0.08)
        self.affordance_show_point = getattr(config.eval_cfg, 'affordance_show_point', True)

        # 确保颜色是tuple格式
        if isinstance(self.affordance_color, list):
            self.affordance_color = tuple(self.affordance_color)

        # 统计信息
        self.affordance_stats = {
            'total_frames': 0,
            'affordance_added': 0,
            'affordance_failed': 0
        }

        if self.use_affordance:
            print(f"🎯 SimplerAdapter Affordance已启用:")
            print(f"  颜色 (BGR): {self.affordance_color}")
            print(f"  粗细: {self.affordance_thickness}")
            print(f"  长度: {self.affordance_length}m")
            print(f"  显示位置点: {self.affordance_show_point}")

    def add_affordance_to_image(self, obs, env):
        """为观测添加affordance"""
        if not self.use_affordance:
            return obs

        self.affordance_stats['total_frames'] += 1

        try:
            obs_with_aff = add_affordance_to_observation(
                obs, env,
                arrow_length=self.affordance_length,
                arrow_color=self.affordance_color,
                arrow_thickness=self.affordance_thickness,
                show_point=self.affordance_show_point
            )

            self.affordance_stats['affordance_added'] += 1
            return obs_with_aff

        except Exception as e:
            self.affordance_stats['affordance_failed'] += 1
            if self.affordance_stats['affordance_failed'] <= 3:
                print(f"⚠️ Affordance添加失败: {e}")
            return obs

    def get_affordance_stats(self):
        """获取affordance统计信息"""
        return self.affordance_stats.copy()


class BridgeSimplerAdapterWithAffordance(BridgeSimplerAdapter):
    """带Affordance功能的BridgeSimplerAdapter"""

    def __init__(self, config):
        super().__init__(config)

        # Affordance配置
        self.use_affordance = getattr(config.eval_cfg, 'use_affordance', False)
        self.affordance_color = getattr(config.eval_cfg, 'affordance_color', [0, 255, 0])
        self.affordance_thickness = getattr(config.eval_cfg, 'affordance_thickness', 3)
        self.affordance_length = getattr(config.eval_cfg, 'affordance_length', 0.08)
        self.affordance_show_point = getattr(config.eval_cfg, 'affordance_show_point', True)

        # 确保颜色是tuple格式
        if isinstance(self.affordance_color, list):
            self.affordance_color = tuple(self.affordance_color)

        # 统计信息
        self.affordance_stats = {
            'total_frames': 0,
            'affordance_added': 0,
            'affordance_failed': 0
        }

        if self.use_affordance:
            print(f"🎯 BridgeSimplerAdapter Affordance已启用:")
            print(f"  颜色 (BGR): {self.affordance_color}")
            print(f"  粗细: {self.affordance_thickness}")
            print(f"  长度: {self.affordance_length}m")
            print(f"  显示位置点: {self.affordance_show_point}")

    def add_affordance_to_image(self, obs, env):
        """为观测添加affordance"""
        if not self.use_affordance:
            return obs

        self.affordance_stats['total_frames'] += 1

        try:
            obs_with_aff = add_affordance_to_observation(
                obs, env,
                arrow_length=self.affordance_length,
                arrow_color=self.affordance_color,
                arrow_thickness=self.affordance_thickness,
                show_point=self.affordance_show_point
            )

            self.affordance_stats['affordance_added'] += 1
            return obs_with_aff

        except Exception as e:
            self.affordance_stats['affordance_failed'] += 1
            if self.affordance_stats['affordance_failed'] <= 3:
                print(f"⚠️ Affordance添加失败: {e}")
            return obs

    def get_affordance_stats(self):
        """获取affordance统计信息"""
        return self.affordance_stats.copy()


class BridgeSimplerSpatialVLAAdapterWithAffordance(BridgeSimplerSpatialVLAAdapter):
    """带Affordance功能的BridgeSimplerSpatialVLAAdapter"""

    def __init__(self, config):
        super().__init__(config)

        # Affordance配置
        self.use_affordance = getattr(config.eval_cfg, 'use_affordance', False)
        self.affordance_color = getattr(config.eval_cfg, 'affordance_color', [0, 255, 0])
        self.affordance_thickness = getattr(config.eval_cfg, 'affordance_thickness', 3)
        self.affordance_length = getattr(config.eval_cfg, 'affordance_length', 0.08)
        self.affordance_show_point = getattr(config.eval_cfg, 'affordance_show_point', True)

        # 确保颜色是tuple格式
        if isinstance(self.affordance_color, list):
            self.affordance_color = tuple(self.affordance_color)

        # 统计信息
        self.affordance_stats = {
            'total_frames': 0,
            'affordance_added': 0,
            'affordance_failed': 0
        }

        if self.use_affordance:
            print(f"🎯 BridgeSimplerSpatialVLAAdapter Affordance已启用:")
            print(f"  颜色 (BGR): {self.affordance_color}")
            print(f"  粗细: {self.affordance_thickness}")
            print(f"  长度: {self.affordance_length}m")
            print(f"  显示位置点: {self.affordance_show_point}")

    @override
    def preprocess(self, obs: dict) -> dict:
        """
        预处理观测，在此阶段添加affordance
        """
        # 首先添加affordance（如果启用）
        if self.use_affordance:
            obs = self.add_affordance_to_image(obs, getattr(self, '_env', None))

        # 然后进行正常的预处理
        images = cv2.resize(
            obs['observation.images.top'],
            self.image_size,
            interpolation=cv2.INTER_AREA,
        )

        images = [Image.fromarray(images).convert("RGB")]  # SpatialVLA requires PIL and in a []

        return {
            "observation.images.top": images,
            "task": obs["task"],
        }

    def add_affordance_to_image(self, obs, env):
        """为观测添加affordance"""
        if not self.use_affordance:
            return obs

        self.affordance_stats['total_frames'] += 1

        try:
            obs_with_aff = add_affordance_to_observation(
                obs, env,
                arrow_length=self.affordance_length,
                arrow_color=self.affordance_color,
                arrow_thickness=self.affordance_thickness,
                show_point=self.affordance_show_point
            )

            self.affordance_stats['affordance_added'] += 1
            return obs_with_aff

        except Exception as e:
            self.affordance_stats['affordance_failed'] += 1
            if self.affordance_stats['affordance_failed'] <= 3:
                print(f"⚠️ Affordance添加失败: {e}")
            return obs

    def set_env(self, env):
        """设置环境引用，用于affordance功能"""
        self._env = env
    
    def get_affordance_stats(self):
        """获取affordance统计信息"""
        return self.affordance_stats.copy()
