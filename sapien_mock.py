"""
临时的 sapien 模拟模块
用于在没有安装 sapien 的情况下让代码能够运行
注意：这只是一个模拟，不提供真实的物理仿真功能
"""

import numpy as np

class Pose:
    def __init__(self, p=None, q=None):
        self.p = np.array(p) if p is not None else np.array([0.0, 0.0, 0.0])
        self.q = np.array(q) if q is not None else np.array([1.0, 0.0, 0.0, 0.0])
    
    def __mul__(self, other):
        # 简单的位姿组合（不完整实现）
        return Pose(self.p + other.p, self.q)
    
    def inv(self):
        # 简单的位姿逆变换（不完整实现）
        return Pose(-self.p, self.q)
    
    def transform(self, other):
        # 简单的位姿变换（不完整实现）
        return other

class Actor:
    def __init__(self, *args, **kwargs):
        pass

class Articulation:
    def __init__(self, *args, **kwargs):
        pass

class Link:
    def __init__(self, *args, **kwargs):
        pass

class Joint:
    def __init__(self, *args, **kwargs):
        pass

class Viewer:
    def __init__(self, *args, **kwargs):
        pass

class utils:
    Viewer = Viewer

class core:
    Pose = Pose
    Actor = Actor
    Articulation = Articulation
    Link = Link
    Joint = Joint

# 创建模拟的 sapien 模块
class MockSapien:
    core = core()
    utils = utils()

# 将模拟模块添加到 sys.modules
import sys
sys.modules['sapien'] = MockSapien()
sys.modules['sapien.core'] = core
sys.modules['sapien.utils'] = utils

print("已加载 sapien 模拟模块（注意：这只是模拟，不提供真实物理仿真功能）")
