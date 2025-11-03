#!/usr/bin/env python3
"""
简化的pi0测试脚本 - 用于快速检查环境
"""

import os
import sys
from pathlib import Path

def setup_paths():
    """设置Python路径"""
    project_root = Path(__file__).parent.absolute()
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(project_root / 'third_party/lerobot'))

    # 设置环境变量
    os.environ['PYTHONPATH'] = str(project_root)
    print(f"✅ 项目根目录: {project_root}")

def test_imports():
    """测试关键模块导入"""
    print("\n🔍 测试模块导入...")

    # 首先测试基础包
    basic_imports_ok = True

    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"❌ PyTorch 导入失败: {e}")
        basic_imports_ok = False

    try:
        import torchvision
        print(f"✅ torchvision {torchvision.__version__}")
    except ImportError as e:
        print(f"❌ torchvision 导入失败: {e}")
        print("💡 提示: 可能是PyTorch和torchvision版本不兼容")
        print("   运行: python fix_torch_version.py")
        basic_imports_ok = False
    except RuntimeError as e:
        print(f"❌ torchvision 运行时错误: {e}")
        print("💡 提示: PyTorch和torchvision版本不兼容")
        print("   运行: python fix_torch_version.py")
        basic_imports_ok = False

    try:
        import transformers
        print(f"✅ Transformers {transformers.__version__}")
    except ImportError as e:
        print(f"❌ Transformers 导入失败: {e}")
        basic_imports_ok = False

    try:
        import numpy as np
        print(f"✅ NumPy {np.__version__}")
    except ImportError as e:
        print(f"❌ NumPy 导入失败: {e}")
        basic_imports_ok = False

    if not basic_imports_ok:
        print("\n⚠️ 基础包导入失败，跳过PI0模块测试")
        return False

    # 测试PI0相关模块
    try:
        from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
        from lerobot.common.policies.pi0.configuration_pi0 import PI0Config
        print("✅ PI0 模块导入成功")
        return True

    except ImportError as e:
        print(f"❌ PI0模块导入失败: {e}")
        return False
    except Exception as e:
        print(f"❌ PI0模块导入出错: {e}")
        return False

def test_pi0_creation():
    """测试pi0模型创建"""
    print("\n🔧 测试PI0模型创建...")

    try:
        from lerobot.common.policies.pi0.configuration_pi0 import PI0Config
        from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy

        # 创建配置
        config = PI0Config()
        print(f"✅ PI0配置创建成功: pi0")
        print(f"   - chunk_size: {config.chunk_size}")
        print(f"   - n_action_steps: {config.n_action_steps}")

        # 创建策略（不加载权重）
        policy = PI0Policy(config)
        print("✅ PI0策略创建成功")

        return True

    except Exception as e:
        print(f"❌ PI0创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_simpler_env():
    """检查SimplerEnv相关文件"""
    print("\n🌍 检查SimplerEnv环境...")

    project_root = Path(__file__).parent
    simpler_path = project_root / 'third_party/SimplerEnv'

    if simpler_path.exists():
        print(f"✅ SimplerEnv路径存在: {simpler_path}")
        return True
    else:
        print(f"❌ SimplerEnv路径不存在: {simpler_path}")
        return False

def main():
    print("🤖 INT-ACT PI0 简化测试")
    print("=" * 50)

    # 设置路径
    setup_paths()

    # 测试导入
    import_ok = test_imports()

    # 测试pi0创建
    pi0_ok = test_pi0_creation() if import_ok else False

    # 检查SimplerEnv
    simpler_ok = check_simpler_env()

    print("\n" + "=" * 50)
    print("📊 测试结果:")
    print(f"  模块导入: {'✅' if import_ok else '❌'}")
    print(f"  PI0模型: {'✅' if pi0_ok else '❌'}")
    print(f"  SimplerEnv: {'✅' if simpler_ok else '❌'}")

    if import_ok and pi0_ok:
        print("\n🎉 基础环境测试通过！可以进行下一步。")
        print("\n📝 下一步建议:")
        print("1. 下载预训练模型")
        print("2. 安装SimplerEnv依赖")
        print("3. 运行完整评估")
    else:
        print("\n⚠️ 需要先解决环境问题")

if __name__ == "__main__":
    main()
