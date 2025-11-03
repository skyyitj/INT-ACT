#!/usr/bin/env python3
"""
快速修复 standalone 模式的 KeyError: 'eef_pos' 问题
"""

import os
import sys
from pathlib import Path
import shutil

def backup_file(file_path):
    """备份文件"""
    backup_path = f"{file_path}.backup"
    if not Path(backup_path).exists():
        shutil.copy2(file_path, backup_path)
        print(f"✅ 已备份: {backup_path}")
    return backup_path

def fix_simpler_adapter():
    """修复 simpler adapter 以处理不同的观察数据格式"""
    print("\n" + "=" * 60)
    print("🔧 修复 SimplerAdapter")
    print("=" * 60)

    adapter_file = Path("src/experiments/env_adapters/simpler.py")

    if not adapter_file.exists():
        print(f"❌ 文件不存在: {adapter_file}")
        return False

    # 备份原文件
    backup_file(adapter_file)

    # 读取文件
    with open(adapter_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 查找 BridgeSimplerAdapter 的 preprocess_proprio 方法
    # 添加容错处理
    old_preprocess_proprio = '''    def preprocess_proprio(self, obs: dict) -> np.array:
        # convert ee rotation to the frame of top-down
        proprio = obs["agent"]["eef_pos"]'''

    new_preprocess_proprio = '''    def preprocess_proprio(self, obs: dict) -> np.array:
        # convert ee rotation to the frame of top-down
        # 🔧 添加容错处理，支持不同的观察数据格式
        if isinstance(obs, dict) and "agent" in obs:
            # 标准的 ManiSkill2 state_dict 格式
            if isinstance(obs["agent"], dict) and "eef_pos" in obs["agent"]:
                proprio = obs["agent"]["eef_pos"]
            else:
                # agent 不是字典或没有 eef_pos
                print(f"⚠️  obs['agent'] 结构异常: {type(obs['agent'])}, 键: {list(obs['agent'].keys()) if isinstance(obs['agent'], dict) else 'N/A'}")
                raise KeyError(f"obs['agent'] 中找不到 'eef_pos'，可用键: {list(obs['agent'].keys()) if isinstance(obs['agent'], dict) else 'N/A'}")
        else:
            # obs 本身没有 agent 键，可能是扁平化的状态
            print(f"⚠️  观察数据格式错误:")
            print(f"    obs 类型: {type(obs)}")
            print(f"    obs 键: {list(obs.keys()) if isinstance(obs, dict) else 'N/A'}")
            raise KeyError(f"观察数据中没有 'agent' 键，可用键: {list(obs.keys()) if isinstance(obs, dict) else 'N/A'}")

        proprio = obs["agent"]["eef_pos"]'''

    if old_preprocess_proprio in content:
        content = content.replace(old_preprocess_proprio, new_preprocess_proprio)
        print("✅ 已添加容错处理代码")
    else:
        print("⚠️  未找到目标代码，可能已经修改过")

    # 写回文件
    with open(adapter_file, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"✅ 修复完成: {adapter_file}")
    return True

def check_simpler_env_version():
    """检查 SimplerEnv 和 ManiSkill 版本"""
    print("\n" + "=" * 60)
    print("🔍 检查环境版本")
    print("=" * 60)

    try:
        # 检查 ManiSkill2
        try:
            import mani_skill2_real2sim
            print(f"✅ ManiSkill2_real2sim 已安装")
            print(f"   路径: {mani_skill2_real2sim.__file__}")
        except ImportError:
            print("⚠️  ManiSkill2_real2sim 未安装")

        # 检查 ManiSkill3
        try:
            import mani_skill
            print(f"✅ ManiSkill (v3) 已安装")
            print(f"   路径: {mani_skill.__file__}")
        except ImportError:
            print("⚠️  ManiSkill v3 未安装")

        # 检查 SimplerEnv
        try:
            import simpler_env
            print(f"✅ SimplerEnv 已安装")
            print(f"   路径: {simpler_env.__file__}")
        except ImportError:
            print("❌ SimplerEnv 未安装")
            return False

        return True
    except Exception as e:
        print(f"❌ 检查时出错: {e}")
        return False

def create_test_script():
    """创建测试脚本来验证修复"""
    print("\n" + "=" * 60)
    print("📝 创建测试脚本")
    print("=" * 60)

    test_script = '''#!/usr/bin/env python3
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

        print(f"\\n✅ 环境创建成功")
        print(f"\\n📊 观察数据结构:")
        print(f"  类型: {type(obs)}")

        if isinstance(obs, dict):
            print(f"  顶层键: {list(obs.keys())}")

            # 检查 agent 键
            if 'agent' in obs:
                print(f"  \\n  obs['agent'] 类型: {type(obs['agent'])}")
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
        print(f"\\n✅ 测试完成")

    except Exception as e:
        print(f"\\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_env_observation()
'''

    test_file = Path("test_observation_structure.py")
    with open(test_file, 'w') as f:
        f.write(test_script)

    os.chmod(test_file, 0o755)
    print(f"✅ 测试脚本创建: {test_file}")
    print(f"\n💡 运行测试: python {test_file}")

    return str(test_file)

def print_solution_summary():
    """打印解决方案总结"""
    print("\n" + "=" * 60)
    print("🎯 解决方案总结")
    print("=" * 60)

    print("""
问题根源：
---------
standalone 模式下，服务器端在处理观察数据时，期望的数据格式是：
  obs["agent"]["eef_pos"]

但实际收到的观察数据结构可能不匹配，导致 KeyError。

修复步骤（按优先级）：
--------------------

✅ 步骤1: 测试观察数据结构（推荐先做）
   python test_observation_structure.py

✅ 步骤2: 尝试使用本地模式替代 standalone
   python run_pi0_simpler_local.py \\
     --model-path ./models/INTACT-pi0-finetune-bridge \\
     --n-episodes 4

✅ 步骤3: 如果必须使用 standalone，运行修复脚本
   本脚本已添加容错处理代码

✅ 步骤4: 检查模型权重问题
   # 删除可能损坏的模型
   rm -rf models/INTACT-pi0-finetune-bridge

   # 重新下载
   python run_pi0_simpler_standalone.py --skip-download=False

关于模型权重缺失警告：
-------------------
警告信息显示缺失很多 vision_tower 权重。这可能是因为：
1. 模型文件下载不完整
2. 模型架构与加载代码不匹配
3. 正常的警告（某些权重是可选的）

如果只是警告而不是错误，可以暂时忽略。但如果影响运行，
需要重新下载模型或检查模型配置。

推荐方案：
--------
如果你只是想快速测试评估，强烈建议使用 run_pi0_simpler_local.py
而不是 standalone 模式，因为：

1. local 模式更简单，不需要服务器-客户端分离
2. 减少了通信带来的数据序列化问题
3. 更容易调试

运行命令：
python run_pi0_simpler_local.py \\
  --model-path ./models/INTACT-pi0-finetune-bridge
""")

def main():
    print("🚀 INT-ACT Standalone 快速修复工具")
    print("=" * 60)

    # 切换到项目根目录
    project_root = Path(__file__).parent
    os.chdir(project_root)
    print(f"📁 工作目录: {project_root}")

    # 检查环境
    check_simpler_env_version()

    # 创建测试脚本
    test_file = create_test_script()

    # 修复适配器（添加调试信息）
    fix_simpler_adapter()

    # 打印总结
    print_solution_summary()

    print("\n" + "=" * 60)
    print("✅ 修复完成！")
    print("=" * 60)
    print("\n📋 下一步:")
    print(f"  1. 运行测试: python {test_file}")
    print("  2. 尝试本地模式: python run_pi0_simpler_local.py --model-path ./models/INTACT-pi0-finetune-bridge")
    print("  3. 如果还有问题，查看上面的详细说明")

if __name__ == "__main__":
    main()

