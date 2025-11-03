#!/usr/bin/env python3
"""
本地conda环境下的pi0 SimplerEnv评估脚本
适配了conda环境，不依赖SLURM和Singularity
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def setup_environment():
    """设置环境变量和Python路径"""
    project_root = Path(__file__).parent.absolute()

    # 设置Python路径
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(project_root / 'third_party/lerobot'))

    # 设置环境变量
    env_vars = {
        'VLA_DATA_DIR': str(project_root / 'data'),
        'VLA_LOG_DIR': str(project_root / 'log'),
        'VLA_WANDB_ENTITY': 'your_wandb_entity',
        'TRANSFORMERS_CACHE': str(Path.home() / '.cache/huggingface/transformers'),
        'HF_HOME': str(Path.home() / '.cache/huggingface'),
        'MS2_REAL2SIM_ASSET_DIR': str(project_root / 'third_party/ManiSkill2_real2sim/data'),
        'MS_ASSET_DIR': str(project_root / 'third_party/ManiSkill/data'),
        'XLA_PYTHON_CLIENT_PREALLOCATE': 'false',
        'PYTHONPATH': str(project_root),
        # 单机模式下的分布式训练环境变量
        'LOCAL_RANK': '0',
        'RANK': '0',
        'WORLD_SIZE': '1',
        'MASTER_ADDR': 'localhost',
        'MASTER_PORT': '29500'
    }

    for key, value in env_vars.items():
        os.environ[key] = value

    # 创建必要的目录
    for dir_path in [env_vars['VLA_DATA_DIR'], env_vars['VLA_LOG_DIR']]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

    print(f"✅ 环境设置完成，项目根目录: {project_root}")

def download_pi0_model(model_name="juexzz/INTACT-pi0-finetune-bridge"):
    """下载pi0预训练模型"""
    print(f"\n📥 下载pi0模型: {model_name}")

    try:
        from huggingface_hub import snapshot_download

        # 下载模型到本地目录
        model_dir = Path("./models") / model_name.split("/")[-1]
        model_dir.mkdir(parents=True, exist_ok=True)

        print(f"正在下载模型到: {model_dir}")
        snapshot_download(
            repo_id=model_name,
            local_dir=str(model_dir),
            local_dir_use_symlinks=False
        )

        print(f"✅ 模型下载完成: {model_dir}")
        return str(model_dir)

    except Exception as e:
        print(f"❌ 模型下载失败: {e}")
        print("💡 请手动下载模型或检查网络连接")
        return None

def create_local_config(model_path=None):
    """创建本地评估配置文件"""
    print("\n📝 创建本地评估配置...")

    # 使用绝对路径
    actual_model_path = model_path if model_path else "./models/INTACT-pi0-finetune-bridge"

    config_content = f"""name: pi0_local_test
seed: 42
model_cfg: !include ../../models/hf_pi0.json

eval_cfg:
  simulator_name: "simpler"
  env_adapter: "BridgeSimplerAdapter"
  task_list: [
    # 基础测试任务（减少任务数量以便快速测试）
    "widowx_spoon_on_towel",
    "widowx_carrot_on_plate",
    "widowx_stack_cube",
    "widowx_put_eggplant_in_basket",
  ]

  n_eval_episode: 4  # 减少测试轮数
  n_video: 2  # 减少视频录制数量
  recording: True
  pretrained_model_path: {actual_model_path}
  role: "client"
  host: "0.0.0.0"
  port: 5000

env:
  dataset_statistics_path: ./config/dataset/bridge_statistics.json

wandb:
  project: "vla_benchmark_local"
"""

    config_path = Path("config/experiment/simpler/pi0_local_test.yaml")
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, 'w') as f:
        f.write(config_content)

    print(f"✅ 配置文件创建: {config_path}")
    return str(config_path)

def run_simple_evaluation(config_path, seed=42):
    """运行简化的pi0评估"""
    print(f"\n🚀 开始pi0评估 (配置: {config_path}, 种子: {seed})")

    try:
        # 构建评估命令
        cmd = [
            "python", "src/agent/run.py",
            "--config_path", config_path,
            "--seed", str(seed),
            "--use_bf16", "False",
            "--use_wandb", "False"  # 本地测试不使用wandb
        ]

        print(f"执行命令: {' '.join(cmd)}")

        # 运行评估
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1小时超时

        if result.returncode == 0:
            print("✅ 评估完成成功！")
            print("\n📊 输出:")
            print(result.stdout[-1000:])  # 显示最后1000个字符
        else:
            print("❌ 评估失败")
            print(f"错误输出: {result.stderr}")

        return result.returncode == 0

    except subprocess.TimeoutExpired:
        print("⏰ 评估超时")
        return False
    except Exception as e:
        print(f"❌ 评估出错: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="本地pi0 SimplerEnv评估")
    parser.add_argument("--model-name", default="juexzz/INTACT-pi0-finetune-bridge",
                       help="HuggingFace模型名称")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--skip-download", action="store_true", help="跳过模型下载")
    parser.add_argument("--config-only", action="store_true", help="只创建配置文件")

    args = parser.parse_args()

    print("🤖 INT-ACT PI0 SimplerEnv 本地评估")
    print("=" * 60)

    # 1. 设置环境
    setup_environment()

    # 2. 下载模型（如果需要）
    model_path = None
    if not args.skip_download:
        model_path = download_pi0_model(args.model_name)

    # 3. 创建配置
    config_path = create_local_config(model_path)

    if args.config_only:
        print("✅ 配置文件创建完成，退出")
        return

    # 4. 运行评估
    success = run_simple_evaluation(config_path, args.seed)

    print("\n" + "=" * 60)
    if success:
        print("🎉 评估完成！")
        print("\n📁 结果文件位置:")
        print("- 日志: ./log/")
        print("- 视频: ./log/*/videos/")
        print("- 配置: ./config/experiment/simpler/pi0_local_test.yaml")
    else:
        print("❌ 评估失败，请检查日志")
    print("=" * 60)

if __name__ == "__main__":
    main()
