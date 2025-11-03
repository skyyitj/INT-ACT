#!/usr/bin/env python3
"""
PI0评估 - 带Affordance支持

对比测试有无affordance对策略性能的影响
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def setup_environment():
    """设置环境变量和Python路径"""
    project_root = Path(__file__).parent.absolute()

    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(project_root / 'third_party/lerobot'))

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
        'LOCAL_RANK': '0',
        'RANK': '0',
        'WORLD_SIZE': '1',
        'MASTER_ADDR': 'localhost',
        'MASTER_PORT': '29500'
    }

    for key, value in env_vars.items():
        os.environ[key] = value

    for dir_path in [env_vars['VLA_DATA_DIR'], env_vars['VLA_LOG_DIR']]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

    print(f"✅ 环境设置完成，项目根目录: {project_root}")
    return project_root


def create_affordance_config(model_path, use_affordance=True,
                             affordance_color=(0, 255, 0),
                             affordance_thickness=3,
                             affordance_length=0.08,
                             affordance_show_point=True,
                             config_suffix="affordance",
                             n_episodes=10):
    """创建带affordance的配置文件"""

    print(f"\n📝 创建配置文件 (use_affordance={use_affordance})...")

    # BGR颜色转字符串
    color_str = f"[{affordance_color[0]}, {affordance_color[1]}, {affordance_color[2]}]"

    config_content = f"""name: pi0_{config_suffix}
seed: 42
model_cfg: !include ../../models/pi0_baseline_bridge.json

eval_cfg:
  simulator_name: "simpler"
  env_adapter: "BridgeSimplerAdapter"

  task_list: [
    "widowx_spoon_on_towel",
    "widowx_carrot_on_plate",
    "widowx_stack_cube",
    "widowx_put_eggplant_in_basket",
  ]

  n_eval_episode: {n_episodes}
  n_video: {min(n_episodes, 3)}
  recording: True
  pretrained_model_path: {model_path}

  # Affordance配置
  use_affordance: {str(use_affordance).lower()}
  affordance_color: {color_str}
  affordance_thickness: {affordance_thickness}
  affordance_length: {affordance_length}
  affordance_show_point: {str(affordance_show_point).lower()}

env:
  dataset_statistics_path: ./config/dataset/bridge_statistics.json

wandb:
  project: "vla_affordance_experiment"
"""

    config_path = Path(f"config/experiment/simpler/pi0_{config_suffix}.yaml")
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, 'w') as f:
        f.write(config_content)

    print(f"✅ 配置文件创建: {config_path}")
    return str(config_path)


def run_evaluation(config_path, experiment_name):
    """运行评估"""
    print(f"\n🚀 开始评估: {experiment_name}")
    print("=" * 60)

    cmd = [
        "python", "src/agent/run.py",
        "--config_path", config_path,
        "--use_bf16", "False",
        "--use_wandb", "False"  # 如果要使用wandb，改为True
    ]

    print(f"执行命令: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)  # 2小时超时

        if result.returncode == 0:
            print(f"\n✅ {experiment_name} 评估完成！")
            print("\n📊 输出:")
            print(result.stdout[-1000:])
            return True, result.stdout
        else:
            print(f"\n❌ {experiment_name} 评估失败")
            print(f"错误: {result.stderr}")
            return False, result.stderr

    except subprocess.TimeoutExpired:
        print(f"\n⏰ {experiment_name} 评估超时")
        return False, "Timeout"
    except Exception as e:
        print(f"\n❌ {experiment_name} 评估出错: {e}")
        return False, str(e)


def main():
    parser = argparse.ArgumentParser(description="PI0 Affordance评估")
    parser.add_argument("--model-path", default="./models/INTACT-pi0-finetune-bridge",
                       help="模型路径")
    parser.add_argument("--mode", choices=["baseline", "affordance", "compare"],
                       default="compare", help="评估模式")
    parser.add_argument("--affordance-color", default="0,255,0",
                       help="Affordance颜色 (B,G,R)")
    parser.add_argument("--affordance-thickness", type=int, default=3,
                       help="Affordance粗细")
    parser.add_argument("--affordance-length", type=float, default=0.08,
                       help="Affordance长度(米)")
    parser.add_argument("--n-episodes", type=int, default=10,
                       help="每个任务的评估次数")
    parser.add_argument("--config-only", action="store_true",
                       help="只创建配置文件")

    args = parser.parse_args()

    print("🎯 PI0 Affordance评估实验")
    print("=" * 60)

    # 设置环境
    project_root = setup_environment()

    # 解析颜色
    try:
        color = tuple(map(int, args.affordance_color.split(',')))
        if len(color) != 3:
            raise ValueError
    except:
        print("❌ 颜色格式错误，使用默认绿色")
        color = (0, 255, 0)

    results = {}

    if args.mode in ["baseline", "compare"]:
        print("\n" + "=" * 60)
        print("📊 实验组1: Baseline (无Affordance)")
        print("=" * 60)

        # 创建baseline配置
        baseline_config = create_affordance_config(
            args.model_path,
            use_affordance=False,
            config_suffix="baseline",
            n_episodes=args.n_episodes
        )

        if not args.config_only:
            success, output = run_evaluation(baseline_config, "Baseline")
            results['baseline'] = {'success': success, 'output': output}

    if args.mode in ["affordance", "compare"]:
        print("\n" + "=" * 60)
        print("📊 实验组2: With Affordance (有Affordance)")
        print("=" * 60)

        # 创建affordance配置
        affordance_config = create_affordance_config(
            args.model_path,
            use_affordance=True,
            affordance_color=color,
            affordance_thickness=args.affordance_thickness,
            affordance_length=args.affordance_length,
            config_suffix="with_affordance",
            n_episodes=args.n_episodes
        )

        if not args.config_only:
            success, output = run_evaluation(affordance_config, "With Affordance")
            results['affordance'] = {'success': success, 'output': output}

    # 总结
    print("\n" + "=" * 60)
    print("📈 实验总结")
    print("=" * 60)

    if args.config_only:
        print("✅ 配置文件创建完成")
        print("\n📁 配置文件位置:")
        if args.mode in ["baseline", "compare"]:
            print("  - config/experiment/simpler/pi0_baseline.yaml")
        if args.mode in ["affordance", "compare"]:
            print("  - config/experiment/simpler/pi0_with_affordance.yaml")
        print("\n💡 手动运行评估:")
        print("  python src/agent/run.py --config_path <配置文件路径>")
    else:
        for exp_name, result in results.items():
            status = "✅ 成功" if result['success'] else "❌ 失败"
            print(f"{exp_name}: {status}")

        print("\n📁 结果文件位置:")
        print("  - 日志: ./log/")
        print("  - 视频: ./log/*/videos/")
        print("\n💡 提示: 查看日志文件获取详细的成功率统计")

    print("=" * 60)


if __name__ == "__main__":
    main()

