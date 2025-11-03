 #!/usr/bin/env python3
"""
本地standalone模式的pi0 SimplerEnv评估脚本
同时启动服务器和客户端进程
"""

import os
import sys
import subprocess
import argparse
import time
import signal
from pathlib import Path
from multiprocessing import Process

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
        'MASTER_PORT': '29501',  # 使用不同的端口避免冲突
        # 强制使用单GPU模式
        'CUDA_VISIBLE_DEVICES': '0',
    }

    for key, value in env_vars.items():
        os.environ[key] = value

    # 创建必要的目录
    for dir_path in [env_vars['VLA_DATA_DIR'], env_vars['VLA_LOG_DIR']]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

    print(f"✅ 环境设置完成，项目根目录: {project_root}")
    return project_root

def download_pi0_model(model_name="juexzz/INTACT-pi0-finetune-bridge"):
    """下载pi0预训练模型"""
    print(f"\n📥 下载pi0模型: {model_name}")

    try:
        from huggingface_hub import snapshot_download

        # 下载模型到本地目录
        model_dir = Path("./models") / model_name.split("/")[-1]

        if model_dir.exists():
            print(f"✅ 模型已存在: {model_dir}")
            return str(model_dir)

        model_dir.mkdir(parents=True, exist_ok=True)

        print(f"正在下载模型到: {model_dir}")
        snapshot_download(
            repo_id=model_name,
            local_dir=str(model_dir),
        )

        print(f"✅ 模型下载完成: {model_dir}")
        return str(model_dir)

    except Exception as e:
        print(f"❌ 模型下载失败: {e}")
        print("💡 请手动下载模型或检查网络连接")
        return None

def create_server_config(model_path, port=5000, use_affordance=False, affordance_config=None):
    """创建服务器配置文件"""
    print("\n📝 创建服务器配置...")

    # 默认affordance配置
    if affordance_config is None:
        affordance_config = {
            'color': [0, 255, 0],  # 绿色 (BGR格式)
            'thickness': 3,
            'length': 0.08,
            'show_point': True
        }

    # 根据是否使用affordance选择不同的环境适配器
    # 注意：配置系统会自动处理适配器映射，这里只需要指定标准适配器名称
    env_adapter = "BridgeSimplerAdapter"

    config_content = f"""name: pi0_server
seed: 42
model_cfg: !include ../../models/hf_pi0.json

# 强制禁用多GPU模式
multi_gpu: false
n_nodes: 1

eval_cfg:
  simulator_name: "simpler"
  env_adapter: "{env_adapter}"
  pretrained_model_path: {model_path}
  role: "server"
  host: "0.0.0.0"
  port: {port}

  # Affordance配置
  use_affordance: {str(use_affordance).lower()}
  affordance_color: {affordance_config['color']}
  affordance_thickness: {affordance_config['thickness']}
  affordance_length: {affordance_config['length']}
  affordance_show_point: {str(affordance_config['show_point']).lower()}

env:
  dataset_statistics_path: ./config/dataset/bridge_statistics.json
"""

    config_path = Path("config/experiment/simpler/pi0_server.yaml")
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, 'w') as f:
        f.write(config_content)

    print(f"✅ 服务器配置创建: {config_path}")
    return str(config_path)

def create_client_config(model_path, port=5000, n_episodes=20, n_videos=5, use_affordance=False, affordance_config=None):
    """创建客户端配置文件"""
    print(f"\n📝 创建客户端配置... (测试轮数: {n_episodes})")

    # 默认affordance配置
    if affordance_config is None:
        affordance_config = {
            'color': [0, 255, 0],  # 绿色 (BGR格式)
            'thickness': 3,
            'length': 0.08,
            'show_point': True
        }

    # 根据是否使用affordance选择不同的环境适配器
    # 注意：配置系统会自动处理适配器映射，这里只需要指定标准适配器名称
    env_adapter = "BridgeSimplerAdapter"

    config_content = f"""name: pi0_client
seed: 42
model_cfg: !include ../../models/hf_pi0.json

# 强制禁用多GPU模式
multi_gpu: false
n_nodes: 1

eval_cfg:
  simulator_name: "simpler"
  env_adapter: "{env_adapter}"
  task_list: [
    "widowx_spoon_on_towel",
    "widowx_carrot_on_plate",
    "widowx_stack_cube",
    "widowx_put_eggplant_in_basket",
  ]

  n_eval_episode: {n_episodes}
  n_video: {n_videos}
  recording: True
  pretrained_model_path: {model_path}
  role: "client"
  host: "127.0.0.1"
  port: {port}

  # Affordance配置
  use_affordance: {str(use_affordance).lower()}
  affordance_color: {affordance_config['color']}
  affordance_thickness: {affordance_config['thickness']}
  affordance_length: {affordance_config['length']}
  affordance_show_point: {str(affordance_config['show_point']).lower()}

env:
  dataset_statistics_path: ./config/dataset/bridge_statistics.json

wandb:
  project: "vla_benchmark_local"
"""

    config_path = Path("config/experiment/simpler/pi0_client.yaml")
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, 'w') as f:
        f.write(config_content)

    print(f"✅ 客户端配置创建: {config_path}")
    return str(config_path)

def run_server(server_config):
    """运行策略服务器"""
    print("\n🚀 启动策略服务器...")

    cmd = [
        "python", "src/agent/run.py",
        "--config_path", server_config,
        "--use_bf16", "False",
        "--use_wandb", "False"
    ]

    print(f"服务器命令: {' '.join(cmd)}")

    # 运行服务器（会一直运行直到客户端完成）
    subprocess.run(cmd)

def run_client(client_config, n_episodes=20):
    """运行评估客户端"""
    print("\n🚀 启动评估客户端...")

    # 等待服务器启动
    print("⏳ 等待服务器启动...")
    time.sleep(10)

    cmd = [
        "python", "src/agent/run.py",
        "--config_path", client_config,
        "--use_bf16", "False",
        "--use_wandb", "False"
    ]

    print(f"客户端命令: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print("✅ 评估完成成功！")

        # 解析和显示结果统计
        parse_and_display_results(result.stdout, n_episodes)

        print("\n📊 详细输出:")
        print(result.stdout[-1500:])
    else:
        print("❌ 评估失败")
        print(f"错误: {result.stderr}")
        print(f"输出: {result.stdout[-1000:]}")

    return result.returncode == 0

def parse_and_display_results(output, n_episodes):
    """解析并显示评估结果统计"""
    print("\n" + "="*60)
    print("📈 评估结果统计")
    print("="*60)

    tasks = [
        "widowx_spoon_on_towel",
        "widowx_carrot_on_plate",
        "widowx_stack_cube",
        "widowx_put_eggplant_in_basket"
    ]

    total_success = 0
    total_episodes = 0

    for task in tasks:
        # 尝试从输出中提取成功率信息
        success_count = 0
        if f"{task}" in output:
            # 简单的成功率估算（实际解析可能需要更复杂的逻辑）
            task_lines = [line for line in output.split('\n') if task in line]
            for line in task_lines:
                if 'success' in line.lower() or 'completed' in line.lower():
                    success_count += 1

        # 如果无法从输出解析，显示配置信息
        success_rate = (success_count / n_episodes * 100) if n_episodes > 0 else 0

        print(f"🎯 {task}:")
        print(f"   测试轮数: {n_episodes}")
        print(f"   成功次数: {success_count}")
        print(f"   成功率: {success_rate:.1f}%")
        print()

        total_success += success_count
        total_episodes += n_episodes

    overall_success_rate = (total_success / total_episodes * 100) if total_episodes > 0 else 0

    print("🏆 总体统计:")
    print(f"   总测试次数: {total_episodes}")
    print(f"   总成功次数: {total_success}")
    print(f"   总体成功率: {overall_success_rate:.1f}%")
    print("="*60)

def check_and_kill_port(port):
    """检查并释放端口"""
    try:
        import psutil
        for conn in psutil.net_connections():
            if conn.laddr.port == port:
                print(f"⚠️  端口 {port} 被进程 {conn.pid} 占用，尝试释放...")
                try:
                    process = psutil.Process(conn.pid)
                    process.terminate()
                    process.wait(timeout=3)
                    print(f"✅ 已释放端口 {port}")
                except:
                    print(f"⚠️  无法自动释放，请手动执行: kill -9 {conn.pid}")
                    return False
        return True
    except ImportError:
        print("💡 提示: 安装 psutil 可以自动清理端口 (pip install psutil)")
        return True
    except Exception as e:
        print(f"⚠️  检查端口时出错: {e}")
        return True

def main():
    parser = argparse.ArgumentParser(description="本地standalone模式pi0评估")
    parser.add_argument("--model-name", default="juexzz/INTACT-pi0-finetune-bridge",
                       help="HuggingFace模型名称")
    parser.add_argument("--port", type=int, default=5000, help="服务器端口")
    parser.add_argument("--episodes", type=int, default=20, help="每个任务的测试轮数")
    parser.add_argument("--videos", type=int, default=5, help="录制视频的数量")
    parser.add_argument("--skip-download", action="store_true", help="跳过模型下载")
    parser.add_argument("--config-only", action="store_true", help="只创建配置文件")

    # Affordance相关参数
    parser.add_argument("--use-affordance", action="store_true", help="启用affordance功能")
    parser.add_argument("--affordance-color", nargs=3, type=int, default=[0, 255, 0],
                       help="Affordance箭头颜色 (B G R), 默认绿色")
    parser.add_argument("--affordance-thickness", type=int, default=3,
                       help="Affordance箭头粗细")
    parser.add_argument("--affordance-length", type=float, default=0.08,
                       help="Affordance箭头长度(米)")
    parser.add_argument("--no-affordance-point", action="store_true",
                       help="不显示affordance位置点")

    args = parser.parse_args()

    print("🤖 INT-ACT PI0 SimplerEnv Standalone评估")
    print("=" * 60)

    # 检查并清理端口
    check_and_kill_port(args.port)

    # 1. 设置环境
    project_root = setup_environment()

    # 2. 下载模型（如果需要）
    model_path = None
    if not args.skip_download:
        model_path = download_pi0_model(args.model_name)
        if model_path is None and not args.config_only:
            print("❌ 无法继续，模型下载失败")
            return
    else:
        model_path = f"./models/{args.model_name.split('/')[-1]}"

    # 3. 准备affordance配置
    affordance_config = {
        'color': args.affordance_color,
        'thickness': args.affordance_thickness,
        'length': args.affordance_length,
        'show_point': not args.no_affordance_point
    }

    # 显示affordance配置信息
    if args.use_affordance:
        print(f"\n🎯 Affordance功能已启用:")
        print(f"  颜色 (BGR): {affordance_config['color']}")
        print(f"  粗细: {affordance_config['thickness']}")
        print(f"  长度: {affordance_config['length']}m")
        print(f"  显示位置点: {affordance_config['show_point']}")

    # 4. 创建配置
    server_config = create_server_config(model_path, args.port, args.use_affordance, affordance_config)
    client_config = create_client_config(model_path, args.port, args.episodes, args.videos, args.use_affordance, affordance_config)

    print(f"\n📊 测试配置:")
    print(f"  - 每个任务测试轮数: {args.episodes}")
    print(f"  - 录制视频数量: {args.videos}")
    print(f"  - 总测试次数: {args.episodes * 4} (4个任务)")

    if args.config_only:
        print("✅ 配置文件创建完成，退出")
        return

    # 4. 启动服务器和客户端
    print("\n" + "=" * 60)
    print("🎬 启动服务器-客户端评估")
    print("=" * 60)

    # 创建服务器进程
    server_process = Process(target=run_server, args=(server_config,))
    server_process.start()

    try:
        # 运行客户端（主进程）
        success = run_client(client_config, args.episodes)
        
        print("\n" + "=" * 60)
        if success:
            print("🎉 评估完成！")
            print("\n📁 结果文件位置:")
            print("- 日志: ./log/")
            print("- 视频: ./log/*/videos/")
            print("- 配置: ./config/experiment/simpler/")
        else:
            print("❌ 评估失败，请检查日志")
        print("=" * 60)
        
    finally:
        # 终止服务器进程
        print("\n🛑 关闭服务器...")
        server_process.terminate()
        server_process.join(timeout=5)
        if server_process.is_alive():
            server_process.kill()
        print("✅ 服务器已关闭")

if __name__ == "__main__":
    main()

