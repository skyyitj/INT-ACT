#!/usr/bin/env python3
"""
PI0评估 - 带Affordance支持

对比测试有无affordance对策略性能的影响
"""

import os
import sys
import subprocess
import argparse
import time
import signal
from pathlib import Path
from multiprocessing import Process


def setup_environment(master_port=29501):
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
        'MASTER_PORT': str(master_port),  # 动态设置PyTorch分布式端口
        'CUDA_VISIBLE_DEVICES': '0',
    }

    for key, value in env_vars.items():
        os.environ[key] = value

    for dir_path in [env_vars['VLA_DATA_DIR'], env_vars['VLA_LOG_DIR']]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

    print(f"✅ 环境设置完成，项目根目录: {project_root}")
    print(f"📡 PyTorch分布式端口: {master_port}")
    return project_root


def create_server_config(model_path, use_affordance=True,
                        affordance_color=(0, 255, 0),
                        affordance_thickness=3,
                        affordance_length=0.08,
                        affordance_show_point=True,
                        config_suffix="affordance",
                        port=5000):
    """创建服务器配置文件"""
    print(f"\n📝 创建服务器配置 (use_affordance={use_affordance})...")

    # BGR颜色转字符串
    color_str = f"[{affordance_color[0]}, {affordance_color[1]}, {affordance_color[2]}]"

    config_content = f"""name: pi0_server_{config_suffix}
seed: 42
model_cfg: !include ../../models/hf_pi0.json

# 强制禁用多GPU模式
multi_gpu: false
n_nodes: 1

eval_cfg:
  simulator_name: "simpler"
  env_adapter: "BridgeSimplerAdapter"
  pretrained_model_path: {model_path}
  role: "server"
  host: "0.0.0.0"
  port: {port}

  # Affordance配置
  use_affordance: {str(use_affordance).lower()}
  affordance_color: {color_str}
  affordance_thickness: {affordance_thickness}
  affordance_length: {affordance_length}
  affordance_show_point: {str(affordance_show_point).lower()}

env:
  dataset_statistics_path: ./config/dataset/bridge_statistics.json
"""

    config_path = Path(f"config/experiment/simpler/pi0_server_{config_suffix}.yaml")
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, 'w') as f:
        f.write(config_content)

    print(f"✅ 服务器配置创建: {config_path}")
    return str(config_path)


def create_client_config(model_path, use_affordance=True,
                        affordance_color=(0, 255, 0),
                        affordance_thickness=3,
                        affordance_length=0.08,
                        affordance_show_point=True,
                        config_suffix="affordance",
                        n_episodes=10,
                        port=5000):
    """创建客户端配置文件"""
    print(f"\n📝 创建客户端配置 (use_affordance={use_affordance})...")

    # BGR颜色转字符串
    color_str = f"[{affordance_color[0]}, {affordance_color[1]}, {affordance_color[2]}]"

    config_content = f"""name: pi0_client_{config_suffix}
seed: 42
model_cfg: !include ../../models/hf_pi0.json

# 强制禁用多GPU模式
multi_gpu: false
n_nodes: 1

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
  role: "client"
  host: "127.0.0.1"
  port: {port}

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

    config_path = Path(f"config/experiment/simpler/pi0_client_{config_suffix}.yaml")
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


def run_client(client_config):
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
        print("\n📊 输出:")
        print(result.stdout[-1000:])
        return True, result.stdout
    else:
        print("❌ 评估失败")
        print(f"错误: {result.stderr}")
        return False, result.stderr


def check_and_kill_port(port):
    """检查并释放端口"""
    try:
        import psutil
        killed_any = False
        for conn in psutil.net_connections():
            if hasattr(conn, 'laddr') and conn.laddr and conn.laddr.port == port:
                print(f"⚠️  端口 {port} 被进程 {conn.pid} 占用，尝试释放...")
                try:
                    if conn.pid:  # 确保pid存在
                        process = psutil.Process(conn.pid)
                        process.terminate()
                        process.wait(timeout=3)
                        print(f"✅ 已释放端口 {port}")
                        killed_any = True
                    else:
                        print(f"⚠️  无法获取进程ID，跳过端口 {port}")
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired) as e:
                    print(f"⚠️  无法释放端口 {port}: {e}")
                except Exception as e:
                    print(f"⚠️  释放端口时出错: {e}")

        if not killed_any:
            print(f"✅ 端口 {port} 未被占用")
        return True

    except ImportError:
        print("💡 提示: 安装 psutil 可以自动清理端口 (pip install psutil)")
        return True
    except Exception as e:
        print(f"⚠️  检查端口时出错: {e}")
        return True


def simple_port_check(port):
    """简单的端口检查（不杀进程）"""
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            result = s.connect_ex(('localhost', port))
            if result == 0:
                print(f"⚠️  端口 {port} 被占用")
                return False
            else:
                print(f"✅ 端口 {port} 可用")
                return True
    except Exception as e:
        print(f"⚠️  检查端口 {port} 时出错: {e}")
        return True  # 假设可用


def find_free_port(start_port, max_attempts=10):
    """找到可用的端口"""
    import socket

    for i in range(max_attempts):
        port = start_port + i
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                print(f"✅ 找到可用端口: {port}")
                return port
        except OSError:
            print(f"⚠️  端口 {port} 被占用，尝试下一个...")
            continue

    print(f"❌ 无法找到可用端口 (尝试了 {start_port}-{start_port + max_attempts - 1})")
    return None


def run_experiment_with_server_client(server_config, client_config, experiment_name):
    """运行带服务器-客户端架构的实验"""
    print(f"\n🚀 开始实验: {experiment_name}")
    print("=" * 60)

    # 创建服务器进程
    server_process = Process(target=run_server, args=(server_config,))
    server_process.start()

    try:
        # 运行客户端（主进程）
        success, output = run_client(client_config)
        return success, output

    finally:
        # 终止服务器进程
        print("\n🛑 关闭服务器...")
        server_process.terminate()
        server_process.join(timeout=5)
        if server_process.is_alive():
            server_process.kill()
        print("✅ 服务器已关闭")


def run_evaluation(config_path, experiment_name):
    """运行评估 - 保留原函数用于兼容性"""
    print(f"\n⚠️  警告: 使用旧的单进程模式运行 {experiment_name}")
    print("建议使用服务器-客户端模式获得更好的性能")

    cmd = [
        "python", "src/agent/run.py",
        "--config_path", config_path,
        "--use_bf16", "False",
        "--use_wandb", "False"
    ]

    print(f"执行命令: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)

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
    parser.add_argument("--port", type=int, default=5001,
                       help="服务器端口")
    parser.add_argument("--use-single-process", action="store_true",
                       help="使用单进程模式（不推荐）")
    parser.add_argument("--safe-port-check", action="store_true",
                       help="使用安全的端口检查（不杀进程）")

    args = parser.parse_args()

    print("🎯 PI0 Affordance评估实验")
    print("=" * 60)

    # 为不同实验分配不同的PyTorch分布式端口
    pytorch_base_port = 29500
    server_port = args.port

    # 检查并清理端口
    if not args.use_single_process:
        print("🔍 检查端口占用情况...")

        if args.safe_port_check:
            # 使用安全的端口检查（不杀进程）
            simple_port_check(args.port)
            simple_port_check(pytorch_base_port)
            simple_port_check(pytorch_base_port + 1)
        else:
            # 使用psutil检查并尝试释放端口
            try:
                check_and_kill_port(args.port)
                check_and_kill_port(pytorch_base_port)
                check_and_kill_port(pytorch_base_port + 1)
            except Exception as e:
                print(f"⚠️  端口检查出错，切换到安全模式: {e}")
                simple_port_check(args.port)
                simple_port_check(pytorch_base_port)

        # 找到可用的PyTorch分布式端口
        pytorch_port = find_free_port(pytorch_base_port)
        if pytorch_port is None:
            print("❌ 无法找到可用的PyTorch分布式端口，退出")
            return
    else:
        pytorch_port = pytorch_base_port

    # 设置环境（使用找到的PyTorch端口）
    project_root = setup_environment(master_port=pytorch_port)

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

        if args.use_single_process:
            # 单进程模式（旧方式）
            baseline_config = create_client_config(
                args.model_path,
                use_affordance=False,
                config_suffix="baseline",
                n_episodes=args.n_episodes,
                port=server_port
            )

            if not args.config_only:
                success, output = run_evaluation(baseline_config, "Baseline")
                results['baseline'] = {'success': success, 'output': output}
        else:
            # 服务器-客户端模式（推荐）
            server_config = create_server_config(
                args.model_path,
                use_affordance=False,
                config_suffix="baseline",
                port=server_port
            )

            client_config = create_client_config(
                args.model_path,
                use_affordance=False,
                config_suffix="baseline",
                n_episodes=args.n_episodes,
                port=server_port
            )

            if not args.config_only:
                success, output = run_experiment_with_server_client(
                    server_config, client_config, "Baseline"
                )
                results['baseline'] = {'success': success, 'output': output}

    if args.mode in ["affordance", "compare"]:
        print("\n" + "=" * 60)
        print("📊 实验组2: With Affordance (有Affordance)")
        print("=" * 60)

        # 为affordance实验使用不同的端口
        affordance_server_port = server_port + 1
        affordance_pytorch_port = pytorch_port + 1

        if not args.use_single_process:
            # 更新环境变量使用新的PyTorch端口
            os.environ['MASTER_PORT'] = str(affordance_pytorch_port)
            print(f"📡 Affordance实验PyTorch端口: {affordance_pytorch_port}")

        if args.use_single_process:
            # 单进程模式（旧方式）
            affordance_config = create_client_config(
                args.model_path,
                use_affordance=True,
                affordance_color=color,
                affordance_thickness=args.affordance_thickness,
                affordance_length=args.affordance_length,
                config_suffix="with_affordance",
                n_episodes=args.n_episodes,
                port=affordance_server_port
            )

            if not args.config_only:
                success, output = run_evaluation(affordance_config, "With Affordance")
                results['affordance'] = {'success': success, 'output': output}
        else:
            # 服务器-客户端模式（推荐）
            server_config = create_server_config(
                args.model_path,
                use_affordance=True,
                affordance_color=color,
                affordance_thickness=args.affordance_thickness,
                affordance_length=args.affordance_length,
                config_suffix="with_affordance",
                port=affordance_server_port
            )

            client_config = create_client_config(
                args.model_path,
                use_affordance=True,
                affordance_color=color,
                affordance_thickness=args.affordance_thickness,
                affordance_length=args.affordance_length,
                config_suffix="with_affordance",
                n_episodes=args.n_episodes,
                port=affordance_server_port
            )

            if not args.config_only:
                success, output = run_experiment_with_server_client(
                    server_config, client_config, "With Affordance"
                )
                results['affordance'] = {'success': success, 'output': output}

    # 总结
    print("\n" + "=" * 60)
    print("📈 实验总结")
    print("=" * 60)

    if args.config_only:
        print("✅ 配置文件创建完成")
        print("\n📁 配置文件位置:")
        if args.mode in ["baseline", "compare"]:
            if args.use_single_process:
                print("  - config/experiment/simpler/pi0_client_baseline.yaml")
            else:
                print("  - config/experiment/simpler/pi0_server_baseline.yaml")
                print("  - config/experiment/simpler/pi0_client_baseline.yaml")
        if args.mode in ["affordance", "compare"]:
            if args.use_single_process:
                print("  - config/experiment/simpler/pi0_client_with_affordance.yaml")
            else:
                print("  - config/experiment/simpler/pi0_server_with_affordance.yaml")
                print("  - config/experiment/simpler/pi0_client_with_affordance.yaml")
        print("\n💡 手动运行评估:")
        if args.use_single_process:
            print("  python src/agent/run.py --config_path <客户端配置文件路径>")
        else:
            print("  使用本脚本运行: python run_pi0_with_affordance.py --mode <模式>")
    else:
        for exp_name, result in results.items():
            status = "✅ 成功" if result['success'] else "❌ 失败"
            print(f"{exp_name}: {status}")

        print("\n📁 结果文件位置:")
        print("  - 日志: ./log/")
        print("  - 视频: ./log/*/videos/")
        print("\n💡 提示: 查看日志文件获取详细的成功率统计")

    if not args.use_single_process:
        print(f"\n🚀 使用了服务器-客户端架构，PyTorch端口: {pytorch_port}")

    print("=" * 60)


if __name__ == "__main__":
    main()

