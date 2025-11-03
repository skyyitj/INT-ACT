#!/usr/bin/env python3
"""
简化的Affordance功能测试
只测试配置生成和基本功能，避免复杂的环境依赖
"""

import os
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

def test_affordance_config_generation():
    """测试affordance配置生成"""
    print("=== 测试Affordance配置生成 ===")

    try:
        from run_pi0_simpler_standalone import create_server_config, create_client_config

        # 测试不同的affordance配置
        test_configs = [
            {
                'name': '默认绿色',
                'use_affordance': True,
                'config': {
                    'color': [0, 255, 0],
                    'thickness': 3,
                    'length': 0.08,
                    'show_point': True
                }
            },
            {
                'name': '红色粗箭头',
                'use_affordance': True,
                'config': {
                    'color': [0, 0, 255],
                    'thickness': 5,
                    'length': 0.1,
                    'show_point': True
                }
            },
            {
                'name': '无affordance',
                'use_affordance': False,
                'config': {
                    'color': [0, 255, 0],
                    'thickness': 3,
                    'length': 0.08,
                    'show_point': True
                }
            }
        ]

        success_count = 0

        for i, test_config in enumerate(test_configs):
            print(f"\n{i+1}. 测试配置: {test_config['name']}")

            try:
                # 生成服务器配置
                server_config = create_server_config(
                    model_path="./test_model",
                    port=5000 + i,
                    use_affordance=test_config['use_affordance'],
                    affordance_config=test_config['config']
                )

                # 生成客户端配置
                client_config = create_client_config(
                    model_path="./test_model",
                    port=5000 + i,
                    n_episodes=3,
                    n_videos=1,
                    use_affordance=test_config['use_affordance'],
                    affordance_config=test_config['config']
                )

                # 验证配置内容
                with open(server_config, 'r') as f:
                    server_content = f.read()

                with open(client_config, 'r') as f:
                    client_content = f.read()

                # 检查affordance设置
                expected_affordance = str(test_config['use_affordance']).lower()
                if f"use_affordance: {expected_affordance}" in server_content:
                    print(f"   ✅ 服务器配置正确: use_affordance={expected_affordance}")
                else:
                    print(f"   ❌ 服务器配置错误")
                    continue

                if f"use_affordance: {expected_affordance}" in client_content:
                    print(f"   ✅ 客户端配置正确: use_affordance={expected_affordance}")
                else:
                    print(f"   ❌ 客户端配置错误")
                    continue

                # 如果启用了affordance，检查参数
                if test_config['use_affordance']:
                    color = test_config['config']['color']
                    thickness = test_config['config']['thickness']
                    length = test_config['config']['length']

                    if f"affordance_color: {color}" in server_content:
                        print(f"   ✅ 颜色配置正确: {color}")
                    else:
                        print(f"   ❌ 颜色配置错误")
                        continue

                    if f"affordance_thickness: {thickness}" in server_content:
                        print(f"   ✅ 粗细配置正确: {thickness}")
                    else:
                        print(f"   ❌ 粗细配置错误")
                        continue

                    if f"affordance_length: {length}" in server_content:
                        print(f"   ✅ 长度配置正确: {length}")
                    else:
                        print(f"   ❌ 长度配置错误")
                        continue

                print(f"   ✅ 配置 '{test_config['name']}' 测试通过")
                success_count += 1

            except Exception as e:
                print(f"   ❌ 配置 '{test_config['name']}' 测试失败: {e}")

        print(f"\n配置生成测试结果: {success_count}/{len(test_configs)} 通过")
        return success_count == len(test_configs)

    except Exception as e:
        print(f"❌ 配置生成测试失败: {e}")
        return False

def test_command_line_parsing():
    """测试命令行参数解析"""
    print("\n=== 测试命令行参数解析 ===")

    try:
        import argparse

        # 模拟run_pi0_simpler_standalone.py的参数解析
        parser = argparse.ArgumentParser(description="测试affordance参数")
        parser.add_argument("--use-affordance", action="store_true", help="启用affordance功能")
        parser.add_argument("--affordance-color", nargs=3, type=int, default=[0, 255, 0],
                           help="Affordance箭头颜色 (B G R), 默认绿色")
        parser.add_argument("--affordance-thickness", type=int, default=3,
                           help="Affordance箭头粗细")
        parser.add_argument("--affordance-length", type=float, default=0.08,
                           help="Affordance箭头长度(米)")
        parser.add_argument("--no-affordance-point", action="store_true",
                           help="不显示affordance位置点")

        # 测试不同的参数组合
        test_cases = [
            {
                'name': '默认参数',
                'args': [],
                'expected': {
                    'use_affordance': False,
                    'affordance_color': [0, 255, 0],
                    'affordance_thickness': 3,
                    'affordance_length': 0.08,
                    'no_affordance_point': False
                }
            },
            {
                'name': '启用affordance',
                'args': ['--use-affordance'],
                'expected': {
                    'use_affordance': True,
                    'affordance_color': [0, 255, 0],
                    'affordance_thickness': 3,
                    'affordance_length': 0.08,
                    'no_affordance_point': False
                }
            },
            {
                'name': '自定义红色箭头',
                'args': ['--use-affordance', '--affordance-color', '0', '0', '255', '--affordance-thickness', '5'],
                'expected': {
                    'use_affordance': True,
                    'affordance_color': [0, 0, 255],
                    'affordance_thickness': 5,
                    'affordance_length': 0.08,
                    'no_affordance_point': False
                }
            },
            {
                'name': '完整自定义',
                'args': ['--use-affordance', '--affordance-color', '255', '255', '0',
                        '--affordance-thickness', '4', '--affordance-length', '0.1', '--no-affordance-point'],
                'expected': {
                    'use_affordance': True,
                    'affordance_color': [255, 255, 0],
                    'affordance_thickness': 4,
                    'affordance_length': 0.1,
                    'no_affordance_point': True
                }
            }
        ]

        success_count = 0

        for i, test_case in enumerate(test_cases):
            print(f"\n{i+1}. 测试: {test_case['name']}")
            print(f"   参数: {' '.join(test_case['args'])}")

            try:
                args = parser.parse_args(test_case['args'])

                # 验证解析结果
                all_correct = True
                for key, expected_value in test_case['expected'].items():
                    actual_value = getattr(args, key.replace('-', '_'))
                    if actual_value != expected_value:
                        print(f"   ❌ {key}: 期望 {expected_value}, 实际 {actual_value}")
                        all_correct = False
                    else:
                        print(f"   ✅ {key}: {actual_value}")

                if all_correct:
                    print(f"   ✅ 参数解析正确")
                    success_count += 1
                else:
                    print(f"   ❌ 参数解析错误")

            except Exception as e:
                print(f"   ❌ 参数解析失败: {e}")

        print(f"\n参数解析测试结果: {success_count}/{len(test_cases)} 通过")
        return success_count == len(test_cases)

    except Exception as e:
        print(f"❌ 参数解析测试失败: {e}")
        return False

def test_imports():
    """测试关键模块导入"""
    print("\n=== 测试关键模块导入 ===")

    imports_to_test = [
        {
            'name': 'Affordance工具函数',
            'import_statement': 'from src.utils.affordance_utils import add_affordance_to_observation, draw_affordance_arrow'
        },
        {
            'name': 'Affordance适配器',
            'import_statement': 'from src.experiments.env_adapters.simpler_with_affordance import BridgeSimplerAdapterWithAffordance'
        },
        {
            'name': '动态类加载',
            'import_statement': 'from src.utils.pipeline import get_class_from_path'
        }
    ]

    success_count = 0

    for i, test_import in enumerate(imports_to_test):
        print(f"\n{i+1}. 测试: {test_import['name']}")

        try:
            exec(test_import['import_statement'])
            print(f"   ✅ 导入成功")
            success_count += 1
        except Exception as e:
            print(f"   ❌ 导入失败: {e}")

    print(f"\n导入测试结果: {success_count}/{len(imports_to_test)} 通过")
    return success_count == len(imports_to_test)

def main():
    """主测试函数"""
    print("🎯 INT-ACT Affordance功能简化测试")
    print("=" * 50)

    tests = [
        ("配置生成", test_affordance_config_generation),
        ("参数解析", test_command_line_parsing),
        ("模块导入", test_imports),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        result = test_func()
        results.append((test_name, result))

    # 总结
    print("\n" + "=" * 50)
    print("📊 测试结果总结:")
    print("=" * 50)

    passed = 0
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1

    total = len(results)
    print(f"\n总计: {passed}/{total} 个测试通过")

    if passed == total:
        print("\n🎉 所有测试通过！Affordance功能基础组件正常！")
        print("\n💡 下一步:")
        print("  1. 修复评估器中的logger问题（已修复）")
        print("  2. 运行完整测试:")
        print("     python run_pi0_simpler_standalone.py --use-affordance --episodes 3")
        print("  3. 检查生成的视频是否包含affordance箭头")

    else:
        print(f"\n⚠️  有 {total - passed} 个测试失败，请检查配置")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
