#!/usr/bin/env python3
"""
测试增强的物体标注功能
显示所有物体，包括超出图像范围的物体
"""

import numpy as np
import cv2
import simpler_env
from get_pose_corrected_coordinates import (
    get_object_coordinates_from_env,
    get_robot_pose_and_image_from_env,
    create_enhanced_annotated_image,
    create_annotated_image
)

def test_enhanced_object_annotation():
    """测试增强的物体标注功能"""
    print("=== 测试增强物体标注功能 ===\n")

    task_name = "widowx_carrot_on_plate"
    env = simpler_env.make(task_name)

    try:
        obs, reset_info = env.reset(seed=42)

        # 获取数据
        data = get_robot_pose_and_image_from_env(env, obs)
        objects_info = get_object_coordinates_from_env(env, obs, reset_info)
        data['objects_info'] = objects_info

        print(f"找到 {len(objects_info)} 个物体:")
        for obj_key, obj_info in objects_info.items():
            name = obj_info['name']
            pos = obj_info['position']
            obj_type = obj_info['type']
            print(f"  - {name} ({obj_type}): [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")

        # 创建对比图像
        test_configs = [
            {
                'name': '原始标注方法',
                'function': create_annotated_image,
                'filename': 'original_annotation.png',
                'description': '只显示在图像范围内的物体'
            },
            {
                'name': '增强标注方法',
                'function': create_enhanced_annotated_image,
                'filename': 'enhanced_annotation.png',
                'description': '显示所有物体，包括超出范围的'
            }
        ]

        saved_images = []

        for i, config in enumerate(test_configs):
            print(f"\n{i+1}. {config['name']}:")
            print(f"   {config['description']}")

            # 创建标注图像
            annotated_img = config['function'](
                data,
                config['name'],
                config['filename'],
                include_objects=True
            )

            saved_images.append(annotated_img)
            print(f"   保存: {config['filename']}")

        # 创建对比图
        print(f"\n创建对比图...")
        if len(saved_images) >= 2:
            # 调整大小以便对比
            target_h = 400
            resized = []
            for img in saved_images:
                h, w = img.shape[:2]
                target_w = int(w * target_h / h)
                resized.append(cv2.resize(img, (target_w, target_h)))

            # 水平拼接
            comparison = np.hstack(resized)
            cv2.imwrite('annotation_comparison.png', cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
            print("保存对比图: annotation_comparison.png")

        print("\n✅ 增强标注测试完成!")
        print("\n生成的文件:")
        for config in test_configs:
            print(f"  - {config['filename']}: {config['description']}")
        print("  - annotation_comparison.png: 对比图")

        print("\n=== 功能说明 ===")
        print("增强标注方法的改进:")
        print("1. ✓ 显示所有物体，不跳过超出范围的")
        print("2. ✓ 超出范围的物体用空心圆+箭头指示")
        print("3. ✓ 显示详细统计信息（总数/可见/超出范围/投影失败）")
        print("4. ✓ 为超出范围的物体显示实际坐标")
        print("5. ✓ 投影失败的物体在右上角显示信息")

        return saved_images

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None

    finally:
        env.close()

def analyze_projection_issue():
    """分析投影问题的详细原因"""
    print("=== 投影问题详细分析 ===\n")

    # 模拟你的坐标数据
    objects_data = {
        'target_plate': {
            'position': np.array([-0.235, -0.075, 0.873]),
            'projection': [270, 176],
            'status': '✓ 在视野内'
        },
        'episode_target': {
            'position': np.array([-0.235, -0.075, 0.870]),
            'projection': [271, 179],
            'status': '✓ 在视野内'
        },
        'carrot_init': {
            'position': np.array([0.384, -0.047, 0.018]),
            'projection': [59, 1006],
            'status': '❌ Y坐标超出范围 (1006 > 480)'
        },
        'plate_init': {
            'position': np.array([0.382, 0.103, 0.000]),
            'projection': [158, 1100],
            'status': '❌ Y坐标超出范围 (1100 > 480)'
        }
    }

    print("物体投影分析:")
    print("图像尺寸: 640 x 480")
    print("-" * 60)

    for name, data in objects_data.items():
        pos = data['position']
        proj = data['projection']
        status = data['status']

        print(f"{name}:")
        print(f"  3D位置: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
        print(f"  2D投影: [{proj[0]}, {proj[1]}]")
        print(f"  状态: {status}")

        # 分析Z坐标影响
        if pos[2] < 0.1:
            print(f"  ⚠ Z坐标很低 ({pos[2]:.3f}m)，物体接近桌面")
        elif pos[2] > 0.5:
            print(f"  ✓ Z坐标较高 ({pos[2]:.3f}m)，物体在空中")

        print()

    print("=== 问题总结 ===")
    print("1. 盘子当前位置 (Z≈0.87m): 在空中，相机能看到 ✓")
    print("2. 胡萝卜初始位置 (Z=0.018m): 在桌面上，投影到图像下方 ❌")
    print("3. 盘子初始位置 (Z=0.000m): 在桌面水平，投影到图像下方 ❌")
    print("\n解决方案:")
    print("- 使用增强标注方法显示超出范围的物体")
    print("- 在图像边界显示指示器和坐标信息")
    print("- 提供详细的统计信息")

if __name__ == "__main__":
    print("增强物体标注测试工具")
    print("=" * 50)

    print("\n1. 投影问题分析")
    analyze_projection_issue()

    print("\n2. 增强标注功能测试")
    test_enhanced_object_annotation()

    print("\n测试完成！请查看生成的对比图像。")
