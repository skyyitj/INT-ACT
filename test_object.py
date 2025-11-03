#!/usr/bin/env python3
"""
调试物体坐标标注问题
分析为什么有些物体没有正确显示在图像中
"""

import numpy as np
import cv2
import simpler_env
from get_pose_corrected_coordinates import (
    get_object_coordinates_from_env,
    project_3d_to_2d,
    get_robot_pose_and_image_from_env
)

def debug_object_coordinates():
    """调试物体坐标和投影问题"""
    print("=== 调试物体坐标标注问题 ===\n")

    task_name = "widowx_carrot_on_plate"
    env = simpler_env.make(task_name)

    try:
        obs, reset_info = env.reset(seed=42)

        # 获取物体坐标信息
        objects_info = get_object_coordinates_from_env(env, obs, reset_info)

        print(f"找到 {len(objects_info)} 个物体:")
        for obj_key, obj_info in objects_info.items():
            name = obj_info['name']
            pos = obj_info['position']
            obj_type = obj_info['type']
            color = obj_info['color']
            print(f"  - {name} ({obj_type}): [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] 颜色:{color}")

        # 获取相机参数和图像
        data = get_robot_pose_and_image_from_env(env, obs)
        image = data['image'].copy()
        h, w = image.shape[:2]

        print(f"\n图像尺寸: {w} x {h}")
        print(f"相机名称: {data['camera_name']}")

        # 检查相机参数
        camera_params = data['camera_params']
        if camera_params and data['camera_name'] in camera_params:
            params = camera_params[data['camera_name']]
            intrinsic = params['intrinsic_cv']
            extrinsic = params['extrinsic_cv']
            print(f"相机内参矩阵:\n{intrinsic}")
            print(f"相机外参矩阵:\n{extrinsic}")
            use_real_projection = True
        else:
            print("⚠ 无法获取相机参数，无法进行3D到2D投影")
            use_real_projection = False

        # 分析每个物体的投影结果
        print(f"\n=== 物体投影分析 ===")

        annotated_image = image.copy()
        projection_results = {}

        for obj_key, obj_info in objects_info.items():
            name = obj_info['name']
            pos_3d = obj_info['position']
            color = obj_info['color']
            obj_type = obj_info['type']

            print(f"\n物体: {name} ({obj_type})")
            print(f"  3D坐标: [{pos_3d[0]:.3f}, {pos_3d[1]:.3f}, {pos_3d[2]:.3f}]")

            if use_real_projection:
                # 尝试3D到2D投影
                pos_2d = project_3d_to_2d(pos_3d, intrinsic, extrinsic)

                if pos_2d is not None:
                    u, v = pos_2d
                    print(f"  2D投影: [{u}, {v}]")

                    # 检查是否在图像范围内
                    in_bounds = (0 <= u < w and 0 <= v < h)
                    print(f"  在图像范围内: {in_bounds}")

                    if in_bounds:
                        # 绘制标记点
                        cv2.circle(annotated_image, (u, v), 8, color, -1)
                        cv2.circle(annotated_image, (u, v), 10, (255, 255, 255), 2)
                        cv2.circle(annotated_image, (u, v), 12, (0, 0, 0), 1)

                        # 添加标签
                        label = f"{name[:10]}"  # 截短名称
                        cv2.putText(annotated_image, label, (u + 15, v - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

                        projection_results[obj_key] = {
                            'success': True,
                            'pos_2d': pos_2d,
                            'in_bounds': True
                        }
                    else:
                        projection_results[obj_key] = {
                            'success': True,
                            'pos_2d': pos_2d,
                            'in_bounds': False
                        }
                else:
                    print(f"  2D投影: 失败 (物体在相机后方或无法投影)")
                    projection_results[obj_key] = {
                        'success': False,
                        'pos_2d': None,
                        'in_bounds': False
                    }
            else:
                print(f"  2D投影: 跳过 (无相机参数)")
                projection_results[obj_key] = {
                    'success': False,
                    'pos_2d': None,
                    'in_bounds': False
                }

        # 总结投影结果
        print(f"\n=== 投影结果总结 ===")
        successful_projections = 0
        in_bounds_projections = 0

        for obj_key, result in projection_results.items():
            obj_info = objects_info[obj_key]
            name = obj_info['name']

            if result['success']:
                successful_projections += 1
                if result['in_bounds']:
                    in_bounds_projections += 1
                    status = "✓ 成功显示"
                else:
                    status = "⚠ 投影成功但超出图像范围"
            else:
                status = "❌ 投影失败"

            print(f"  {name}: {status}")

        print(f"\n统计:")
        print(f"  总物体数: {len(objects_info)}")
        print(f"  投影成功: {successful_projections}")
        print(f"  在图像范围内: {in_bounds_projections}")

        # 保存调试图像
        cv2.imwrite('debug_object_projection.png', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
        print(f"\n保存调试图像: debug_object_projection.png")

        # 分析坐标分布
        print(f"\n=== 坐标分布分析 ===")
        positions = np.array([obj_info['position'] for obj_info in objects_info.values()])
        names = [obj_info['name'] for obj_info in objects_info.values()]

        print("X坐标范围:", f"[{positions[:, 0].min():.3f}, {positions[:, 0].max():.3f}]")
        print("Y坐标范围:", f"[{positions[:, 1].min():.3f}, {positions[:, 1].max():.3f}]")
        print("Z坐标范围:", f"[{positions[:, 2].min():.3f}, {positions[:, 2].max():.3f}]")

        # 计算物体间距离
        print(f"\n物体间距离:")
        for i in range(len(positions)):
            for j in range(i+1, len(positions)):
                dist = np.linalg.norm(positions[i] - positions[j])
                print(f"  {names[i]} <-> {names[j]}: {dist:.3f}m")

        return projection_results, objects_info

    except Exception as e:
        print(f"❌ 调试过程出错: {e}")
        import traceback
        traceback.print_exc()
        return None, None

    finally:
        env.close()

def analyze_coordinate_types():
    """分析不同类型坐标的含义"""
    print("=== 坐标类型分析 ===\n")

    task_name = "widowx_carrot_on_plate"
    env = simpler_env.make(task_name)

    try:
        obs, reset_info = env.reset(seed=42)
        objects_info = get_object_coordinates_from_env(env, obs, reset_info)

        print("根据你提供的坐标数据分析:")
        print("Found 4 objects:")
        print("  - target_plate (target_current): [-0.235, -0.075, 0.873]")
        print("  - bridge_plate_objaverse_larger (episode_target): [-0.235, -0.075, 0.870]")
        print("  - bridge_carrot_generated_modified (source_init): [0.384, -0.047, 0.018]")
        print("  - bridge_plate_objaverse_larger (target_init): [0.382, 0.103, -0.000]")

        print("\n坐标类型解释:")
        print("1. target_current: 目标物体的当前位置 (实时)")
        print("2. episode_target: Episode中的目标物体位置 (实时)")
        print("3. source_init: 源物体(胡萝卜)的初始位置")
        print("4. target_init: 目标物体(盘子)的初始位置")

        print("\n问题分析:")
        print("- target_plate 和 episode_target 坐标几乎相同 ([-0.235, -0.075, ~0.87])")
        print("  这两个都是盘子的当前位置，只是获取方式不同")
        print("- source_init 胡萝卜初始位置在 [0.384, -0.047, 0.018]")
        print("- target_init 盘子初始位置在 [0.382, 0.103, 0.000]")

        print("\n可能的问题:")
        print("1. 胡萝卜的初始位置 Z=0.018 很低，可能在桌面上")
        print("2. 盘子的初始位置 Z=0.000 在桌面水平")
        print("3. 相机角度可能看不到桌面上的物体")
        print("4. 3D到2D投影可能失败或超出图像范围")

        return objects_info

    except Exception as e:
        print(f"❌ 分析失败: {e}")
        return None

    finally:
        env.close()

def create_detailed_debug_image():
    """创建详细的调试图像，显示所有信息"""
    print("=== 创建详细调试图像 ===\n")

    task_name = "widowx_carrot_on_plate"
    env = simpler_env.make(task_name)

    try:
        obs, reset_info = env.reset(seed=42)

        # 获取数据
        data = get_robot_pose_and_image_from_env(env, obs)
        objects_info = get_object_coordinates_from_env(env, obs, reset_info)

        image = data['image'].copy()
        h, w = image.shape[:2]

        # 创建调试图像
        debug_image = image.copy()

        # 添加标题
        cv2.putText(debug_image, "Object Coordinate Debug", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        # 获取相机参数
        camera_params = data['camera_params']
        use_real_projection = False
        if camera_params and data['camera_name'] in camera_params:
            params = camera_params[data['camera_name']]
            intrinsic = params['intrinsic_cv']
            extrinsic = params['extrinsic_cv']
            use_real_projection = True

        # 在图像上显示物体信息
        y_offset = 60
        for i, (obj_key, obj_info) in enumerate(objects_info.items()):
            name = obj_info['name']
            pos_3d = obj_info['position']
            obj_type = obj_info['type']
            color = obj_info['color']

            # 显示3D坐标信息
            info_text = f"{name[:15]} ({obj_type})"
            coord_text = f"3D: [{pos_3d[0]:.3f}, {pos_3d[1]:.3f}, {pos_3d[2]:.3f}]"

            cv2.putText(debug_image, info_text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.putText(debug_image, coord_text, (10, y_offset + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            # 尝试投影
            if use_real_projection:
                pos_2d = project_3d_to_2d(pos_3d, intrinsic, extrinsic)
                if pos_2d is not None:
                    u, v = pos_2d
                    in_bounds = (0 <= u < w and 0 <= v < h)

                    proj_text = f"2D: [{u}, {v}] {'✓' if in_bounds else '✗'}"
                    cv2.putText(debug_image, proj_text, (10, y_offset + 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                               (0, 255, 0) if in_bounds else (0, 0, 255), 1)

                    # 如果在范围内，绘制标记
                    if in_bounds:
                        cv2.circle(debug_image, (u, v), 6, color, -1)
                        cv2.circle(debug_image, (u, v), 8, (255, 255, 255), 1)
                        cv2.putText(debug_image, f"{i+1}", (u + 10, v - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                else:
                    cv2.putText(debug_image, "2D: 投影失败", (10, y_offset + 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            else:
                cv2.putText(debug_image, "2D: 无相机参数", (10, y_offset + 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)

            y_offset += 50

        # 保存调试图像
        cv2.imwrite('detailed_debug_objects.png', cv2.cvtColor(debug_image, cv2.COLOR_RGB2BGR))
        print("保存详细调试图像: detailed_debug_objects.png")

        return debug_image

    except Exception as e:
        print(f"❌ 创建调试图像失败: {e}")
        import traceback
        traceback.print_exc()
        return None

    finally:
        env.close()

if __name__ == "__main__":
    print("物体坐标调试工具")
    print("=" * 40)

    print("\n1. 基础坐标调试")
    debug_object_coordinates()

    print("\n2. 坐标类型分析")
    analyze_coordinate_types()

    print("\n3. 创建详细调试图像")
    create_detailed_debug_image()

    print("\n调试完成！请查看生成的图像文件。")
