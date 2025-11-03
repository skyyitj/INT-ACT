#!/usr/bin/env python3
"""
从 SimplerEnv 获取机械臂末端手抓 6D 位姿的演示脚本
并实现在虚拟环境图像上标注6D位姿的功能

注意：由于 sapien 依赖问题，这个脚本展示了概念和代码结构，
实际运行需要正确安装 sapien 和 SimplerEnv。
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional
import simpler_env
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict

def get_robot_ee_pose_from_simpler_env():
    """从 SimplerEnv 获取机械臂末端位姿"""
    
    # 1. 创建环境
    task_name = "widowx_carrot_on_plate"
    env = simpler_env.make(task_name)
    
    # 2. 重置环境
    obs, reset_info = env.reset(seed=42)
    
    # 3. 获取机械臂状态
    agent_state = obs["observation.state"]["agent"]
    eef_pos = agent_state["eef_pos"]  # [x, y, z, qw, qx, qy, qz, gripper_width]
    
    # 4. 提取位姿信息
    position = eef_pos[:3]        # 位置 (x, y, z)
    quaternion = eef_pos[3:7]     # 四元数 (w, x, y, z)
    gripper_width = eef_pos[7]    # 手抓开合度
    
    # 5. 也可以通过机械臂对象直接获取
    agent = env.agent
    ee_pose = agent.ee_pose       # 末端执行器位姿
    base_pose = agent.base_pose   # 基座位姿
    
    # 6. 计算相对于基座的位姿
    ee_in_base = base_pose.inv() * ee_pose
    
    return {
        'position': position,
        'quaternion': quaternion,
        'gripper_width': gripper_width,
        'ee_pose_world': ee_pose,
        'ee_pose_base': ee_in_base,
        'env': env,
        'obs': obs
    }

def get_camera_params_from_env(env):
    """从环境中获取相机参数"""
    try:
        # 获取相机参数
        camera_params = env.get_camera_params()
        return camera_params
    except Exception as e:
        print(f"无法获取相机参数: {e}")
        return None

def project_3d_to_2d(point_3d, intrinsic_matrix, extrinsic_matrix):
    """
    将3D点投影到2D图像坐标
    
    Args:
        point_3d: 3D点坐标 [x, y, z]
        intrinsic_matrix: 相机内参矩阵 (3x3)
        extrinsic_matrix: 相机外参矩阵 (4x4)
    
    Returns:
        2D图像坐标 [u, v] 或 None (如果投影失败)
    """
    # 将3D点转换为齐次坐标
    point_3d_homo = np.array([point_3d[0], point_3d[1], point_3d[2], 1.0])
    
    # 应用外参变换 (世界坐标 -> 相机坐标)
    point_cam = extrinsic_matrix @ point_3d_homo
    
    # 检查点是否在相机前方
    if point_cam[2] <= 0:
        return None
    
    # 应用内参变换 (相机坐标 -> 图像坐标)
    point_2d_homo = intrinsic_matrix @ point_cam[:3]
    
    # 转换为像素坐标
    u = point_2d_homo[0] / point_2d_homo[2]
    v = point_2d_homo[1] / point_2d_homo[2]
    
    return [int(u), int(v)]

def draw_coordinate_frame(image, origin_2d, axis_length=50):
    """
    在图像上绘制坐标系
    
    Args:
        image: 输入图像
        origin_2d: 坐标系原点在图像中的2D坐标 [u, v]
        axis_length: 坐标轴长度（像素）
    
    Returns:
        绘制了坐标系的图像
    """
    u, v = origin_2d
    
    # 绘制坐标轴
    # X轴 - 红色
    cv2.arrowedLine(image, (u, v), (u + axis_length, v), (0, 0, 255), 3)
    cv2.putText(image, 'X', (u + axis_length + 5, v), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Y轴 - 绿色
    cv2.arrowedLine(image, (u, v), (u, v + axis_length), (0, 255, 0), 3)
    cv2.putText(image, 'Y', (u, v + axis_length + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Z轴 - 蓝色
    cv2.arrowedLine(image, (u, v), (u - axis_length//2, v - axis_length//2), (255, 0, 0), 3)
    cv2.putText(image, 'Z', (u - axis_length//2 - 20, v - axis_length//2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    # 绘制原点
    cv2.circle(image, (u, v), 5, (255, 255, 255), -1)
    cv2.circle(image, (u, v), 8, (0, 0, 0), 2)
    
    return image

def draw_pose_annotation(image, position_2d, quaternion, gripper_width):
    """
    在图像上绘制位姿标注
    
    Args:
        image: 输入图像
        position_2d: 夹爪位置在图像中的2D坐标
        quaternion: 夹爪姿态四元数
        gripper_width: 夹爪开合度
    
    Returns:
        绘制了位姿标注的图像
    """
    u, v = position_2d
    
    # 绘制夹爪位置点
    cv2.circle(image, (u, v), 8, (0, 255, 255), -1)  # 黄色圆点
    cv2.circle(image, (u, v), 12, (0, 0, 0), 2)      # 黑色边框
    
    # 绘制位姿信息文本
    info_text = f"Pos: ({position_2d[0]}, {position_2d[1]})"
    cv2.putText(image, info_text, (u + 15, v - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # 绘制四元数信息
    quat_text = f"Quat: ({quaternion[0]:.2f}, {quaternion[1]:.2f}, {quaternion[2]:.2f}, {quaternion[3]:.2f})"
    cv2.putText(image, quat_text, (u + 15, v + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # 绘制夹爪开合度
    gripper_text = f"Gripper: {gripper_width:.2f}"
    cv2.putText(image, gripper_text, (u + 15, v + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    return image

def annotate_robot_pose_on_image():
    """
    在虚拟环境图像上标注机械臂夹爪6D位姿的主函数
    """
    print("=== 在虚拟环境图像上标注机械臂夹爪6D位姿 ===\n")
    
    # 1. 获取机械臂位姿和环境
    pose_info = get_robot_ee_pose_from_simpler_env()
    env = pose_info['env']
    obs = pose_info['obs']
    
    print("1. 机械臂位姿信息:")
    print(f"   位置 (x, y, z): {pose_info['position']}")
    print(f"   四元数 (w, x, y, z): {pose_info['quaternion']}")
    print(f"   夹爪开合度: {pose_info['gripper_width']:.3f}")
    
    # 2. 获取虚拟环境图像
    print("\n2. 获取虚拟环境图像...")
    image = get_image_from_maniskill2_obs_dict(env, obs)
    print(f"   图像尺寸: {image.shape}")
    
    # 3. 获取相机参数
    print("\n3. 获取相机参数...")
    camera_params = get_camera_params_from_env(env)
    
    if camera_params is None:
        print("   警告: 无法获取相机参数，将使用简化的标注方法")
        # 使用简化的标注方法（假设图像中心为夹爪位置）
        annotated_image = image.copy()
        h, w = image.shape[:2]
        center_u, center_v = w // 2, h // 2
        
        # 绘制简化的标注
        annotated_image = draw_pose_annotation(
            annotated_image, 
            [center_u, center_v], 
            pose_info['quaternion'], 
            pose_info['gripper_width']
        )
        
        # 保存图像
        cv2.imwrite('robot_pose_annotation_simple.png', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
        print("   已保存简化标注图像: robot_pose_annotation_simple.png")
        
        return annotated_image
    
    # 4. 使用相机参数进行3D到2D投影
    print("\n4. 进行3D到2D投影...")
    
    # 假设使用第一个相机
    camera_name = list(camera_params.keys())[0]
    params = camera_params[camera_name]
    
    intrinsic_matrix = params['intrinsic_cv']
    extrinsic_matrix = params['extrinsic_cv']
    
    print(f"   使用相机: {camera_name}")
    print(f"   内参矩阵形状: {intrinsic_matrix.shape}")
    print(f"   外参矩阵形状: {extrinsic_matrix.shape}")
    
    # 投影夹爪位置到图像坐标
    position_2d = project_3d_to_2d(
        pose_info['position'], 
        intrinsic_matrix, 
        extrinsic_matrix
    )
    
    if position_2d is None:
        print("   警告: 3D点投影失败，夹爪可能不在相机视野内")
        return image
    
    print(f"   夹爪在图像中的位置: {position_2d}")
    
    # 5. 在图像上绘制标注
    print("\n5. 绘制位姿标注...")
    annotated_image = image.copy()
    
    # 绘制坐标系（如果位置在图像范围内）
    h, w = image.shape[:2]
    if 0 <= position_2d[0] < w and 0 <= position_2d[1] < h:
        annotated_image = draw_coordinate_frame(annotated_image, position_2d)
        annotated_image = draw_pose_annotation(
            annotated_image, 
            position_2d, 
            pose_info['quaternion'], 
            pose_info['gripper_width']
        )
    else:
        print("   警告: 夹爪位置超出图像范围，无法绘制完整标注")
    
    # 6. 保存结果
    print("\n6. 保存结果...")
    cv2.imwrite('robot_pose_annotation.png', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
    print("   已保存标注图像: robot_pose_annotation.png")
    
    # 7. 显示结果（如果在支持的环境中）
    try:
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title('原始图像')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(annotated_image)
        plt.title('标注了6D位姿的图像')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('pose_annotation_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("   已保存对比图像: pose_annotation_comparison.png")
    except Exception as e:
        print(f"   无法显示图像: {e}")
    
    # 8. 清理环境
    env.close()
    
    return annotated_image

# 使用示例
if __name__ == "__main__":
    try:
        # 运行位姿标注演示
        annotated_image = annotate_robot_pose_on_image()
        print("\n=== 标注完成 ===")
        
    except Exception as e:
        print(f"运行出错: {e}")
        print("请确保已正确安装 SimplerEnv 和相关依赖")
        
        # 运行简化的位姿获取演示
        print("\n=== 运行简化演示 ===")
        pose_info = get_robot_ee_pose_from_simpler_env()
        print("机械臂位姿信息:", pose_info)

    
#     print("代码示例:")
#     print(code_example)

# def show_coordinate_systems():
#     """展示坐标系说明"""
#     print("\n=== 坐标系说明 ===\n")
    
#     print("1. 世界坐标系 (World Frame):")
#     print("   - 环境中的全局坐标系")
#     print("   - 原点通常在地面或桌面上")
#     print("   - Z轴向上，X轴向前，Y轴向右")
    
#     print("\n2. 基座坐标系 (Base Frame):")
#     print("   - 机械臂基座的局部坐标系")
#     print("   - 原点在机械臂基座中心")
#     print("   - 用于描述机械臂各关节的相对位置")
    
#     print("\n3. 末端坐标系 (End-Effector Frame):")
#     print("   - 机械臂末端执行器的坐标系")
#     print("   - 原点在末端执行器中心")
#     print("   - 用于描述手抓的位置和姿态")
    
#     print("\n4. 坐标系转换:")
#     print("   - 世界坐标 → 基座坐标: base_pose.inv() * ee_pose")
#     print("   - 基座坐标 → 世界坐标: base_pose * ee_in_base")

# def main():
#     """主函数"""
#     print("机械臂末端手抓 6D 位姿获取完整指南")
#     print("=" * 50)
    
#     # 演示位姿提取
#     poses = demonstrate_pose_extraction()
    
#     # 展示实际使用方法
#     show_simpler_env_usage()
    
#     # 展示坐标系说明
#     show_coordinate_systems()
    
#     print("\n=== 注意事项 ===")
#     print("1. 四元数格式: SimplerEnv 使用 wxyz 格式")
#     print("2. 手抓状态: 0 表示闭合，1 表示张开")
#     print("3. 位姿更新: 每次调用 env.step() 后位姿会更新")
#     print("4. 坐标系转换: 可以使用相应的变换方法")
#     print("5. 实际使用需要正确安装 sapien 和 SimplerEnv")

# if __name__ == "__main__":
#     main()
