#!/usr/bin/env python3
"""
机械臂夹爪6D位姿图像标注示例

这个脚本展示了如何在虚拟环境图像上标注机械臂夹爪的6D位姿信息。
包括位置、姿态（四元数）和夹爪开合度。

使用方法:
1. 确保已安装 SimplerEnv 和相关依赖
2. 运行: python pose_annotation_example.py
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from get_robot_pose_demo import (
    get_robot_ee_pose_from_simpler_env,
    get_camera_params_from_env,
    project_3d_to_2d,
    draw_coordinate_frame,
    draw_pose_annotation,
    annotate_robot_pose_on_image
)

def simple_pose_annotation_demo():
    """
    简化的位姿标注演示
    如果无法获取相机参数，使用图像中心作为夹爪位置
    """
    print("=== 简化位姿标注演示 ===\n")
    
    try:
        # 获取机械臂位姿和环境
        pose_info = get_robot_ee_pose_from_simpler_env()
        env = pose_info['env']
        obs = pose_info['obs']
        
        print("机械臂位姿信息:")
        print(f"  位置: {pose_info['position']}")
        print(f"  四元数: {pose_info['quaternion']}")
        print(f"  夹爪开合度: {pose_info['gripper_width']:.3f}")
        
        # 获取图像
        from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
        image = get_image_from_maniskill2_obs_dict(env, obs)
        print(f"图像尺寸: {image.shape}")
        
        # 使用图像中心作为夹爪位置（简化方法）
        h, w = image.shape[:2]
        center_u, center_v = w // 2, h // 2
        
        # 绘制标注
        annotated_image = image.copy()
        annotated_image = draw_pose_annotation(
            annotated_image, 
            [center_u, center_v], 
            pose_info['quaternion'], 
            pose_info['gripper_width']
        )
        
        # 保存结果
        cv2.imwrite('simple_pose_annotation.png', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
        print("已保存简化标注图像: simple_pose_annotation.png")
        
        # 显示结果
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title('原始图像')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(annotated_image)
        plt.title('标注了位姿信息的图像')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('simple_annotation_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        env.close()
        return annotated_image
        
    except Exception as e:
        print(f"运行出错: {e}")
        return None

def advanced_pose_annotation_demo():
    """
    高级位姿标注演示
    使用相机参数进行精确的3D到2D投影
    """
    print("=== 高级位姿标注演示 ===\n")
    
    try:
        # 运行完整的位姿标注功能
        annotated_image = annotate_robot_pose_on_image()
        return annotated_image
        
    except Exception as e:
        print(f"运行出错: {e}")
        return None

def create_pose_visualization():
    """
    创建位姿可视化图表
    """
    print("=== 创建位姿可视化图表 ===\n")
    
    try:
        # 获取位姿信息
        pose_info = get_robot_ee_pose_from_simpler_env()
        env = pose_info['env']
        
        # 创建可视化图表
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 位置信息
        position = pose_info['position']
        axes[0, 0].bar(['X', 'Y', 'Z'], position, color=['red', 'green', 'blue'])
        axes[0, 0].set_title('夹爪位置 (米)')
        axes[0, 0].set_ylabel('位置 (m)')
        
        # 四元数信息
        quaternion = pose_info['quaternion']
        axes[0, 1].bar(['W', 'X', 'Y', 'Z'], quaternion, color=['purple', 'red', 'green', 'blue'])
        axes[0, 1].set_title('夹爪姿态四元数')
        axes[0, 1].set_ylabel('四元数值')
        
        # 夹爪开合度
        gripper_width = pose_info['gripper_width']
        axes[1, 0].bar(['夹爪开合度'], [gripper_width], color='orange')
        axes[1, 0].set_title('夹爪开合度')
        axes[1, 0].set_ylabel('开合度 (0=闭合, 1=张开)')
        axes[1, 0].set_ylim(0, 1)
        
        # 位姿信息文本
        info_text = f"""位姿信息:
位置: ({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f})
四元数: ({quaternion[0]:.3f}, {quaternion[1]:.3f}, {quaternion[2]:.3f}, {quaternion[3]:.3f})
夹爪开合度: {gripper_width:.3f}"""
        
        axes[1, 1].text(0.1, 0.5, info_text, fontsize=12, verticalalignment='center')
        axes[1, 1].set_title('位姿详细信息')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig('pose_visualization.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        env.close()
        print("已保存位姿可视化图表: pose_visualization.png")
        
    except Exception as e:
        print(f"创建可视化图表出错: {e}")

def main():
    """主函数"""
    print("机械臂夹爪6D位姿图像标注示例")
    print("=" * 50)
    
    # 选择运行模式
    print("请选择运行模式:")
    print("1. 简化标注演示（使用图像中心）")
    print("2. 高级标注演示（使用相机参数）")
    print("3. 位姿可视化图表")
    print("4. 运行所有演示")
    
    try:
        choice = input("请输入选择 (1-4): ").strip()
        
        if choice == '1':
            simple_pose_annotation_demo()
        elif choice == '2':
            advanced_pose_annotation_demo()
        elif choice == '3':
            create_pose_visualization()
        elif choice == '4':
            print("\n运行所有演示...")
            simple_pose_annotation_demo()
            print("\n" + "="*50 + "\n")
            advanced_pose_annotation_demo()
            print("\n" + "="*50 + "\n")
            create_pose_visualization()
        else:
            print("无效选择，运行简化演示...")
            simple_pose_annotation_demo()
            
    except KeyboardInterrupt:
        print("\n用户中断程序")
    except Exception as e:
        print(f"程序运行出错: {e}")

if __name__ == "__main__":
    main()
