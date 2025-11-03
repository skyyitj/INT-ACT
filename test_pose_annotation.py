#!/usr/bin/env python3
"""
测试机械臂夹爪6D位姿图像标注功能

这个脚本用于测试位姿标注功能是否正常工作
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

def test_basic_functions():
    """测试基础功能"""
    print("=== 测试基础功能 ===\n")
    
    # 测试3D到2D投影函数
    print("1. 测试3D到2D投影...")
    
    # 模拟相机参数
    intrinsic_matrix = np.array([
        [525.0, 0.0, 320.0],
        [0.0, 525.0, 240.0],
        [0.0, 0.0, 1.0]
    ])
    
    extrinsic_matrix = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    
    # 测试点
    test_point_3d = [0.5, 0.0, 1.0]  # 在相机前方1米处
    
    try:
        from get_robot_pose_demo import project_3d_to_2d
        position_2d = project_3d_to_2d(test_point_3d, intrinsic_matrix, extrinsic_matrix)
        print(f"   测试点3D坐标: {test_point_3d}")
        print(f"   投影到2D坐标: {position_2d}")
        print("   ✓ 3D到2D投影测试通过")
    except Exception as e:
        print(f"   ✗ 3D到2D投影测试失败: {e}")
    
    # 测试图像绘制函数
    print("\n2. 测试图像绘制...")
    
    try:
        # 创建测试图像
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        test_image.fill(128)  # 灰色背景
        
        from get_robot_pose_demo import draw_coordinate_frame, draw_pose_annotation
        
        # 测试坐标系绘制
        center_point = [320, 240]  # 图像中心
        annotated_image = draw_coordinate_frame(test_image.copy(), center_point)
        print("   ✓ 坐标系绘制测试通过")
        
        # 测试位姿标注绘制
        test_quaternion = [1.0, 0.0, 0.0, 0.0]
        test_gripper_width = 0.5
        annotated_image = draw_pose_annotation(
            annotated_image, 
            center_point, 
            test_quaternion, 
            test_gripper_width
        )
        print("   ✓ 位姿标注绘制测试通过")
        
        # 保存测试图像
        cv2.imwrite('test_annotation.png', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
        print("   已保存测试图像: test_annotation.png")
        
    except Exception as e:
        print(f"   ✗ 图像绘制测试失败: {e}")

def test_pose_extraction():
    """测试位姿提取功能"""
    print("\n=== 测试位姿提取功能 ===\n")
    
    try:
        from get_robot_pose_demo import get_robot_ee_pose_from_simpler_env
        
        print("1. 尝试获取机械臂位姿...")
        pose_info = get_robot_ee_pose_from_simpler_env()
        
        print("   ✓ 位姿提取成功")
        print(f"   位置: {pose_info['position']}")
        print(f"   四元数: {pose_info['quaternion']}")
        print(f"   夹爪开合度: {pose_info['gripper_width']}")
        
        # 清理环境
        if 'env' in pose_info:
            pose_info['env'].close()
            print("   ✓ 环境清理完成")
        
    except Exception as e:
        print(f"   ✗ 位姿提取测试失败: {e}")
        print("   这可能是因为 SimplerEnv 未正确安装或配置")

def test_camera_params():
    """测试相机参数获取"""
    print("\n=== 测试相机参数获取 ===\n")
    
    try:
        from get_robot_pose_demo import get_robot_ee_pose_from_simpler_env, get_camera_params_from_env
        
        print("1. 尝试获取相机参数...")
        pose_info = get_robot_ee_pose_from_simpler_env()
        env = pose_info['env']
        
        camera_params = get_camera_params_from_env(env)
        
        if camera_params:
            print("   ✓ 相机参数获取成功")
            for name, params in camera_params.items():
                print(f"   相机 {name}:")
                print(f"     内参矩阵形状: {params['intrinsic_cv'].shape}")
                print(f"     外参矩阵形状: {params['extrinsic_cv'].shape}")
        else:
            print("   ⚠ 无法获取相机参数，将使用简化模式")
        
        # 清理环境
        env.close()
        
    except Exception as e:
        print(f"   ✗ 相机参数获取测试失败: {e}")

def create_test_visualization():
    """创建测试可视化"""
    print("\n=== 创建测试可视化 ===\n")
    
    try:
        # 创建测试数据
        position = np.array([0.5, 0.0, 0.3])
        quaternion = np.array([1.0, 0.0, 0.0, 0.0])
        gripper_width = 0.8
        
        # 创建可视化图表
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 位置信息
        axes[0, 0].bar(['X', 'Y', 'Z'], position, color=['red', 'green', 'blue'])
        axes[0, 0].set_title('夹爪位置 (米)')
        axes[0, 0].set_ylabel('位置 (m)')
        
        # 四元数信息
        axes[0, 1].bar(['W', 'X', 'Y', 'Z'], quaternion, color=['purple', 'red', 'green', 'blue'])
        axes[0, 1].set_title('夹爪姿态四元数')
        axes[0, 1].set_ylabel('四元数值')
        
        # 夹爪开合度
        axes[1, 0].bar(['夹爪开合度'], [gripper_width], color='orange')
        axes[1, 0].set_title('夹爪开合度')
        axes[1, 0].set_ylabel('开合度 (0=闭合, 1=张开)')
        axes[1, 0].set_ylim(0, 1)
        
        # 位姿信息文本
        info_text = f"""测试位姿信息:
位置: ({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f})
四元数: ({quaternion[0]:.3f}, {quaternion[1]:.3f}, {quaternion[2]:.3f}, {quaternion[3]:.3f})
夹爪开合度: {gripper_width:.3f}"""
        
        axes[1, 1].text(0.1, 0.5, info_text, fontsize=12, verticalalignment='center')
        axes[1, 1].set_title('位姿详细信息')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig('test_pose_visualization.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("   ✓ 测试可视化创建成功")
        print("   已保存测试可视化图表: test_pose_visualization.png")
        
    except Exception as e:
        print(f"   ✗ 测试可视化创建失败: {e}")

def main():
    """主测试函数"""
    print("机械臂夹爪6D位姿图像标注功能测试")
    print("=" * 50)
    
    # 运行所有测试
    test_basic_functions()
    test_pose_extraction()
    test_camera_params()
    create_test_visualization()
    
    print("\n" + "=" * 50)
    print("测试完成！")
    print("\n如果所有测试都通过，说明功能正常工作。")
    print("如果有测试失败，请检查相关依赖是否正确安装。")

if __name__ == "__main__":
    main()
