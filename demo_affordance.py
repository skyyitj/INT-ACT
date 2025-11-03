#!/usr/bin/env python3
"""
快速演示Affordance功能

运行这个脚本可以快速看到affordance的效果
"""

import numpy as np
import cv2
import simpler_env
from get_pose_corrected_coordinates import add_affordance_to_observation, get_robot_pose_and_image_from_env
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict


def quick_demo():
    """快速演示：生成对比图"""
    print("=" * 60)
    print("Affordance快速演示")
    print("=" * 60)
    print("\n正在初始化环境...")

    task_name = "widowx_carrot_on_plate"
    env = simpler_env.make(task_name)

    try:
        obs, reset_info = env.reset(seed=42)
        print("✓ 环境已初始化\n")

        # 获取原始图像
        print("正在获取原始图像...")
        if "image" in obs:
            cam_imgs = obs["image"]
            if "3rd_view_camera" in cam_imgs and "rgb" in cam_imgs["3rd_view_camera"]:
                img_original = cam_imgs["3rd_view_camera"]["rgb"].copy()
            elif "base_camera" in cam_imgs and "rgb" in cam_imgs["base_camera"]:
                img_original = cam_imgs["base_camera"]["rgb"].copy()
            else:
                img_original = get_image_from_maniskill2_obs_dict(env, obs)
        else:
            img_original = get_image_from_maniskill2_obs_dict(env, obs)

        print(f"✓ 原始图像大小: {img_original.shape}\n")

        # 添加affordance
        print("正在添加affordance...")
        obs_with_aff = add_affordance_to_observation(
            obs, env,
            arrow_color=(0, 255, 0),  # 绿色
            arrow_thickness=4,
            arrow_length=0.08,
            show_point=True
        )
        print("✓ Affordance已添加\n")

        # 获取带affordance的图像
        if "image" in obs_with_aff:
            cam_imgs = obs_with_aff["image"]
            if "3rd_view_camera" in cam_imgs and "rgb" in cam_imgs["3rd_view_camera"]:
                img_affordance = cam_imgs["3rd_view_camera"]["rgb"]
            elif "base_camera" in cam_imgs and "rgb" in cam_imgs["base_camera"]:
                img_affordance = cam_imgs["base_camera"]["rgb"]
            else:
                img_affordance = get_image_from_maniskill2_obs_dict(env, obs_with_aff)
        else:
            img_affordance = get_image_from_maniskill2_obs_dict(env, obs_with_aff)

        # 保存图像
        print("正在保存图像...")
        cv2.imwrite('demo_original.png', cv2.cvtColor(img_original, cv2.COLOR_RGB2BGR))
        cv2.imwrite('demo_with_affordance.png', cv2.cvtColor(img_affordance, cv2.COLOR_RGB2BGR))
        print("✓ 保存: demo_original.png")
        print("✓ 保存: demo_with_affordance.png")

        # 创建对比图
        print("\n正在创建对比图...")

        # 添加标签
        img_orig_labeled = img_original.copy()
        img_aff_labeled = img_affordance.copy()

        cv2.putText(img_orig_labeled, "Original", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(img_aff_labeled, "With Affordance", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # 水平拼接
        comparison = np.hstack([img_orig_labeled, img_aff_labeled])
        cv2.imwrite('demo_comparison.png', cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
        print("✓ 保存: demo_comparison.png")

        # 打印位姿信息
        print("\n" + "=" * 60)
        print("夹爪位姿信息:")
        print("=" * 60)
        data = get_robot_pose_and_image_from_env(env, obs)
        print(f"3D位置: {data['position']}")
        print(f"四元数: {data['quaternion']}")
        print(f"夹爪开合度: {data['gripper_width']:.3f}")

        print("\n" + "=" * 60)
        print("演示完成！")
        print("=" * 60)
        print("\n生成的文件:")
        print("  1. demo_original.png - 原始观测图像")
        print("  2. demo_with_affordance.png - 添加affordance的图像")
        print("  3. demo_comparison.png - 对比图")

        print("\n💡 提示:")
        print("  - 绿色箭头表示夹爪的朝向（X轴方向）")
        print("  - 箭头起点的圆点表示夹爪的位置")
        print("  - 这个affordance可以帮助策略更好地理解操作方向")

        print("\n下一步:")
        print("  1. 查看生成的图像")
        print("  2. 运行 'python get_pose_corrected_coordinates.py --affordance' 查看更多样式")
        print("  3. 运行 'python affordance_usage_guide.py' 学习如何集成到你的代码")
        print("  4. 运行 'python affordance_wrapper.py' 测试环境包装器")

        return True

    except Exception as e:
        print(f"\n❌ 演示失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        env.close()


if __name__ == "__main__":
    success = quick_demo()
    exit(0 if success else 1)

