#!/usr/bin/env python3
"""
Affordance工具函数
从get_pose_corrected_coordinates.py中提取的affordance相关功能
"""

import numpy as np
import cv2
from scipy.spatial.transform import Rotation
import copy

def project_3d_to_2d(point_3d, intrinsic_matrix, extrinsic_matrix):
    """将3D点投影到2D图像坐标"""
    point_3d_homo = np.array([point_3d[0], point_3d[1], point_3d[2], 1.0])
    point_cam = extrinsic_matrix @ point_3d_homo

    if point_cam[2] <= 0:
        return None

    point_2d_homo = intrinsic_matrix @ point_cam[:3]
    u = point_2d_homo[0] / point_2d_homo[2]
    v = point_2d_homo[1] / point_2d_homo[2]

    return [int(u), int(v)]

def draw_affordance_arrow(image, origin_2d, position_3d, quaternion,
                         intrinsic_matrix, extrinsic_matrix,
                         arrow_length=0.08, arrow_color=(0, 255, 0),
                         arrow_thickness=3, show_point=True):
    """
    在图像上绘制夹爪朝向的affordance箭头

    Args:
        image: 输入图像
        origin_2d: 夹爪在图像中的2D坐标
        position_3d: 夹爪的3D世界坐标
        quaternion: 姿态四元数 [w, x, y, z]
        intrinsic_matrix: 相机内参矩阵
        extrinsic_matrix: 相机外参矩阵
        arrow_length: 箭头长度 (米)
        arrow_color: 箭头颜色 (B, G, R)
        arrow_thickness: 箭头粗细
        show_point: 是否显示夹爪位置点

    Returns:
        是否绘制成功
    """
    try:
        # 将四元数转换为旋转矩阵
        quat_scipy = [quaternion[1], quaternion[2], quaternion[3], quaternion[0]]
        rotation = Rotation.from_quat(quat_scipy)
        rotation_matrix = rotation.as_matrix()

        # X轴方向代表夹爪朝向（向前）
        direction_3d = rotation_matrix @ np.array([arrow_length, 0, 0])

        # 计算箭头端点的3D世界坐标
        arrow_end_3d = position_3d + direction_3d

        # 投影箭头端点到2D
        end_2d = project_3d_to_2d(arrow_end_3d, intrinsic_matrix, extrinsic_matrix)

        if end_2d is not None:
            u_orig, v_orig = origin_2d
            # 绘制箭头
            cv2.arrowedLine(image, (u_orig, v_orig), tuple(end_2d),
                          arrow_color, arrow_thickness, tipLength=0.3)

            # 可选：绘制夹爪位置点
            if show_point:
                cv2.circle(image, (u_orig, v_orig), 5, arrow_color, -1)
                cv2.circle(image, (u_orig, v_orig), 7, (255, 255, 255), 2)

            return True
        else:
            return False

    except Exception as e:
        print(f"绘制affordance箭头失败: {e}")
        return False

def draw_affordance_arrow_simple(image, origin_2d, quaternion,
                                 arrow_length=60, arrow_color=(0, 255, 0),
                                 arrow_thickness=3, show_point=True):
    """
    简化版affordance箭头绘制（当无法进行3D投影时使用）
    直接在2D图像平面上根据四元数估算方向

    Args:
        image: 输入图像
        origin_2d: 夹爪在图像中的2D坐标
        quaternion: 姿态四元数 [w, x, y, z]
        arrow_length: 箭头长度（像素）
        arrow_color: 箭头颜色 (B, G, R)
        arrow_thickness: 箭头粗细
        show_point: 是否显示夹爪位置点
    """
    u, v = origin_2d

    # 将四元数转换为旋转矩阵
    quat_scipy = [quaternion[1], quaternion[2], quaternion[3], quaternion[0]]
    rotation = Rotation.from_quat(quat_scipy)
    rotation_matrix = rotation.as_matrix()

    # 获取X轴方向（夹爪朝向）
    direction_3d = rotation_matrix @ np.array([1, 0, 0])

    # 简化投影：假设从第三人称视角看
    # X -> 图像右方向, Y -> 图像左方向, Z -> 图像上方向
    dx = direction_3d[0] * arrow_length * 0.8 - direction_3d[1] * arrow_length * 0.3
    dy = -direction_3d[2] * arrow_length * 0.3 + direction_3d[0] * arrow_length * 0.2

    end_x = int(u + dx)
    end_y = int(v + dy)

    # 绘制箭头
    cv2.arrowedLine(image, (u, v), (end_x, end_y),
                   arrow_color, arrow_thickness, tipLength=0.3)

    # 可选：绘制夹爪位置点
    if show_point:
        cv2.circle(image, (u, v), 5, arrow_color, -1)
        cv2.circle(image, (u, v), 7, (255, 255, 255), 2)

def get_robot_pose_from_obs(obs):
    """从观测中获取机械臂位姿信息"""
    if "extra" in obs and "tcp_pose" in obs["extra"]:
        tcp_pose = obs["extra"]["tcp_pose"]
        position = tcp_pose[:3]        # 位置 (x, y, z)
        quaternion = tcp_pose[3:7]     # 四元数 (w, x, y, z)
        return position, quaternion
    else:
        raise KeyError("未在obs中找到tcp_pose")

def add_affordance_to_observation(obs, env, arrow_length=0.08,
                                 arrow_color=(0, 255, 0),
                                 arrow_thickness=3, show_point=True):
    """
    向观测图像中添加affordance信息（夹爪朝向箭头）

    Args:
        obs: 环境观测字典
        env: 环境实例
        arrow_length: 箭头长度（米，用于3D投影）
        arrow_color: 箭头颜色 (B, G, R)，默认绿色
        arrow_thickness: 箭头粗细
        show_point: 是否显示夹爪位置点

    Returns:
        添加了affordance的观测字典（深拷贝）
    """
    # 深拷贝观测以避免修改原始数据
    obs_with_affordance = copy.deepcopy(obs)

    try:
        # 获取位姿信息
        position, quaternion = get_robot_pose_from_obs(obs)

        # 获取所有相机的图像
        if "image" in obs_with_affordance:
            cam_imgs = obs_with_affordance["image"]

            for camera_name in cam_imgs.keys():
                if "rgb" in cam_imgs[camera_name]:
                    image = cam_imgs[camera_name]["rgb"]

                    # 尝试3D到2D投影
                    position_2d = None
                    use_real_projection = False

                    camera_params = obs.get("camera_param", None)
                    if camera_params and camera_name in camera_params:
                        try:
                            params = camera_params[camera_name]
                            intrinsic = params['intrinsic_cv']
                            extrinsic = params['extrinsic_cv']

                            position_2d = project_3d_to_2d(
                                position, intrinsic, extrinsic
                            )

                            if position_2d:
                                use_real_projection = True
                        except Exception:
                            pass

                    # 如果投影失败，使用图像中心
                    if position_2d is None:
                        h, w = image.shape[:2]
                        position_2d = [w // 2, h // 2]

                    # 绘制affordance箭头
                    if use_real_projection:
                        success = draw_affordance_arrow(
                            image, position_2d, position,
                            quaternion, intrinsic, extrinsic,
                            arrow_length, arrow_color, arrow_thickness, show_point
                        )
                        if not success:
                            draw_affordance_arrow_simple(
                                image, position_2d, quaternion,
                                60, arrow_color, arrow_thickness, show_point
                            )
                    else:
                        draw_affordance_arrow_simple(
                            image, position_2d, quaternion,
                            60, arrow_color, arrow_thickness, show_point
                        )

        return obs_with_affordance

    except Exception as e:
        print(f"添加affordance失败: {e}")
        return obs_with_affordance
