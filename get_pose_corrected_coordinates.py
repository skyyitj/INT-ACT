#!/usr/bin/env python3
"""
坐标系修正版本：机械臂末端6D位姿获取和图像标注
使用正确的坐标系约定和更准确的3D到2D投影
"""

import numpy as np
import cv2
import simpler_env
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
from scipy.spatial.transform import Rotation
import warnings

# 抑制警告
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium")

def get_robot_pose_and_image():
    """获取机械臂位姿和环境图像"""
    print("=== 获取机械臂位姿和图像 ===\n")

    task_name = "widowx_carrot_on_plate"
    env = simpler_env.make(task_name)

    try:
        obs, reset_info = env.reset(seed=42)

        if "extra" in obs and "tcp_pose" in obs["extra"]:
            tcp_pose = obs["extra"]["tcp_pose"]
            position = tcp_pose[:3]        # 位置 (x, y, z)
            quaternion = tcp_pose[3:7]     # 四元数 (w, x, y, z)

            print("✓ 成功获取夹爪末端位姿:")
            print(f"  位置 (m): {position}")
            print(f"  四元数 (wxyz): {quaternion}")

        else:
            raise KeyError(f"未在obs中找到tcp_pose")

        # 获取夹爪状态
        gripper_width = 0.5
        try:
            agent = env.unwrapped.agent if hasattr(env, 'unwrapped') else env.agent
            if hasattr(agent, 'gripper_closedness'):
                closedness = agent.gripper_closedness
                gripper_width = max(0.0, min(1.0, 1.0 - closedness))
        except Exception:
            pass

        print(f"  夹爪开合度: {gripper_width:.3f}")

        # 获取图像
        print("\n✓ 获取环境图像:")
        image = None
        camera_name = "unknown"

        if "image" in obs:
            cam_imgs = obs["image"]
            if "3rd_view_camera" in cam_imgs and "rgb" in cam_imgs["3rd_view_camera"]:
                image = cam_imgs["3rd_view_camera"]["rgb"]
                camera_name = "3rd_view_camera"
            elif "base_camera" in cam_imgs and "rgb" in cam_imgs["base_camera"]:
                image = cam_imgs["base_camera"]["rgb"]
                camera_name = "base_camera"

        if image is None:
            image = get_image_from_maniskill2_obs_dict(env, obs)
            camera_name = "default"

        print(f"  图像尺寸: {image.shape}")
        print(f"  使用相机: {camera_name}")

        # 获取相机参数
        camera_params = obs.get("camera_param", None)

        # 获取物体坐标信息
        objects_info = get_object_coordinates_from_env(env, obs, reset_info)

        return {
            'position': position,
            'quaternion': quaternion,
            'gripper_width': gripper_width,
            'image': image,
            'camera_params': camera_params,
            'camera_name': camera_name,
            'objects_info': objects_info,
            'env': env,
            'obs': obs
        }

    except Exception as e:
        env.close()
        raise e

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

def draw_coordinate_frame_correct(image, origin_2d, position_3d, quaternion,
                                 intrinsic_matrix, extrinsic_matrix, axis_length=0.05):
    """
    使用真实3D到2D投影绘制正确的坐标系

    Args:
        origin_2d: 原点在图像中的2D坐标
        position_3d: 原点的3D世界坐标
        quaternion: 姿态四元数 [w, x, y, z]
        axis_length: 坐标轴长度 (米)
    """
    try:
        # 将四元数转换为旋转矩阵
        # scipy使用 [x, y, z, w] 格式
        quat_scipy = [quaternion[1], quaternion[2], quaternion[3], quaternion[0]]
        rotation = Rotation.from_quat(quat_scipy)
        rotation_matrix = rotation.as_matrix()

        # 定义标准坐标轴向量 (在末端执行器坐标系中)
        # ROS/ManiSkill约定: X前, Y左, Z上
        axes_3d = np.array([
            [axis_length, 0, 0],  # X轴: 向前
            [0, axis_length, 0],  # Y轴: 向左
            [0, 0, axis_length]   # Z轴: 向上
        ])

        # 应用旋转得到世界坐标系中的轴方向
        rotated_axes = rotation_matrix @ axes_3d.T

        # 计算轴端点的3D世界坐标
        axis_endpoints_3d = position_3d[:, np.newaxis] + rotated_axes

        # 投影轴端点到2D图像
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # 红绿蓝
        labels = ['X(前)', 'Y(左)', 'Z(上)']

        u_orig, v_orig = origin_2d

        for i, (axis_end_3d, color, label) in enumerate(zip(axis_endpoints_3d.T, colors, labels)):
            # 投影轴端点到2D
            end_2d = project_3d_to_2d(axis_end_3d, intrinsic_matrix, extrinsic_matrix)

            if end_2d is not None:
                # 绘制箭头
                cv2.arrowedLine(image, (u_orig, v_orig), tuple(end_2d), color, 3)

                # 添加标签
                label_pos = (end_2d[0] + 5, end_2d[1])
                cv2.putText(image, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # 添加黑色边框提高可读性
                cv2.putText(image, label, (label_pos[0] + 1, label_pos[1] + 1),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)

        # 绘制原点
        cv2.circle(image, (u_orig, v_orig), 5, (255, 255, 255), -1)
        cv2.circle(image, (u_orig, v_orig), 8, (0, 0, 0), 2)

        return True

    except Exception as e:
        print(f"绘制真实坐标系失败: {e}")
        return False

def draw_coordinate_frame_simple(image, origin_2d, axis_length=50):
    """
    简化的坐标系绘制 (当无法进行3D投影时使用)
    根据常见的相机视角进行近似绘制
    """
    u, v = origin_2d

    # 根据第三人称视角的常见情况进行绘制
    # X轴 - 红色: 向右前方 (近似)
    x_end = (u + int(axis_length * 0.9), v + int(axis_length * 0.1))
    cv2.arrowedLine(image, (u, v), x_end, (0, 0, 255), 3)
    cv2.putText(image, 'X(前)', (x_end[0] + 5, x_end[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Y轴 - 绿色: 向左 (近似)
    y_end = (u - int(axis_length * 0.8), v + int(axis_length * 0.2))
    cv2.arrowedLine(image, (u, v), y_end, (0, 255, 0), 3)
    cv2.putText(image, 'Y(左)', (y_end[0] - 30, y_end[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Z轴 - 蓝色: 向上
    z_end = (u, v - axis_length)
    cv2.arrowedLine(image, (u, v), z_end, (255, 0, 0), 3)
    cv2.putText(image, 'Z(上)', (z_end[0] + 5, z_end[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # 绘制原点
    cv2.circle(image, (u, v), 5, (255, 255, 255), -1)
    cv2.circle(image, (u, v), 8, (0, 0, 0), 2)

def draw_pose_annotation(image, position_2d, quaternion, gripper_width):
    """在图像上绘制位姿标注"""
    u, v = position_2d

    # 绘制夹爪位置点
    cv2.circle(image, (u, v), 8, (0, 255, 255), -1)  # 黄色圆点
    cv2.circle(image, (u, v), 12, (0, 0, 0), 2)      # 黑色边框

    # 绘制位姿信息文本
    info_lines = [
        f"Pos: ({position_2d[0]}, {position_2d[1]})",
        f"Quat: w={quaternion[0]:.2f}",
        f"      xyz=({quaternion[1]:.2f},{quaternion[2]:.2f},{quaternion[3]:.2f})",
        f"Gripper: {gripper_width:.2f}"
    ]

    for i, text in enumerate(info_lines):
        y_offset = v - 10 + i * 15
        # 白色文字，黑色边框
        cv2.putText(image, text, (u + 16, y_offset + 1),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
        cv2.putText(image, text, (u + 15, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    return image

def draw_simple_object_marker(image, position_2d, obj_info):
    """绘制简单的物体标记点（避免中文乱码）"""
    u, v = position_2d
    color = obj_info.get('color', (0, 255, 0))  # 默认绿色

    # 绘制物体位置点 - 简洁风格
    cv2.circle(image, (u, v), 8, color, -1)  # 彩色实心圆
    cv2.circle(image, (u, v), 10, (255, 255, 255), 2)  # 白色边框
    cv2.circle(image, (u, v), 12, (0, 0, 0), 1)       # 黑色外边框

    # 只显示英文标签，避免中文乱码
    obj_type = obj_info.get('type', 'object')
    name = obj_info.get('name', 'unknown')

    # 简化标签
    if 'carrot' in name.lower() or 'source' in obj_type:
        label = "CARROT"
        label_color = (0, 165, 255)  # 橙色
    elif 'plate' in name.lower() or 'target' in obj_type:
        label = "PLATE"
        label_color = (255, 0, 0)  # 蓝色
    else:
        label = "OBJECT"
        label_color = color

    # 绘制标签
    label_x = u + 15
    label_y = v - 10

    # 确保标签不超出图像边界
    if label_x + len(label) * 8 > image.shape[1]:
        label_x = u - len(label) * 8 - 5
    if label_y < 20:
        label_y = v + 25

    # 绘制标签背景
    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
    cv2.rectangle(image, (label_x - 3, label_y - text_size[1] - 3),
                 (label_x + text_size[0] + 3, label_y + 3), (0, 0, 0), -1)

    # 绘制标签文字
    cv2.putText(image, label, (label_x, label_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 2)

    return image

def get_robot_pose_and_image_from_env(env, obs):
    """从现有环境和观察中获取位姿和图像数据"""
    if "extra" in obs and "tcp_pose" in obs["extra"]:
        tcp_pose = obs["extra"]["tcp_pose"]
        position = tcp_pose[:3]        # 位置 (x, y, z)
        quaternion = tcp_pose[3:7]     # 四元数 (w, x, y, z)

        # 获取夹爪状态
        gripper_width = 0.5
        try:
            agent = env.unwrapped.agent if hasattr(env, 'unwrapped') else env.agent
            if hasattr(agent, 'gripper_closedness'):
                closedness = agent.gripper_closedness
                gripper_width = max(0.0, min(1.0, 1.0 - closedness))
        except Exception:
            pass

        # 获取图像
        image = None
        camera_name = "unknown"

        if "image" in obs:
            cam_imgs = obs["image"]
            if "3rd_view_camera" in cam_imgs and "rgb" in cam_imgs["3rd_view_camera"]:
                image = cam_imgs["3rd_view_camera"]["rgb"]
                camera_name = "3rd_view_camera"
            elif "base_camera" in cam_imgs and "rgb" in cam_imgs["base_camera"]:
                image = cam_imgs["base_camera"]["rgb"]
                camera_name = "base_camera"

        if image is None:
            image = get_image_from_maniskill2_obs_dict(env, obs)
            camera_name = "default"

        # 获取相机参数
        camera_params = obs.get("camera_param", None)

        # 获取物体坐标信息
        objects_info = get_object_coordinates_from_env(env, obs)

        return {
            'position': position,
            'quaternion': quaternion,
            'gripper_width': gripper_width,
            'image': image,
            'camera_params': camera_params,
            'camera_name': camera_name,
            'objects_info': objects_info
        }
    else:
        raise KeyError(f"未在obs中找到tcp_pose")

def get_object_coordinates_from_env(env, obs, reset_info=None):
    """从环境中获取目标物体的3D坐标信息"""
    objects_info = {}

    try:
        unwrapped_env = env.unwrapped if hasattr(env, 'unwrapped') else env

        # 方法1: 从环境属性获取目标物体当前位姿
        if hasattr(unwrapped_env, 'target_obj_pose'):
            print("从环境属性获取目标物体当前位姿")
            target_pose = unwrapped_env.target_obj_pose

            # 处理不同的位姿格式
            position = None
            quaternion = None

            if hasattr(target_pose, 'p') and hasattr(target_pose, 'q'):
                # Sapien Pose对象
                position = np.array(target_pose.p)
                # 处理四元数 - 可能是对象或numpy数组
                if hasattr(target_pose.q, 'w'):
                    quaternion = np.array([target_pose.q.w, target_pose.q.x, target_pose.q.y, target_pose.q.z])
                else:
                    # 假设是numpy数组格式 [w, x, y, z]
                    quaternion = np.array(target_pose.q)
            elif isinstance(target_pose, np.ndarray) and len(target_pose) >= 7:
                # numpy数组格式 [x, y, z, qw, qx, qy, qz]
                position = target_pose[:3]
                quaternion = target_pose[3:7]
            elif hasattr(target_pose, '__len__') and len(target_pose) >= 7:
                # 列表或其他序列格式
                position = np.array(target_pose[:3])
                quaternion = np.array(target_pose[3:7])

            if position is not None and quaternion is not None:
                target_name = getattr(unwrapped_env, 'episode_target_obj_name', 'target_plate')
                objects_info['target_object_current'] = {
                    'name': target_name,
                    'position': position,
                    'quaternion': quaternion,
                    'type': 'target_current',
                    'color': (255, 0, 255),  # 紫色 - 目标物体当前位置
                    'description': f'Target({target_name}) current'
                }

        # 方法2: 获取源物体（如胡萝卜）当前位置
        if hasattr(unwrapped_env, 'episode_target_obj'):
            print("2.获取源物体（如胡萝卜）当前位置")
            target_obj = unwrapped_env.episode_target_obj
            if hasattr(target_obj, 'get_pose'):
                pose = target_obj.get_pose()
                position = np.array(pose.p)
                # 处理四元数 - 可能是对象或numpy数组
                if hasattr(pose.q, 'w'):
                    quaternion = np.array([pose.q.w, pose.q.x, pose.q.y, pose.q.z])
                else:
                    # 假设是numpy数组格式 [w, x, y, z]
                    quaternion = np.array(pose.q)
                obj_name = getattr(target_obj, 'name', 'target_object')
                objects_info['episode_target_obj'] = {
                    'name': obj_name,
                    'position': position,
                    'quaternion': quaternion,
                    'type': 'episode_target',
                    'color': (0, 255, 255),  # 青色 - episode目标物体
                    'description': f'Episode目标物体({obj_name})'
                }

        # 方法3: 从reset_info获取物体初始位姿
        if reset_info is None:
            print("3.从reset_info获取物体初始位姿")
            # 尝试从环境获取reset_info
            if hasattr(unwrapped_env, '_last_reset_info'):
                reset_info = unwrapped_env._last_reset_info
            else:
                reset_info = getattr(unwrapped_env, 'reset_info', None)

        if reset_info and isinstance(reset_info, dict):
            # 获取源物体（胡萝卜）初始位置
            if 'episode_source_obj_init_pose_wrt_robot_base' in reset_info:
                source_pose = reset_info['episode_source_obj_init_pose_wrt_robot_base']

                # 处理不同的位姿格式
                position = None
                quaternion = None

                if hasattr(source_pose, 'p') and hasattr(source_pose, 'q'):
                    # Sapien Pose对象
                    position = np.array(source_pose.p)
                    # 处理四元数 - 可能是对象或numpy数组
                    if hasattr(source_pose.q, 'w'):
                        quaternion = np.array([source_pose.q.w, source_pose.q.x, source_pose.q.y, source_pose.q.z])
                    else:
                        # 假设是numpy数组格式 [w, x, y, z]
                        quaternion = np.array(source_pose.q)
                elif isinstance(source_pose, np.ndarray) and len(source_pose) >= 7:
                    # numpy数组格式 [x, y, z, qw, qx, qy, qz]
                    position = source_pose[:3]
                    quaternion = source_pose[3:7]
                elif hasattr(source_pose, '__len__') and len(source_pose) >= 7:
                    # 列表或其他序列格式
                    position = np.array(source_pose[:3])
                    quaternion = np.array(source_pose[3:7])

                if position is not None and quaternion is not None:
                    source_name = reset_info.get('episode_source_obj_name', 'carrot')
                    objects_info['source_object_init'] = {
                        'name': source_name,
                        'position': position,
                        'quaternion': quaternion,
                        'type': 'source_init',
                        'color': (0, 165, 255),  # 橙色 - 源物体初始位置
                        'description': f'Source({source_name}) init'
                    }

            # 获取目标物体（盘子）初始位置
            if 'episode_target_obj_init_pose_wrt_robot_base' in reset_info:
                target_pose = reset_info['episode_target_obj_init_pose_wrt_robot_base']

                # 处理不同的位姿格式
                position = None
                quaternion = None

                if hasattr(target_pose, 'p') and hasattr(target_pose, 'q'):
                    # Sapien Pose对象
                    position = np.array(target_pose.p)
                    # 处理四元数 - 可能是对象或numpy数组
                    if hasattr(target_pose.q, 'w'):
                        quaternion = np.array([target_pose.q.w, target_pose.q.x, target_pose.q.y, target_pose.q.z])
                    else:
                        # 假设是numpy数组格式 [w, x, y, z]
                        quaternion = np.array(target_pose.q)
                elif isinstance(target_pose, np.ndarray) and len(target_pose) >= 7:
                    # numpy数组格式 [x, y, z, qw, qx, qy, qz]
                    position = target_pose[:3]
                    quaternion = target_pose[3:7]
                elif hasattr(target_pose, '__len__') and len(target_pose) >= 7:
                    # 列表或其他序列格式
                    position = np.array(target_pose[:3])
                    quaternion = np.array(target_pose[3:7])

                if position is not None and quaternion is not None:
                    target_name = reset_info.get('episode_target_obj_name', 'plate')
                    objects_info['target_object_init'] = {
                        'name': target_name,
                        'position': position,
                        'quaternion': quaternion,
                        'type': 'target_init',
                        'color': (255, 0, 0),  # 蓝色 - 目标物体初始位置
                        'description': f'Target({target_name}) init'
                    }

        # 方法4: 尝试通过场景获取所有相关物体
        if hasattr(unwrapped_env, 'scene'):
            print("4.尝试通过场景获取所有相关物体")
            scene = unwrapped_env.scene
            if hasattr(scene, 'get_all_actors'):
                actors = scene.get_all_actors()
                for actor in actors:
                    name = getattr(actor, 'name', '')
                    # 查找包含关键词的物体
                    if any(keyword in name.lower() for keyword in ['carrot', 'plate', 'target', 'obj']):
                        try:
                            pose = actor.get_pose()
                            position = np.array(pose.p)
                            # 处理四元数 - 可能是对象或numpy数组
                            if hasattr(pose.q, 'w'):
                                quaternion = np.array([pose.q.w, pose.q.x, pose.q.y, pose.q.z])
                            else:
                                # 假设是numpy数组格式 [w, x, y, z]
                                quaternion = np.array(pose.q)
                            objects_info[f'scene_object_{name}'] = {
                                'name': name,
                                'position': position,
                                'quaternion': quaternion,
                                'type': 'scene_object',
                                'color': (0, 255, 0),  # 绿色 - 场景物体
                                'description': f'场景物体({name})'
                            }
                        except Exception:
                            continue

        return objects_info

    except Exception as e:
        print(f"获取物体坐标失败: {e}")
        return {}

def draw_simple_gripper_marker(image, position_2d):
    """绘制简单的夹爪标记点（无复杂坐标系）"""
    u, v = position_2d

    # 绘制简单的夹爪标记
    cv2.circle(image, (u, v), 10, (0, 255, 255), -1)  # 青色实心圆
    cv2.circle(image, (u, v), 12, (255, 255, 255), 2)  # 白色边框
    cv2.circle(image, (u, v), 14, (0, 0, 0), 1)       # 黑色外边框

    # 简单标签
    label = "GRIPPER"
    label_color = (0, 255, 255)  # 青色

    # 绘制标签
    label_x = u + 18
    label_y = v - 12

    # 确保标签不超出图像边界
    if label_x + len(label) * 8 > image.shape[1]:
        label_x = u - len(label) * 8 - 5
    if label_y < 20:
        label_y = v + 30

    # 绘制标签背景
    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
    cv2.rectangle(image, (label_x - 3, label_y - text_size[1] - 3),
                 (label_x + text_size[0] + 3, label_y + 3), (0, 0, 0), -1)

    # 绘制标签文字
    cv2.putText(image, label, (label_x, label_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 2)

    return image

def create_simple_annotated_image(data, step_name, save_name=None, include_objects=True):
    """创建简化的标注图像（无复杂坐标系）"""
    image = data['image'].copy()
    h, w = image.shape[:2]

    # 获取相机参数
    intrinsic = None
    extrinsic = None
    use_real_projection = False

    if data['camera_params'] and data['camera_name'] in data['camera_params']:
        try:
            params = data['camera_params'][data['camera_name']]
            intrinsic = params['intrinsic_cv']
            extrinsic = params['extrinsic_cv']
            use_real_projection = True
        except Exception:
            pass

    # 1. 处理夹爪位姿标注 - 简化版本
    gripper_position_2d = None
    if use_real_projection:
        gripper_position_2d = project_3d_to_2d(
            data['position'], intrinsic, extrinsic
        )

    # 如果投影失败，使用图像中心
    if gripper_position_2d is None:
        gripper_position_2d = [w // 2, h // 2]

    # 检查位置是否在图像范围内
    if not (0 <= gripper_position_2d[0] < w and 0 <= gripper_position_2d[1] < h):
        gripper_position_2d = [w // 2, h // 2]

    # 绘制标注
    annotated_image = image.copy()

    # 绘制夹爪朝向箭头（统一风格）
    if use_real_projection:
        success = draw_affordance_arrow(
            annotated_image, gripper_position_2d, data['position'], data['quaternion'],
            intrinsic, extrinsic, arrow_length=0.08, arrow_color=(0, 255, 0),
            arrow_thickness=3, show_point=True
        )
        if not success:
            draw_affordance_arrow_simple(
                annotated_image, gripper_position_2d, data['quaternion'],
                arrow_length=60, arrow_color=(0, 255, 0), arrow_thickness=3, show_point=True
            )
    else:
        draw_affordance_arrow_simple(
            annotated_image, gripper_position_2d, data['quaternion'],
            arrow_length=60, arrow_color=(0, 255, 0), arrow_thickness=3, show_point=True
        )

    # 绘制夹爪位姿信息
    annotated_image = draw_pose_annotation(
        annotated_image, gripper_position_2d,
        data['quaternion'], data['gripper_width']
    )

    # 2. 处理物体标注 - 简化版本
    if include_objects and 'objects_info' in data:
        objects_info = data['objects_info']

        for obj_key, obj_info in objects_info.items():
            obj_position_3d = obj_info['position']

            # 投影物体位置到2D
            obj_position_2d = None
            if use_real_projection:
                obj_position_2d = project_3d_to_2d(
                    obj_position_3d, intrinsic, extrinsic
                )

            # 如果投影失败或位置超出范围，跳过此物体
            if (obj_position_2d is None or
                not (0 <= obj_position_2d[0] < w and 0 <= obj_position_2d[1] < h)):
                continue

            # 绘制简化的物体标记
            draw_simple_object_marker(annotated_image, obj_position_2d, obj_info)

    # 3. 添加步骤标签
    cv2.putText(annotated_image, f"{step_name}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(annotated_image, f"{step_name}", (11, 31),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)

    # 4. 添加物体数量信息
    if include_objects and 'objects_info' in data:
        obj_count = len(data['objects_info'])
        obj_text = f"Objects: {obj_count}"
        cv2.putText(annotated_image, obj_text, (10, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(annotated_image, obj_text, (11, h - 19),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    # 5. 保存图像
    if save_name:
        cv2.imwrite(save_name, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

    return annotated_image

def create_annotated_image(data, step_name, save_name=None, include_objects=True):
    """创建单个标注图像，支持夹爪和物体标注"""
    image = data['image'].copy()
    h, w = image.shape[:2]

    # 获取相机参数
    intrinsic = None
    extrinsic = None
    use_real_projection = False

    if data['camera_params'] and data['camera_name'] in data['camera_params']:
        try:
            params = data['camera_params'][data['camera_name']]
            intrinsic = params['intrinsic_cv']
            extrinsic = params['extrinsic_cv']
            use_real_projection = True
        except Exception:
            pass

    # 1. 处理夹爪位姿标注
    gripper_position_2d = None
    if use_real_projection:
        gripper_position_2d = project_3d_to_2d(
            data['position'], intrinsic, extrinsic
        )

    # 如果投影失败，使用图像中心
    if gripper_position_2d is None:
        gripper_position_2d = [w // 2, h // 2]

    # 检查位置是否在图像范围内
    if not (0 <= gripper_position_2d[0] < w and 0 <= gripper_position_2d[1] < h):
        gripper_position_2d = [w // 2, h // 2]

    # 绘制标注
    annotated_image = image.copy()

    # 绘制夹爪朝向箭头（统一风格）
    if use_real_projection:
        success = draw_affordance_arrow(
            annotated_image, gripper_position_2d, data['position'], data['quaternion'],
            intrinsic, extrinsic, arrow_length=0.08, arrow_color=(0, 255, 0),
            arrow_thickness=3, show_point=True
        )
        if not success:
            draw_affordance_arrow_simple(
                annotated_image, gripper_position_2d, data['quaternion'],
                arrow_length=60, arrow_color=(0, 255, 0), arrow_thickness=3, show_point=True
            )
    else:
        draw_affordance_arrow_simple(
            annotated_image, gripper_position_2d, data['quaternion'],
            arrow_length=60, arrow_color=(0, 255, 0), arrow_thickness=3, show_point=True
        )

    # 绘制夹爪位姿信息
    annotated_image = draw_pose_annotation(
        annotated_image, gripper_position_2d,
        data['quaternion'], data['gripper_width']
    )

    # 2. 处理物体标注 - 简化版本
    if include_objects and 'objects_info' in data:
        objects_info = data['objects_info']

        for obj_key, obj_info in objects_info.items():
            obj_position_3d = obj_info['position']

            # 投影物体位置到2D
            obj_position_2d = None
            if use_real_projection:
                obj_position_2d = project_3d_to_2d(
                    obj_position_3d, intrinsic, extrinsic
                )

            # 如果投影失败或位置超出范围，跳过此物体
            if (obj_position_2d is None or
                not (0 <= obj_position_2d[0] < w and 0 <= obj_position_2d[1] < h)):
                continue

            # 绘制简化的物体标记
            draw_simple_object_marker(annotated_image, obj_position_2d, obj_info)

    # 3. 添加步骤标签
    cv2.putText(annotated_image, f"{step_name}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(annotated_image, f"{step_name}", (11, 31),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)

    # 4. 添加夹爪3D位置信息
    pos_text = f"Gripper 3D: ({data['position'][0]:.3f}, {data['position'][1]:.3f}, {data['position'][2]:.3f})"
    cv2.putText(annotated_image, pos_text, (10, h - 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(annotated_image, pos_text, (11, h - 19),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    # 5. 添加物体数量信息
    if include_objects and 'objects_info' in data:
        obj_count = len(data['objects_info'])
        obj_text = f"Objects: {obj_count}"
        cv2.putText(annotated_image, obj_text, (10, h - 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(annotated_image, obj_text, (11, h - 39),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    # 6. 保存图像
    if save_name:
        cv2.imwrite(save_name, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

    return annotated_image

def test_multiple_gripper_positions():
    """测试多个不同的夹爪位置"""
    print("=== 测试不同夹爪位置的标注效果 ===\n")

    task_name = "widowx_carrot_on_plate"
    env = simpler_env.make(task_name)

    try:
        # 初始状态
        obs, reset_info = env.reset(seed=42)
        initial_data = get_robot_pose_and_image_from_env(env, obs)

        # 检查动作空间
        print("环境信息:")
        print(f"   动作空间: {env.action_space}")
        print(f"   动作空间维度: {env.action_space.shape}")

        print("\n1. 初始位置:")
        print(f"   3D位置: {initial_data['position']}")

        # 定义多个测试动作 (7维动作向量: [x, y, z, rx, ry, rz, gripper])
        # 使用较小的动作值以避免段错误
        test_actions = [
            {
                'name': '初始位置',
                'action': None,
                'description': '环境重置后的初始位置'
            },
            {
                'name': '向上移动',
                'action': np.array([0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
                'description': '夹爪向上移动5cm'
            },
            {
                'name': '向右移动',
                'action': np.array([0.0, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
                'description': '夹爪向右移动5cm'
            },
            {
                'name': '向前移动',
                'action': np.array([0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
                'description': '夹爪向前移动5cm'
            },
            {
                'name': '向左移动',
                'action': np.array([0.0, -0.05, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
                'description': '夹爪向左移动5cm'
            },
            {
                'name': '轻微旋转',
                'action': np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0], dtype=np.float32),
                'description': '夹爪轻微旋转'
            }
        ]

        annotated_images = []
        position_data = []

        # 处理初始位置
        initial_annotated = create_annotated_image(initial_data, test_actions[0]['name'],
                                                   'gripper_pos_0_initial.png')
        annotated_images.append(initial_annotated)
        position_data.append({
            'name': test_actions[0]['name'],
            'position': initial_data['position'].copy(),
            'quaternion': initial_data['quaternion'].copy()
        })

        print(f"   保存: gripper_pos_0_initial.png")

        # 执行每个动作
        for i, action_info in enumerate(test_actions[1:], 1):
            print(f"\n{i+1}. {action_info['description']}:")

            try:
                # 执行动作
                action = action_info['action']
                print(f"   动作向量维度: {action.shape}")
                print(f"   动作向量: {action}")
                print(f"   动作范围检查: min={action.min():.3f}, max={action.max():.3f}")

                # 确保动作在合法范围内
                action = np.clip(action, env.action_space.low, env.action_space.high)

                print("   执行环境步进...")
                obs, reward, terminated, truncated, info = env.step(action)
                print("   ✓ 环境步进成功")

                # 获取新的位姿数据
                print("   获取新位姿数据...")
                new_data = get_robot_pose_and_image_from_env(env, obs)

                print(f"   新3D位置: {new_data['position']}")
                print(f"   位置变化: {new_data['position'] - initial_data['position']}")

                # 创建标注图像
                filename = f'gripper_pos_{i}_{action_info["name"].replace(" ", "_").lower()}.png'
                print(f"   创建标注图像: {filename}")
                annotated_img = create_annotated_image(new_data, action_info['name'], filename)
                annotated_images.append(annotated_img)

                position_data.append({
                    'name': action_info['name'],
                    'position': new_data['position'].copy(),
                    'quaternion': new_data['quaternion'].copy()
                })

                print(f"   ✓ 保存: {filename}")

                # 检查是否需要终止
                if terminated or truncated:
                    print("   ⚠ 环境指示终止，重置环境...")
                    obs, reset_info = env.reset(seed=42)

            except Exception as e:
                print(f"   ❌ 执行动作失败: {e}")
                print("   尝试重置环境...")
                try:
                    obs, reset_info = env.reset(seed=42)
                    print("   ✓ 环境重置成功")
                except Exception as reset_e:
                    print(f"   ❌ 环境重置也失败: {reset_e}")
                    print("   跳过此动作，继续下一个...")
                    continue

        # 创建网格对比图
        print(f"\n=== 创建对比图 ({len(annotated_images)} 张图像) ===")

        # 计算网格布局
        num_images = len(annotated_images)
        cols = 4  # 每行4张图
        rows = (num_images + cols - 1) // cols

        if num_images > 0:
            h, w = annotated_images[0].shape[:2]

            # 创建网格图像
            grid_h = h * rows
            grid_w = w * cols
            grid_image = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

            for i, img in enumerate(annotated_images):
                row = i // cols
                col = i % cols

                y_start = row * h
                y_end = y_start + h
                x_start = col * w
                x_end = x_start + w

                grid_image[y_start:y_end, x_start:x_end] = img

            # 保存网格对比图
            cv2.imwrite('gripper_positions_grid_comparison.png',
                       cv2.cvtColor(grid_image, cv2.COLOR_RGB2BGR))
            print("保存网格对比图: gripper_positions_grid_comparison.png")

        # 创建水平对比图（选择前6张）
        if len(annotated_images) >= 2:
            selected_count = min(6, len(annotated_images))
            selected_images = annotated_images[:selected_count]

            # 调整图像大小
            target_h = 300  # 统一高度
            resized_images = []
            for img in selected_images:
                h_orig, w_orig = img.shape[:2]
                target_w = int(w_orig * target_h / h_orig)
                resized = cv2.resize(img, (target_w, target_h))
                resized_images.append(resized)

            # 水平拼接
            comparison_horizontal = np.hstack(resized_images)
            cv2.imwrite('gripper_positions_horizontal.png',
                       cv2.cvtColor(comparison_horizontal, cv2.COLOR_RGB2BGR))
            print("保存水平对比图: gripper_positions_horizontal.png")

        # 打印位置统计
        print(f"\n=== 位置变化统计 ===")
        initial_pos = position_data[0]['position']
        for i, data in enumerate(position_data):
            if i == 0:
                print(f"{i+1}. {data['name']}: 基准位置 {data['position']}")
            else:
                diff = data['position'] - initial_pos
                distance = np.linalg.norm(diff)
                print(f"{i+1}. {data['name']}: 位移 {diff} (距离: {distance:.3f}m)")

        print(f"\n=== 测试完成 ===")
        print("生成的文件:")
        for i, action_info in enumerate(test_actions):
            if i == 0:
                print(f"  - gripper_pos_0_initial.png: {action_info['description']}")
            else:
                filename = f'gripper_pos_{i}_{action_info["name"].replace(" ", "_").lower()}.png'
                print(f"  - {filename}: {action_info['description']}")
        print("  - gripper_positions_grid_comparison.png: 网格对比图")
        print("  - gripper_positions_horizontal.png: 水平对比图")

        return annotated_images

    except Exception as e:
        print(f"❌ 测试过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return None

    finally:
        env.close()

def annotate_pose_on_image():
    """主函数：在图像上标注机械臂位姿"""
    print("=== 在图像上标注机械臂6D位姿 (坐标系修正版) ===\n")

    try:
        # 1. 获取位姿和图像数据
        data = get_robot_pose_and_image()

        # 2. 准备标注
        image = data['image'].copy()
        h, w = image.shape[:2]

        # 3. 尝试进行3D到2D投影
        position_2d = None
        use_real_projection = False

        if data['camera_params'] and data['camera_name'] in data['camera_params']:
            print("\n✓ 尝试3D到2D投影:")
            try:
                params = data['camera_params'][data['camera_name']]
                intrinsic = params['intrinsic_cv']
                extrinsic = params['extrinsic_cv']

                position_2d = project_3d_to_2d(
                    data['position'], intrinsic, extrinsic
                )

                if position_2d:
                    print(f"  投影成功: 夹爪位置 -> 图像坐标 {position_2d}")
                    use_real_projection = True
                else:
                    print("  投影失败: 夹爪不在相机视野内")
            except Exception as e:
                print(f"  投影失败: {e}")

        # 4. 如果投影失败，使用图像中心
        if position_2d is None:
            position_2d = [w // 2, h // 2]
            print(f"\n⚠ 使用图像中心作为标注位置: {position_2d}")

        # 5. 检查位置是否在图像范围内
        if not (0 <= position_2d[0] < w and 0 <= position_2d[1] < h):
            position_2d = [w // 2, h // 2]
            print(f"⚠ 位置超出图像范围，调整到图像中心")

        # 6. 绘制标注
        print("\n✓ 绘制位姿标注:")
        annotated_image = image.copy()

        # 尝试绘制真实的坐标系
        if use_real_projection:
            success = draw_coordinate_frame_correct(
                annotated_image, position_2d, data['position'], data['quaternion'],
                intrinsic, extrinsic
            )
            if success:
                print("  使用真实3D投影绘制坐标系")
            else:
                print("  真实投影失败，使用简化坐标系")
                draw_coordinate_frame_simple(annotated_image, position_2d)
        else:
            print("  使用简化坐标系绘制")
            draw_coordinate_frame_simple(annotated_image, position_2d)

        # 绘制位姿信息
        annotated_image = draw_pose_annotation(
            annotated_image, position_2d,
            data['quaternion'], data['gripper_width']
        )

        # 7. 保存结果
        output_path = 'robot_pose_annotation_corrected.png'
        cv2.imwrite(output_path, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
        print(f"  保存标注图像: {output_path}")

        # 8. 创建对比图
        comparison_image = np.hstack([image, annotated_image])
        comparison_path = 'pose_annotation_comparison_corrected.png'
        cv2.imwrite(comparison_path, cv2.cvtColor(comparison_image, cv2.COLOR_RGB2BGR))
        print(f"  保存对比图: {comparison_path}")

        # 9. 清理
        data['env'].close()

        return annotated_image

    except Exception as e:
        print(f"❌ 标注过程出错: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """主函数"""
    print("机械臂末端6D位姿获取和图像标注演示 (坐标系修正版)")
    print("=" * 60)

    print("\n=== 坐标系约定说明 ===")
    print("ROS/ManiSkill标准 (右手坐标系):")
    print("  - X轴 (红色): 向前")
    print("  - Y轴 (绿色): 向左")
    print("  - Z轴 (蓝色): 向上")
    print("\n注意: 在2D图像中的显示会根据相机角度而变化")

    print("\n=== 选择测试模式 ===")
    print("1. 单次位姿标注（原始功能）")
    print("2. 多位置测试（推荐）- 测试不同手抓位置的标注效果")
    print("3. 安全模式测试 - 使用极小动作值避免段错误")
    print("4. 物体标注测试 - 测试物体3D坐标获取和标注功能")

    try:
        choice = input("\n请选择测试模式 (1, 2, 3 或 4，默认为4): ").strip()
        if not choice:
            choice = "4"
    except:
        choice = "4"

    if choice == "1":
        print("\n=== 执行单次位姿标注 ===")
        # 运行单次位姿标注
        result = annotate_pose_on_image()

        if result is not None:
            print("\n✅ 位姿标注完成！")
            print("\n生成的文件：")
            print("  - robot_pose_annotation_corrected.png: 修正坐标系的标注图像")
            print("  - pose_annotation_comparison_corrected.png: 原始图像与标注图像对比")

            print("\n=== 改进说明 ===")
            print("1. ✓ 使用正确的ROS坐标系约定 (X前, Y左, Z上)")
            print("2. ✓ 尝试使用真实的3D到2D投影绘制坐标轴")
            print("3. ✓ 根据四元数计算实际的轴方向")
            print("4. ✓ 提供简化版本作为备用方案")
            print("5. ✓ 添加坐标轴标签说明方向")
        else:
            print("\n❌ 位姿标注失败，请检查环境配置")

    elif choice == "2":
        print("\n=== 执行多位置测试 ===")
        # 运行多位置测试
        results = test_multiple_gripper_positions()

        if results is not None:
            print("\n✅ 多位置测试完成！")
            print("\n=== 测试总结 ===")
            print(f"✓ 共测试了 {len(results)} 个不同的手抓位置")
            print("✓ 生成了对应的标注图像")
            print("✓ 创建了网格和水平对比图")
            print("✓ 输出了位置变化统计信息")

            print("\n=== 观察重点 ===")
            print("1. 🔍 观察手抓位置变化是否准确反映在图像标注中")
            print("2. 🔍 检查3D到2D投影的准确性")
            print("3. 🔍 验证坐标系方向标注是否正确")
            print("4. 🔍 确认不同位置下的姿态变化")
            print("5. 🔍 对比网格图中的标注一致性")

            print("\n💡 建议: 查看生成的对比图来分析标注效果")
        else:
            print("\n❌ 多位置测试失败，请检查环境配置")

    elif choice == "4":
        print("\n=== Simple Object Annotation Test ===")
        # 运行简化的物体标注测试
        result = test_simple_object_annotation()

        if result is not None:
            print("\n✅ Object annotation test completed!")
            print("\n=== Test Summary ===")
            print("✓ Successfully obtained 3D coordinates of objects")
            print("✓ Implemented 3D to 2D projection for objects")
            print("✓ Created images with object annotations")
            print("✓ Generated comparison images")

            print("\n=== Key Points to Observe ===")
            print("1. 🔍 Check if object 3D coordinates are accurate")
            print("2. 🔍 Verify object to 2D image projection is correct")
            print("3. 🔍 Observe different colors for different object types")
            print("4. 🔍 Confirm object annotations work with gripper annotations")

            print("\n💡 Tip: Check the generated comparison images to analyze annotation effects")
        else:
            print("\n❌ Object annotation test failed, please check environment configuration")

    else:  # choice == "3" 或其他情况
        print("\n=== 执行安全模式测试 ===")
        # 运行安全模式测试
        results = safe_test()

        if results is not None:
            print("\n✅ 安全模式测试完成！")
            print("\n=== 测试总结 ===")
            print(f"✓ 共测试了 {len(results)} 个位置")
            print("✓ 使用了极小的动作值避免段错误")
            print("✓ 生成了标注图像和对比图")

            print("\n=== 观察重点 ===")
            print("1. 🔍 验证标注系统的基本功能")
            print("2. 🔍 检查微小位移是否能被正确标注")
            print("3. 🔍 确认3D到2D投影的准确性")
            print("4. 🔍 验证坐标系标注的正确性")

            print("\n💡 建议: 如果安全模式成功，可以尝试模式2的完整测试")
        else:
            print("\n❌ 安全模式测试失败，请检查环境配置")

def safe_test():
    """安全测试函数 - 使用非常小的动作值"""
    print("=== 安全模式测试（小动作值） ===\n")

    task_name = "widowx_carrot_on_plate"
    env = simpler_env.make(task_name)

    try:
        # 初始状态
        obs, reset_info = env.reset(seed=42)
        initial_data = get_robot_pose_and_image_from_env(env, obs)

        print("环境信息:")
        print(f"   动作空间: {env.action_space}")
        print(f"   动作空间维度: {env.action_space.shape}")
        print(f"   动作范围: {env.action_space.low} 到 {env.action_space.high}")

        print("\n1. 初始位置:")
        print(f"   3D位置: {initial_data['position']}")

        # 非常保守的动作
        safe_actions = [
            {
                'name': '初始位置',
                'action': None,
                'description': '环境重置后的初始位置'
            },
            {
                'name': '微小上移',
                'action': np.array([0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
                'description': '夹爪微小向上移动1cm'
            },
            {
                'name': '微小右移',
                'action': np.array([0.0, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
                'description': '夹爪微小向右移动1cm'
            }
        ]

        annotated_images = []

        # 处理初始位置
        initial_annotated = create_annotated_image(initial_data, safe_actions[0]['name'],
                                                   'safe_gripper_pos_0_initial.png')
        annotated_images.append(initial_annotated)
        print(f"   保存: safe_gripper_pos_0_initial.png")

        # 执行安全动作
        for i, action_info in enumerate(safe_actions[1:], 1):
            print(f"\n{i+1}. {action_info['description']}:")

            try:
                action = action_info['action']
                print(f"   动作向量: {action}")

                # 执行动作
                obs, reward, terminated, truncated, info = env.step(action)
                print("   ✓ 步进成功")

                # 获取新数据
                new_data = get_robot_pose_and_image_from_env(env, obs)
                print(f"   新位置: {new_data['position']}")
                print(f"   位移: {new_data['position'] - initial_data['position']}")

                # 创建图像
                filename = f'safe_gripper_pos_{i}_{action_info["name"].replace(" ", "_")}.png'
                annotated_img = create_annotated_image(new_data, action_info['name'], filename)
                annotated_images.append(annotated_img)
                print(f"   ✓ 保存: {filename}")

            except Exception as e:
                print(f"   ❌ 失败: {e}")
                break

        # 创建简单对比图
        if len(annotated_images) >= 2:
            comparison = np.hstack(annotated_images[:3])  # 最多3张
            cv2.imwrite('safe_comparison.png', cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
            print(f"\n✓ 保存对比图: safe_comparison.png")

        print(f"\n✅ 安全测试完成，生成 {len(annotated_images)} 张图像")
        return annotated_images

    except Exception as e:
        print(f"❌ 安全测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None

    finally:
        env.close()

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
    import copy

    # 深拷贝观测以避免修改原始数据
    obs_with_affordance = copy.deepcopy(obs)

    try:
        # 获取位姿信息
        data = get_robot_pose_and_image_from_env(env, obs)

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
                                data['position'], intrinsic, extrinsic
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
                        draw_affordance_arrow(
                            image, position_2d, data['position'],
                            data['quaternion'], intrinsic, extrinsic,
                            arrow_length, arrow_color, arrow_thickness, show_point
                        )
                    else:
                        draw_affordance_arrow_simple(
                            image, position_2d, data['quaternion'],
                            60, arrow_color, arrow_thickness, show_point
                        )

        return obs_with_affordance

    except Exception as e:
        print(f"添加affordance失败: {e}")
        return obs_with_affordance

def test_affordance_visualization():
    """测试affordance可视化效果"""
    print("=== 测试Affordance可视化 ===\n")

    task_name = "widowx_carrot_on_plate"
    env = simpler_env.make(task_name)

    try:
        obs, reset_info = env.reset(seed=42)

        print("测试不同的可视化选项:")

        # 测试配置
        test_configs = [
            {
                'name': '原始图像',
                'add_affordance': False,
                'filename': 'test_original.png'
            },
            {
                'name': 'Affordance-绿色箭头',
                'add_affordance': True,
                'arrow_color': (0, 255, 0),  # 绿色
                'arrow_thickness': 3,
                'show_point': True,
                'filename': 'test_affordance_green.png'
            },
            {
                'name': 'Affordance-红色粗箭头',
                'add_affordance': True,
                'arrow_color': (0, 0, 255),  # 红色
                'arrow_thickness': 5,
                'show_point': True,
                'filename': 'test_affordance_red_thick.png'
            },
            {
                'name': 'Affordance-蓝色细箭头（无点）',
                'add_affordance': True,
                'arrow_color': (255, 0, 0),  # 蓝色
                'arrow_thickness': 2,
                'show_point': False,
                'filename': 'test_affordance_blue_thin.png'
            }
        ]

        saved_images = []

        for i, config in enumerate(test_configs):
            print(f"\n{i+1}. {config['name']}:")

            if config['add_affordance']:
                # 添加affordance
                obs_with_aff = add_affordance_to_observation(
                    obs, env,
                    arrow_color=config['arrow_color'],
                    arrow_thickness=config['arrow_thickness'],
                    show_point=config['show_point']
                )

                # 获取图像
                if "image" in obs_with_aff:
                    cam_imgs = obs_with_aff["image"]
                    if "3rd_view_camera" in cam_imgs and "rgb" in cam_imgs["3rd_view_camera"]:
                        image = cam_imgs["3rd_view_camera"]["rgb"]
                    elif "base_camera" in cam_imgs and "rgb" in cam_imgs["base_camera"]:
                        image = cam_imgs["base_camera"]["rgb"]
                    else:
                        image = get_image_from_maniskill2_obs_dict(env, obs_with_aff)
                else:
                    image = get_image_from_maniskill2_obs_dict(env, obs_with_aff)
            else:
                # 原始图像
                if "image" in obs:
                    cam_imgs = obs["image"]
                    if "3rd_view_camera" in cam_imgs and "rgb" in cam_imgs["3rd_view_camera"]:
                        image = cam_imgs["3rd_view_camera"]["rgb"]
                    elif "base_camera" in cam_imgs and "rgb" in cam_imgs["base_camera"]:
                        image = cam_imgs["base_camera"]["rgb"]
                    else:
                        image = get_image_from_maniskill2_obs_dict(env, obs)
                else:
                    image = get_image_from_maniskill2_obs_dict(env, obs)

            # 保存图像
            cv2.imwrite(config['filename'], cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            print(f"   保存: {config['filename']}")
            saved_images.append(image)

        # 创建对比图
        print(f"\n创建对比图...")
        if len(saved_images) >= 2:
            # 调整大小
            target_h = 300
            resized = []
            for img in saved_images[:4]:  # 最多4张
                h, w = img.shape[:2]
                target_w = int(w * target_h / h)
                resized.append(cv2.resize(img, (target_w, target_h)))

            # 水平拼接
            comparison = np.hstack(resized)
            cv2.imwrite('affordance_comparison.png', cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
            print("保存对比图: affordance_comparison.png")

        print("\n✅ Affordance可视化测试完成!")
        print("\n生成的文件:")
        for config in test_configs:
            print(f"  - {config['filename']}: {config['name']}")
        print("  - affordance_comparison.png: 对比图")

        return saved_images

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None

    finally:
        env.close()

def test_affordance_with_actions():
    """测试不同动作下的affordance可视化"""
    print("=== 测试不同动作下的Affordance ===\n")

    task_name = "widowx_carrot_on_plate"
    env = simpler_env.make(task_name)

    try:
        obs, reset_info = env.reset(seed=42)

        # 定义测试动作
        test_actions = [
            {
                'name': '初始位置',
                'action': None
            },
            {
                'name': '向上移动',
                'action': np.array([0.0, 0.0, 0.03, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            },
            {
                'name': '向前移动',
                'action': np.array([0.03, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            },
            {
                'name': '旋转',
                'action': np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.15, 0.0], dtype=np.float32)
            }
        ]

        images_original = []
        images_with_affordance = []

        for i, action_info in enumerate(test_actions):
            print(f"\n{i+1}. {action_info['name']}:")

            # 执行动作（如果不是初始位置）
            if action_info['action'] is not None:
                try:
                    obs, _, _, _, _ = env.step(action_info['action'])
                except Exception as e:
                    print(f"   ⚠ 动作执行失败: {e}")
                    continue

            # 获取原始图像
            if "image" in obs:
                cam_imgs = obs["image"]
                if "3rd_view_camera" in cam_imgs and "rgb" in cam_imgs["3rd_view_camera"]:
                    img_orig = cam_imgs["3rd_view_camera"]["rgb"].copy()
                elif "base_camera" in cam_imgs and "rgb" in cam_imgs["base_camera"]:
                    img_orig = cam_imgs["base_camera"]["rgb"].copy()
                else:
                    img_orig = get_image_from_maniskill2_obs_dict(env, obs)
            else:
                img_orig = get_image_from_maniskill2_obs_dict(env, obs)

            # 添加affordance
            obs_with_aff = add_affordance_to_observation(obs, env)

            if "image" in obs_with_aff:
                cam_imgs = obs_with_aff["image"]
                if "3rd_view_camera" in cam_imgs and "rgb" in cam_imgs["3rd_view_camera"]:
                    img_aff = cam_imgs["3rd_view_camera"]["rgb"]
                elif "base_camera" in cam_imgs and "rgb" in cam_imgs["base_camera"]:
                    img_aff = cam_imgs["base_camera"]["rgb"]
                else:
                    img_aff = get_image_from_maniskill2_obs_dict(env, obs_with_aff)
            else:
                img_aff = get_image_from_maniskill2_obs_dict(env, obs_with_aff)

            # 保存图像
            cv2.imwrite(f'action_{i}_original.png', cv2.cvtColor(img_orig, cv2.COLOR_RGB2BGR))
            cv2.imwrite(f'action_{i}_affordance.png', cv2.cvtColor(img_aff, cv2.COLOR_RGB2BGR))
            print(f"   保存: action_{i}_original.png, action_{i}_affordance.png")

            images_original.append(img_orig)
            images_with_affordance.append(img_aff)

        # 创建网格对比图
        print(f"\n创建对比图...")
        if len(images_original) >= 2:
            # 上下对比（原始vs affordance）
            target_h = 200

            row1_images = []  # 原始图像行
            row2_images = []  # affordance图像行

            for img_orig, img_aff in zip(images_original, images_with_affordance):
                h, w = img_orig.shape[:2]
                target_w = int(w * target_h / h)
                row1_images.append(cv2.resize(img_orig, (target_w, target_h)))
                row2_images.append(cv2.resize(img_aff, (target_w, target_h)))

            row1 = np.hstack(row1_images)
            row2 = np.hstack(row2_images)
            grid = np.vstack([row1, row2])

            cv2.imwrite('affordance_actions_comparison.png', cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
            print("保存对比图: affordance_actions_comparison.png")
            print("  上行: 原始图像")
            print("  下行: 添加affordance的图像")

        print("\n✅ 动作测试完成!")
        return images_with_affordance

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None

    finally:
        env.close()

def test_simple_object_annotation():
    """简化的物体标注测试（仿照affordance风格）"""
    print("=== Simple Object Annotation Test ===\n")

    task_name = "widowx_carrot_on_plate"
    env = simpler_env.make(task_name)

    try:
        obs, reset_info = env.reset(seed=42)

        print("Testing different annotation options:")

        # 测试配置 - 仿照affordance测试的风格
        test_configs = [
            {
                'name': 'Original Image',
                'include_objects': False,
                'filename': 'test_original_image.png'
            },
            {
                'name': 'Gripper Only',
                'include_objects': False,
                'filename': 'test_gripper_only.png'
            },
            {
                'name': 'Gripper + Objects',
                'include_objects': True,
                'filename': 'test_gripper_objects.png'
            }
        ]

        saved_images = []

        # 获取数据（传递reset_info以获取物体信息）
        data = get_robot_pose_and_image_from_env(env, obs)

        # 手动添加物体信息（因为get_robot_pose_and_image_from_env没有reset_info参数）
        objects_info = get_object_coordinates_from_env(env, obs, reset_info)
        data['objects_info'] = objects_info

        # 显示找到的物体信息
        objects_info = data['objects_info']
        if objects_info:
            print(f"\nFound {len(objects_info)} objects:")
            for obj_key, obj_info in objects_info.items():
                name = obj_info['name']
                pos = obj_info['position']
                obj_type = obj_info['type']
                print(f"  - {name} ({obj_type}): [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
        else:
            print("\nNo objects found")

        for i, config in enumerate(test_configs):
            print(f"\n{i+1}. {config['name']}:")

            if config['name'] == 'Original Image':
                # 原始图像
                image = data['image']
            else:
                # 创建标注图像
                image = create_annotated_image(
                    data,
                    config['name'],
                    config['filename'],
                    include_objects=config['include_objects']
                )

            # 保存图像
            if config['name'] == 'Original Image':
                cv2.imwrite(config['filename'], cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

            print(f"   Saved: {config['filename']}")
            saved_images.append(image)

        # 创建对比图
        print(f"\nCreating comparison image...")
        if len(saved_images) >= 2:
            # 调整大小
            target_h = 300
            resized = []
            for img in saved_images:
                h, w = img.shape[:2]
                target_w = int(w * target_h / h)
                resized.append(cv2.resize(img, (target_w, target_h)))

            # 水平拼接
            comparison = np.hstack(resized)
            cv2.imwrite('simple_object_comparison.png', cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
            print("Saved comparison: simple_object_comparison.png")

        print("\n✅ Simple object annotation test completed!")
        print("\nGenerated files:")
        for config in test_configs:
            print(f"  - {config['filename']}: {config['name']}")
        print("  - simple_object_comparison.png: Comparison image")

        return saved_images

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

    finally:
        env.close()

def draw_out_of_bounds_indicator(image, position_2d, obj_info, img_w, img_h):
    """为超出图像范围的物体绘制边界指示器"""
    u, v = position_2d
    name = obj_info.get('name', 'unknown')
    color = obj_info.get('color', (0, 255, 0))

    # 将超出范围的坐标限制到边界
    u_clamped = max(10, min(img_w - 10, u))
    v_clamped = max(10, min(img_h - 10, v))

    # 绘制边界指示器
    cv2.circle(image, (u_clamped, v_clamped), 6, color, 2)  # 空心圆表示超出范围
    cv2.circle(image, (u_clamped, v_clamped), 8, (255, 255, 255), 1)

    # 添加箭头指示实际方向
    if u < 0:
        # 在左边界，箭头指向左
        cv2.arrowedLine(image, (u_clamped + 5, v_clamped), (u_clamped - 5, v_clamped), color, 2)
    elif u >= img_w:
        # 在右边界，箭头指向右
        cv2.arrowedLine(image, (u_clamped - 5, v_clamped), (u_clamped + 5, v_clamped), color, 2)

    if v < 0:
        # 在上边界，箭头指向上
        cv2.arrowedLine(image, (u_clamped, v_clamped + 5), (u_clamped, v_clamped - 5), color, 2)
    elif v >= img_h:
        # 在下边界，箭头指向下
        cv2.arrowedLine(image, (u_clamped, v_clamped - 5), (u_clamped, v_clamped + 5), color, 2)

    # 添加标签
    label = f"{name[:8]}*"  # 加*表示超出范围
    cv2.putText(image, label, (u_clamped + 12, v_clamped - 5),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # 添加坐标信息
    coord_text = f"({u},{v})"
    cv2.putText(image, coord_text, (u_clamped + 12, v_clamped + 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

def draw_failed_projection_info(image, obj_info, total_objects):
    """为投影失败的物体显示信息"""
    name = obj_info.get('name', 'unknown')
    color = obj_info.get('color', (0, 255, 0))

    # 在图像右上角显示失败信息
    h, w = image.shape[:2]
    y_pos = 50 + (total_objects * 15)  # 避免重叠

    fail_text = f"{name[:12]}: PROJ FAIL"
    cv2.putText(image, fail_text, (w - 200, y_pos),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

def create_enhanced_annotated_image(data, step_name, save_name=None, include_objects=True):
    """创建增强的标注图像，显示所有物体（包括超出范围的）"""
    image = data['image'].copy()
    h, w = image.shape[:2]

    # 获取相机参数
    intrinsic = None
    extrinsic = None
    use_real_projection = False

    if data['camera_params'] and data['camera_name'] in data['camera_params']:
        try:
            params = data['camera_params'][data['camera_name']]
            intrinsic = params['intrinsic_cv']
            extrinsic = params['extrinsic_cv']
            use_real_projection = True
        except Exception:
            pass

    # 1. 处理夹爪位姿标注
    gripper_position_2d = None
    if use_real_projection:
        gripper_position_2d = project_3d_to_2d(
            data['position'], intrinsic, extrinsic
        )

    if gripper_position_2d is None:
        gripper_position_2d = [w // 2, h // 2]

    if not (0 <= gripper_position_2d[0] < w and 0 <= gripper_position_2d[1] < h):
        gripper_position_2d = [w // 2, h // 2]

    # 绘制标注
    annotated_image = image.copy()

    # 绘制夹爪朝向箭头
    if use_real_projection:
        success = draw_affordance_arrow(
            annotated_image, gripper_position_2d, data['position'], data['quaternion'],
            intrinsic, extrinsic, arrow_length=0.08, arrow_color=(0, 255, 0),
            arrow_thickness=3, show_point=True
        )
        if not success:
            draw_affordance_arrow_simple(
                annotated_image, gripper_position_2d, data['quaternion'],
                arrow_length=60, arrow_color=(0, 255, 0), arrow_thickness=3, show_point=True
            )
    else:
        draw_affordance_arrow_simple(
            annotated_image, gripper_position_2d, data['quaternion'],
            arrow_length=60, arrow_color=(0, 255, 0), arrow_thickness=3, show_point=True
        )

    # 绘制夹爪位姿信息
    annotated_image = draw_pose_annotation(
        annotated_image, gripper_position_2d,
        data['quaternion'], data['gripper_width']
    )

    # 2. 处理物体标注 - 增强版本（显示所有物体）
    if include_objects and 'objects_info' in data:
        objects_info = data['objects_info']
        in_bounds_count = 0
        out_bounds_count = 0
        failed_count = 0

        for obj_key, obj_info in objects_info.items():
            obj_position_3d = obj_info['position']

            # 投影物体位置到2D
            obj_position_2d = None
            if use_real_projection:
                obj_position_2d = project_3d_to_2d(
                    obj_position_3d, intrinsic, extrinsic
                )

            # 处理投影结果
            if obj_position_2d is not None:
                u, v = obj_position_2d
                in_bounds = (0 <= u < w and 0 <= v < h)

                if in_bounds:
                    # 在图像范围内，正常绘制
                    draw_simple_object_marker(annotated_image, obj_position_2d, obj_info)
                    in_bounds_count += 1
                else:
                    # 超出范围，绘制边界指示器
                    draw_out_of_bounds_indicator(annotated_image, obj_position_2d, obj_info, w, h)
                    out_bounds_count += 1
            else:
                # 投影失败，在图像角落显示信息
                draw_failed_projection_info(annotated_image, obj_info, failed_count)
                failed_count += 1

    # 3. 添加步骤标签
    cv2.putText(annotated_image, f"{step_name}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(annotated_image, f"{step_name}", (11, 31),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)

    # 4. 添加详细统计信息
    if include_objects and 'objects_info' in data:
        total_count = len(data['objects_info'])
        stats_lines = [
            f"Objects: {total_count} total",
            f"In view: {in_bounds_count}",
            f"Out of bounds: {out_bounds_count}",
            f"Proj failed: {failed_count}"
        ]

        for i, line in enumerate(stats_lines):
            y_pos = h - 60 + i * 12
            cv2.putText(annotated_image, line, (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(annotated_image, line, (11, y_pos + 1),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)

    # 5. 保存图像
    if save_name:
        cv2.imwrite(save_name, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

    return annotated_image

def quick_test():
    """快速测试函数 - 直接运行多位置测试"""
    print("快速多位置测试模式")
    print("=" * 40)
    results = test_multiple_gripper_positions()
    if results:
        print(f"\n✅ 成功生成 {len(results)} 个不同位置的标注图像！")
    return results

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "--quick":
            quick_test()
        elif sys.argv[1] == "--safe":
            safe_test()
        elif sys.argv[1] == "--affordance":
            test_affordance_visualization()
        elif sys.argv[1] == "--affordance-actions":
            test_affordance_with_actions()
        elif sys.argv[1] == "--objects":
            test_simple_object_annotation()
        else:
            main()
    else:
        main()