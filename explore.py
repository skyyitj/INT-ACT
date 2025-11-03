#!/usr/bin/env python3
"""
探索如何从simpler_env环境中获取目标物体的3D坐标
"""

import numpy as np
import simpler_env
import warnings

# 抑制警告
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium")

def explore_object_coordinates():
    """探索获取物体坐标的方法"""
    print("=== 探索物体坐标获取方法 ===\n")

    task_name = "widowx_carrot_on_plate"
    env = simpler_env.make(task_name)

    try:
        obs, reset_info = env.reset(seed=42)

        print("1. 观测字典结构:")
        print(f"   顶层键: {list(obs.keys())}")

        if 'extra' in obs:
            print(f"   extra中的键: {list(obs['extra'].keys())}")

        if 'agent' in obs:
            if isinstance(obs['agent'], dict):
                print(f"   agent中的键: {list(obs['agent'].keys())}")
                # 检查agent中是否有物体位置信息
                for key, value in obs['agent'].items():
                    print(f"     {key}: {type(value)} - {np.array(value).shape if hasattr(value, '__len__') else 'scalar'}")
            else:
                print(f"   agent类型: {type(obs['agent'])}")

        print("\n2. 尝试通过env.unwrapped获取场景信息:")
        try:
            unwrapped_env = env.unwrapped
            print(f"   unwrapped环境类型: {type(unwrapped_env)}")

            # 尝试获取场景
            if hasattr(unwrapped_env, 'scene'):
                scene = unwrapped_env.scene
                print(f"   场景对象类型: {type(scene)}")

                # 尝试获取所有actors
                if hasattr(scene, 'get_all_actors'):
                    actors = scene.get_all_actors()
                    print(f"   场景中的actors数量: {len(actors)}")

                    for i, actor in enumerate(actors):
                        if hasattr(actor, 'name'):
                            name = actor.name
                        else:
                            name = f"unnamed_actor_{i}"

                        print(f"     Actor {i}: {name}")

                        # 尝试获取位置
                        if hasattr(actor, 'get_pose'):
                            pose = actor.get_pose()
                            print(f"       位置: {pose.p if hasattr(pose, 'p') else pose}")
                        elif hasattr(actor, 'pose'):
                            pose = actor.pose
                            print(f"       位置: {pose.p if hasattr(pose, 'p') else pose}")
                        elif hasattr(actor, 'get_position'):
                            pos = actor.get_position()
                            print(f"       位置: {pos}")
                        else:
                            print(f"       无法获取位置信息")

                        # 检查是否是目标物体（胡萝卜）
                        if 'carrot' in name.lower() or 'target' in name.lower():
                            print(f"       *** 可能的目标物体: {name} ***")

                elif hasattr(scene, 'actors'):
                    actors = scene.actors
                    print(f"   场景中的actors数量: {len(actors)}")
                    for i, actor in enumerate(actors[:10]):  # 只显示前10个
                        name = getattr(actor, 'name', f'unnamed_{i}')
                        print(f"     Actor {i}: {name}")
                else:
                    print("   无法获取actors列表")

            else:
                print("   unwrapped环境没有scene属性")

        except Exception as e:
            print(f"   获取场景信息失败: {e}")

        print("\n3. 尝试通过环境属性获取物体信息:")
        try:
            # 检查环境是否有特定的物体属性
            env_attrs = [attr for attr in dir(unwrapped_env) if not attr.startswith('_')]
            print(f"   环境可用属性: {env_attrs[:20]}...")  # 只显示前20个

            # 查找可能包含物体信息的属性
            object_related_attrs = [attr for attr in env_attrs if any(keyword in attr.lower()
                                  for keyword in ['object', 'target', 'carrot', 'goal', 'item'])]

            if object_related_attrs:
                print(f"   可能相关的属性: {object_related_attrs}")

                for attr in object_related_attrs:
                    try:
                        value = getattr(unwrapped_env, attr)
                        print(f"     {attr}: {type(value)} - {value}")
                    except Exception as e:
                        print(f"     {attr}: 获取失败 - {e}")
            else:
                print("   未找到明显的物体相关属性")

        except Exception as e:
            print(f"   检查环境属性失败: {e}")

        print("\n4. 尝试通过任务特定方法获取目标信息:")
        try:
            # 对于widowx_carrot_on_plate任务，尝试获取胡萝卜位置
            if hasattr(unwrapped_env, 'get_target_pose'):
                target_pose = unwrapped_env.get_target_pose()
                print(f"   目标位姿: {target_pose}")
            elif hasattr(unwrapped_env, 'target_pose'):
                target_pose = unwrapped_env.target_pose
                print(f"   目标位姿: {target_pose}")
            elif hasattr(unwrapped_env, 'goal_pose'):
                goal_pose = unwrapped_env.goal_pose
                print(f"   目标位姿: {goal_pose}")
            else:
                print("   未找到标准的目标位姿获取方法")

        except Exception as e:
            print(f"   获取目标信息失败: {e}")

        print("\n5. 检查reset_info中是否有物体信息:")
        if reset_info:
            print(f"   reset_info键: {list(reset_info.keys()) if isinstance(reset_info, dict) else type(reset_info)}")
            if isinstance(reset_info, dict):
                for key, value in reset_info.items():
                    print(f"     {key}: {type(value)} - {value}")
        else:
            print("   reset_info为空")

        return env, obs

    except Exception as e:
        print(f"❌ 探索过程出错: {e}")
        import traceback
        traceback.print_exc()
        env.close()
        return None, None

def test_specific_object_access():
    """测试特定的物体访问方法"""
    print("\n=== 测试特定物体访问方法 ===\n")

    task_name = "widowx_carrot_on_plate"
    env = simpler_env.make(task_name)

    try:
        obs, reset_info = env.reset(seed=42)
        unwrapped_env = env.unwrapped

        # 方法1: 通过场景查找特定名称的物体
        if hasattr(unwrapped_env, 'scene'):
            scene = unwrapped_env.scene

            # 尝试不同的方法获取actors
            actors = []
            if hasattr(scene, 'get_all_actors'):
                actors = scene.get_all_actors()
            elif hasattr(scene, 'actors'):
                actors = scene.actors
            elif hasattr(scene, 'get_actors'):
                actors = scene.get_actors()

            print(f"找到 {len(actors)} 个actors")

            # 查找胡萝卜
            carrot_actors = []
            for actor in actors:
                name = getattr(actor, 'name', '')
                if any(keyword in name.lower() for keyword in ['carrot', 'target', 'obj']):
                    carrot_actors.append(actor)
                    print(f"找到可能的目标物体: {name}")

                    # 获取位置
                    try:
                        if hasattr(actor, 'get_pose'):
                            pose = actor.get_pose()
                            position = pose.p if hasattr(pose, 'p') else pose[:3]
                            print(f"  位置: {position}")
                        elif hasattr(actor, 'pose'):
                            pose = actor.pose
                            position = pose.p if hasattr(pose, 'p') else pose[:3]
                            print(f"  位置: {position}")
                    except Exception as e:
                        print(f"  获取位置失败: {e}")

            if not carrot_actors:
                print("未找到明显的胡萝卜物体，显示所有物体:")
                for i, actor in enumerate(actors[:10]):
                    name = getattr(actor, 'name', f'actor_{i}')
                    print(f"  {i}: {name}")

        # 方法2: 检查是否有预定义的目标物体属性
        target_attrs = ['target_obj', 'carrot', 'goal_obj', 'manipulated_object']
        for attr in target_attrs:
            if hasattr(unwrapped_env, attr):
                obj = getattr(unwrapped_env, attr)
                print(f"找到目标物体属性 {attr}: {obj}")

                if hasattr(obj, 'get_pose'):
                    pose = obj.get_pose()
                    print(f"  位置: {pose.p if hasattr(pose, 'p') else pose}")

        return env, obs

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        env.close()
        return None, None

if __name__ == "__main__":
    # 探索基本方法
    env, obs = explore_object_coordinates()

    if env is not None:
        env.close()

    # 测试特定访问方法
    env, obs = test_specific_object_access()

    if env is not None:
        env.close()
