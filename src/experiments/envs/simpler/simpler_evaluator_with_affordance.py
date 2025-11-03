"""
SimplerEnv评估器 - 添加Affordance支持

在原有评估器基础上添加affordance功能，用于测试affordance对性能的影响
"""
import collections
import os
import sys
import time
from pathlib import Path

import imageio
import numpy as np
import simpler_env
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
from typing_extensions import override

import wandb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))

from src.agent.configuration_pipeline import TrainPipelineConfig
from src.experiments.env_adapters.language_mapper import PersistentLanguageMapper
from src.experiments.envs.base_evaluator import BaseEvaluator
from src.utils.monitor import setup_logger

# 导入affordance功能
from get_pose_corrected_coordinates import add_affordance_to_observation

os.environ["WANDB__SERVICE_WAIT"] = "300"


class SimplerEvaluatorWithAffordance(BaseEvaluator):
    """
    带Affordance的SimplerEnv评估器

    在每次获取观测图像时自动添加affordance信息（夹爪朝向箭头）
    """

    def __init__(self, pipeline_cfg: TrainPipelineConfig):
        super().__init__(pipeline_cfg)

        self.language_logic_chain = pipeline_cfg.eval_cfg.language_logic_chain

        # Affordance配置
        self.use_affordance = pipeline_cfg.eval_cfg.use_affordance
        affordance_color_raw = pipeline_cfg.eval_cfg.affordance_color
        # 确保颜色是 tuple 格式（OpenCV需要）
        self.affordance_color = tuple(affordance_color_raw) if isinstance(affordance_color_raw, list) else affordance_color_raw
        self.affordance_thickness = pipeline_cfg.eval_cfg.affordance_thickness
        self.affordance_length = pipeline_cfg.eval_cfg.affordance_length
        self.affordance_show_point = pipeline_cfg.eval_cfg.affordance_show_point

        # 统计信息
        self.affordance_stats = {
            'total_frames': 0,
            'affordance_added': 0,
            'affordance_failed': 0
        }

        if self.use_affordance:
            print("=" * 60)
            print("🎯 Affordance功能已启用")
            print(f"  颜色: {self.affordance_color}")
            print(f"  粗细: {self.affordance_thickness}")
            print(f"  长度: {self.affordance_length}m")
            print(f"  显示位置点: {self.affordance_show_point}")
            print("=" * 60)

        if self.language_logic_chain:
            language_mapping_candidates = {
                "carrot": ["the yellow vegetable", "the veggie", "the yellow thing that rabbit likes", "the veggie that rabbit likes"],
                "eggplant": ["the purple vegetable", "the veggie", "the thing that looks like a purple balloon"],
                "spoon": ["the silver spoon", "the thing that people use to eat soup", "the shiny thing"],
                "cube": ["the thing that looks like a box", "lego"],
            }
            self.language_mapper = PersistentLanguageMapper(mapping_candidates=language_mapping_candidates, seed=self.seed)

    def add_affordance_to_image(self, img, env, obs):
        """
        为图像添加affordance

        Args:
            img: 原始RGB图像
            env: SimplerEnv环境实例
            obs: 环境观测字典

        Returns:
            添加了affordance的图像
        """
        if not self.use_affordance:
            return img

        self.affordance_stats['total_frames'] += 1

        try:
            # 添加affordance到观测
            obs_with_aff = add_affordance_to_observation(
                obs, env,
                arrow_length=self.affordance_length,
                arrow_color=self.affordance_color,
                arrow_thickness=self.affordance_thickness,
                show_point=self.affordance_show_point
            )

            # 获取添加了affordance的图像
            img_with_aff = get_image_from_maniskill2_obs_dict(env, obs_with_aff)

            self.affordance_stats['affordance_added'] += 1
            return np.ascontiguousarray(img_with_aff)

        except Exception as e:
            self.affordance_stats['affordance_failed'] += 1
            if self.affordance_stats['affordance_failed'] <= 3:  # 只打印前3次错误
                print(f"⚠️ Affordance添加失败: {e}")
            return img

    @override
    def evaluate(self):
        '''Run evaluation on all tasks in the task list'''
        # 重置统计信息
        self.affordance_stats = {
            'total_frames': 0,
            'affordance_added': 0,
            'affordance_failed': 0
        }

        # 实现评估逻辑（复制自SimplerEvaluator）
        for gradient_step in self.gradient_steps:
            if self.no_gradient_steps:
                model_path = Path(self.eval_cfg.pretrained_model_path)
            else:
                model_path = Path(self.eval_cfg.pretrained_model_path) / f"step_{gradient_step!s}"

            self._initialze_model_client(model_path=str(model_path), gradient_step=gradient_step)

            for task_name in self.task_lists:
                if not self.debug:
                    self._update_n_eval_episode(task_name) # some tasks have different number of possible episodes
                self.evaluate_task(task_name)

            if self.use_wandb:
                wandb.log(self.wandb_metrics, step=int(gradient_step), commit=True)

        # 打印affordance统计
        if self.use_affordance:
            print("\n" + "=" * 60)
            print("📊 Affordance统计信息:")
            print(f"  总帧数: {self.affordance_stats['total_frames']}")
            print(f"  成功添加: {self.affordance_stats['affordance_added']}")
            print(f"  添加失败: {self.affordance_stats['affordance_failed']}")
            if self.affordance_stats['total_frames'] > 0:
                success_rate = self.affordance_stats['affordance_added'] / self.affordance_stats['total_frames']
                print(f"  成功率: {success_rate*100:.1f}%")
            print("=" * 60)

    @override
    def evaluate_task(self, task_name):
        '''
        Evaluates a single task using the trained model.

        Args:
            task_name: str, the name of the task to evaluate
        '''
        # 创建任务日志目录
        task_log_dir = Path(self.log_dir) / "task_logs"
        if self.main_rank:
            task_log_dir.mkdir(parents=True, exist_ok=True)

        task_logger = setup_logger(
            main_rank=self.main_rank,
            filename=task_log_dir / f"{task_name}.log" if not self.debug else None,
            debug=self.debug,
            name=f'{task_name}_logger'
        )
        task_logger.info(f"Evaluating task: {task_name}")

        env = simpler_env.make(task_name)
        elapsed_steps = 0

        instruction = None
        if self.language_logic_chain:
            mapper = PersistentLanguageMapper()
            for key in self.language_logic_chain.keys():
                if key in task_name:
                    mapper.update(key, self.language_logic_chain[key])
            obs, reset_info = env.reset()
            instruction = reset_info.get("text_plan", ["default instruction"])[0]
            for old, new in mapper.mapping.items():
                instruction = instruction.replace(old, new)
        else:
            obs, reset_info = env.reset(seed=self.seed)
            instruction = reset_info.get("text_plan", ["default instruction"])[0]

        episode_highest_rewards = []

        for i_episode in range(self.n_eval_episode):
            episode_return, episode_highest_reward = 0.0, 0.0
            elapsed_steps = 0

            obs, reset_info = env.reset(seed=self.seed + i_episode)

            recording = i_episode < self.n_video and self.pipeline_cfg.eval_cfg.recording
            if recording:
                current_time = time.strftime("%Y%m%d-%H%M%S")
                video_default_path = Path(self.log_dir) / f"{task_name}_episode_{i_episode}_{current_time}.mp4"
                video_default_path.parent.mkdir(parents=True, exist_ok=True)
                video_writer = imageio.get_writer(video_default_path)

            task_logger.info(
                f"Reset info: {reset_info} Instruction: {instruction} Max episode length: {env.spec.max_episode_steps}"
            )

            # Set up receding horizon control
            action_plan = collections.deque()
            while True:
                # 获取原始图像
                img = np.ascontiguousarray(get_image_from_maniskill2_obs_dict(env, obs))

                # 🎯 添加affordance（如果启用）
                img_for_policy = self.add_affordance_to_image(img, env, obs)

                if not action_plan:
                    # action horizon is all executed
                    # Query model to get action
                    element = {
                        "observation.images.top": img_for_policy,  # 使用带affordance的图像
                        "observation.state": obs,
                        "task": str(instruction)
                    }
                    action_chunk = self.client.infer(element)
                    action_plan.extend(action_chunk[: self.action_step])

                action = action_plan.popleft()
                obs, reward, success, truncated, info = env.step(action.copy())

                # Record video frame if enabled
                # 注意：视频记录使用带affordance的图像
                if recording:
                    video_writer.append_data(img_for_policy if self.use_affordance else img)

                elapsed_steps += 1
                episode_return += reward
                episode_highest_reward = max(episode_highest_reward, reward)

                if success or truncated or elapsed_steps >= env.spec.max_episode_steps:
                    episode_highest_rewards.append(episode_highest_reward)

                    task_logger.info(
                        f"Episode {i_episode}: success={success}, truncated={truncated}, "
                        f"steps={elapsed_steps}, return={episode_return:.2f}, "
                        f"highest_reward={episode_highest_reward:.2f}"
                    )

                    if recording:
                        video_writer.close()
                        task_logger.info(f"Video saved: {video_default_path}")

                    break

        env.close()

        # Calculate metrics
        success_rate = np.mean(episode_highest_rewards) * 100
        task_logger.info(f"Task {task_name} completed: Success rate = {success_rate:.2f}%")

        return {
            "task_name": task_name,
            "success_rate": success_rate,
            "episode_rewards": episode_highest_rewards,
            "n_episodes": self.n_eval_episode
        }

    def _update_n_eval_episode(self, task_name):
        """更新评估episode数量（复制自SimplerEvaluator）"""
        if "google_robot" in task_name:
            if 'coke' in task_name:
                self.n_eval_episode = 25 * 4 # 25 locations, 4 urdfs, 10 trials each
            elif 'move' in task_name:
                self.n_eval_episode = 60 * 4 # 60 locations, 4 urdfs, 10 trials each
            elif 'drawer' in task_name:
                self.n_eval_episode = 3 * 4 * 9 # 3 drawers, 4 urdfs, 9 locations/rgb_overlay_paths, 10 trials each
            elif 'apple' in task_name:
                self.n_eval_episode = 9 * 4 * 3 # 9 apple locations, 4 urdfs, 3 robot locations/rgb_overlay_paths, 10 trials each
            self.n_video = self.n_eval_episode

    def _process_episode_stats(self, metric, episode_stats, success):
        '''
        Process episode stats to extract relevant information
        '''
        # Extract relevant information from episode_stats
        metric['Success Rate'].append(success)
        metric['Move Correct'].append(episode_stats.get('moved_correct_obj', 0))
        metric['Wrong Obj Attempt'].append(episode_stats.get('moved_wrong_obj', 0))
        metric['Grasp Correct'].append(episode_stats.get('is_src_obj_grasped', 0))
        metric['Src Intention Correct'].append(episode_stats.get('source_intention', 0))

    def _aggregate_metrics(self, metrics):
        '''
        Aggregate metrics across all episodes
        '''
        aggregated_metrics = {}
        for key in metrics.keys():
            aggregated_metrics[key] = np.mean(metrics[key])
        return aggregated_metrics

    def _log_summary(self, logger, cnt_episode, eval_time, metrics):
        """记录评估总结"""
        logger.info(f"Evaluated {cnt_episode} episodes in {eval_time:.2f} seconds")
        for key, value in metrics.items():
            logger.info(f"{key}: {value:.4f}")
        logger.info("=" * 50)

    def _preprocess_task_instruction(self, instruction):
        """预处理任务指令"""
        if self.language_logic_chain:
            return self.language_mapper.map(instruction)
        return instruction

