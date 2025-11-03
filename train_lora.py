#!/usr/bin/env python
"""
PI0 LoRA微调训练脚本

这个脚本展示了如何使用LoRA对PI0模型进行微调训练。
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from lerobot.common.policies.pi0.configuration_pi0 import PI0Config
from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.common.datasets import LeRobotDataset


class PI0LoRATrainer:
    """PI0 LoRA微调训练器"""

    def __init__(self, config: PI0Config, dataset_stats=None):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 创建模型
        self.policy = PI0Policy(config=config, dataset_stats=dataset_stats)
        self.policy.to(self.device)

        # 设置优化器
        self.optimizer = self._setup_optimizer()

        # 设置学习率调度器
        self.scheduler = self._setup_scheduler()

        print(f"训练器初始化完成，使用设备: {self.device}")
        if self.config.use_lora:
            print("LoRA微调模式已启用")
            self.policy.print_trainable_parameters()

    def _setup_optimizer(self):
        """设置优化器"""
        if self.config.use_lora:
            # LoRA模式下只优化LoRA参数
            params = self.policy.get_optim_params()
        else:
            # 全参数微调
            params = self.policy.parameters()

        optimizer = torch.optim.AdamW(
            params,
            lr=self.config.optimizer_lr,
            betas=self.config.optimizer_betas,
            eps=self.config.optimizer_eps,
            weight_decay=self.config.optimizer_weight_decay,
        )
        return optimizer

    def _setup_scheduler(self):
        """设置学习率调度器"""
        from lerobot.common.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
        scheduler_config = self.config.get_scheduler_preset()

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=scheduler_config.num_warmup_steps,
            T_mult=1,
            eta_min=scheduler_config.decay_lr,
        )
        return scheduler

    def train_step(self, batch):
        """执行一个训练步骤"""
        self.policy.train()

        # 将数据移到设备上
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()}

        # 前向传播
        loss, loss_dict = self.policy.forward(batch)

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪
        if hasattr(self.config, 'max_grad_norm'):
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)

        self.optimizer.step()
        self.scheduler.step()

        return loss.item(), loss_dict

    def save_checkpoint(self, path, epoch, step):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
        }

        # 如果使用LoRA，也保存LoRA适配器
        if self.config.use_lora:
            lora_path = f"{path}_lora"
            self.policy.save_lora_adapters(lora_path)
            checkpoint['lora_path'] = lora_path

        torch.save(checkpoint, f"{path}_epoch_{epoch}_step_{step}.pt")
        print(f"检查点已保存: {path}_epoch_{epoch}_step_{step}.pt")

    def load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.policy.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # 如果使用LoRA且有LoRA路径，加载LoRA适配器
        if self.config.use_lora and 'lora_path' in checkpoint:
            self.policy.load_lora_adapters(checkpoint['lora_path'])

        print(f"检查点已加载: {checkpoint_path}")
        return checkpoint['epoch'], checkpoint['step']


def train_pi0_lora(
    dataset_path: str,
    output_dir: str,
    use_lora: bool = True,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    num_epochs: int = 10,
    batch_size: int = 4,
    learning_rate: float = 2.5e-5,
    save_interval: int = 1000,
):
    """训练PI0模型（支持LoRA微调）"""

    # 创建配置
    config = PI0Config(
        use_lora=use_lora,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        optimizer_lr=learning_rate,
        # 其他配置
        freeze_vision_encoder=True,
        train_expert_only=False,
        train_state_proj=True,
    )

    # 加载数据集
    print("加载数据集...")
    dataset = LeRobotDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 创建训练器
    trainer = PI0LoRATrainer(config, dataset.dataset_stats)

    print(f"开始训练，共 {num_epochs} 个epoch")
    print(f"数据集大小: {len(dataset)}")
    print(f"批次大小: {batch_size}")

    global_step = 0

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            try:
                loss, loss_dict = trainer.train_step(batch)
                epoch_loss += loss
                num_batches += 1
                global_step += 1

                if global_step % 100 == 0:
                    print(f"Epoch {epoch+1}/{num_epochs}, Step {global_step}, Loss: {loss:.4f}")

                # 定期保存检查点
                if global_step % save_interval == 0:
                    trainer.save_checkpoint(f"{output_dir}/checkpoint", epoch, global_step)

            except Exception as e:
                print(f"训练步骤出错: {e}")
                continue

        avg_loss = epoch_loss / max(num_batches, 1)
        print(f"Epoch {epoch+1} 完成，平均损失: {avg_loss:.4f}")

    # 保存最终模型
    trainer.save_checkpoint(f"{output_dir}/final_model", num_epochs, global_step)
    print("训练完成!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PI0 LoRA微调训练")
    parser.add_argument("--dataset_path", type=str, required=True, help="数据集路径")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--use_lora", action="store_true", help="启用LoRA微调")
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    parser.add_argument("--num_epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=4, help="批次大小")
    parser.add_argument("--learning_rate", type=float, default=2.5e-5, help="学习率")
    parser.add_argument("--save_interval", type=int, default=1000, help="保存间隔")

    args = parser.parse_args()

    train_pi0_lora(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        save_interval=args.save_interval,
    )
