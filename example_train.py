#!/usr/bin/env python
"""
PI0 LoRA微调训练示例脚本

使用方法:
python example_lora_training.py --use_lora --lora_rank 16 --lora_alpha 32
"""

import argparse
from lerobot.common.policies.pi0.configuration_pi0 import PI0Config
from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy


def main():
    parser = argparse.ArgumentParser(description="PI0 LoRA微调训练示例")
    parser.add_argument("--use_lora", action="store_true", help="启用LoRA微调")
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    parser.add_argument("--model_path", type=str, default="models/INTACT-pi0-finetune-bridge", help="预训练模型路径")
    parser.add_argument("--save_path", type=str, default="./lora_adapters", help="LoRA适配器保存路径")

    args = parser.parse_args()

    # 创建配置
    config = PI0Config(
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        # 其他配置参数
        freeze_vision_encoder=True,
        train_expert_only=False,
        train_state_proj=True,
    )

    print("=== PI0 LoRA微调配置 ===")
    print(f"使用LoRA: {config.use_lora}")
    if config.use_lora:
        print(f"LoRA rank: {config.lora_rank}")
        print(f"LoRA alpha: {config.lora_alpha}")
        print(f"LoRA dropout: {config.lora_dropout}")
        print(f"目标模块: {config.lora_target_modules}")

    # 创建模型
    print("\n=== 创建PI0模型 ===")
    try:
        policy = PI0Policy.from_pretrained(args.model_path, config=config)
        print("模型创建成功!")

        # 打印可训练参数
        print("\n=== 可训练参数统计 ===")
        policy.print_trainable_parameters()

        # 如果启用了LoRA，展示LoRA相关功能
        if config.use_lora:
            print("\n=== LoRA功能测试 ===")

            # 保存LoRA适配器
            print(f"保存LoRA适配器到: {args.save_path}")
            policy.save_lora_adapters(args.save_path)

            # 获取LoRA状态字典
            lora_state_dict = policy.get_lora_state_dict()
            print(f"LoRA状态字典包含 {len(lora_state_dict)} 个参数")

            print("\nLoRA微调设置完成!")
            print("现在可以使用这个模型进行LoRA微调训练了。")

    except Exception as e:
        print(f"错误: {e}")
        if "PEFT" in str(e):
            print("请安装PEFT库: pip install peft")
        elif "transformers" in str(e):
            print("请安装transformers库: pip install transformers")


if __name__ == "__main__":
    main()
