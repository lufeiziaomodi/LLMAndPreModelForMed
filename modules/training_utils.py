#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练模块
用于训练配置好的MedGemma模型
"""

import os
from trl import SFTTrainer


def train_model(peft_model, tokenizer, dataset, training_args, formatting_func, packing=False):
    """使用SFTTrainer训练模型
    
    Args:
        peft_model: 配置好LoRA的模型
        tokenizer: 分词器
        dataset: 数据集
        training_args: 训练参数
        formatting_func: 数据格式化函数
        packing: 是否使用数据打包
        
    Returns:
        SFTTrainer: 训练器对象
    """
    # 创建Trainer
    trainer = SFTTrainer(
        model=peft_model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        packing=packing,
        max_seq_length=tokenizer.model_max_length if hasattr(tokenizer, 'model_max_length') else 4096,
        formatting_func=formatting_func
    )
    
    # 开始训练
    print("Starting training...")
    trainer.train()
    
    # 保存最终模型
    final_model_dir = os.path.join(training_args.output_dir, "final_model")
    trainer.save_model(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    
    print(f"Training completed! Model saved to {final_model_dir}")
    return trainer