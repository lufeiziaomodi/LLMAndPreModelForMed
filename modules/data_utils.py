#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据加载和预处理模块
用于加载并预处理DDI关系识别任务的数据集
"""

import pandas as pd
from datasets import Dataset, DatasetDict


def load_and_preprocess_data(train_csv_path, test_csv_path=None, test_size=0.1):
    """加载并预处理CSV格式的数据集
    
    Args:
        train_csv_path: 训练数据CSV文件路径
        test_csv_path: 测试数据CSV文件路径（可选）
        test_size: 从训练集划分验证集的比例，仅在未提供test_csv_path时使用
        
    Returns:
        DatasetDict: 包含train和validation数据集的字典
    """
    print(f"Loading dataset from {train_csv_path}")
    df = pd.read_csv(train_csv_path)
    
    # 确保输入列名正确
    if 'input_text' not in df.columns or 'output_text' not in df.columns:
        raise ValueError("Dataset must contain 'input_text' and 'output_text' columns")
    
    # 划分训练集和验证集
    train_dataset = Dataset.from_pandas(df)
    
    if test_csv_path and os.path.exists(test_csv_path):
        # 如果提供了测试集路径，使用该数据集作为验证集
        test_df = pd.read_csv(test_csv_path)
        val_dataset = Dataset.from_pandas(test_df)
    else:
        # 否则从训练集中划分一部分作为验证集
        train_test_split = train_dataset.train_test_split(test_size=test_size, seed=42)
        train_dataset = train_test_split['train']
        val_dataset = train_test_split['test']
    
    return DatasetDict({
        'train': train_dataset,
        'validation': val_dataset
    })


def format_dataset(example, tokenizer):
    """将数据集格式化为模型所需的格式
    
    Args:
        example: 单个数据样本
        tokenizer: 分词器
        
    Returns:
        dict: 格式化后的数据
    """
    # 按照input_text和output_text的格式拼接
    # 注意：MedGemma模型需要特定的格式，这里使用简单的格式
    messages = [
        {"role": "user", "content": example["input_text"]},
        {"role": "assistant", "content": example["output_text"]}
    ]
    
    # 使用tokenizer的apply_chat_template方法格式化聊天记录
    # 对于MedGemma，可能需要调整chat_template
    formatted_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    
    return {"text": formatted_text}


# 需要导入os模块
import os