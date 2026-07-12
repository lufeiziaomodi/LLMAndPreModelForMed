#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型加载和配置模块
用于加载MedGemma模型并配置LoRA参数
"""

import os
import sys
import glob
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    set_seed
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    prepare_model_for_kbit_training
)


def set_random_seed(seed=42):
    """设置随机种子以确保可重复性
    
    Args:
        seed: 随机种子值
    """
    set_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def check_bitsandbytes():
    """
    检查bitsandbytes库是否已安装并打印版本信息
    """
    try:
        import bitsandbytes
        version = bitsandbytes.__version__
        print(f"已安装bitsandbytes库，版本: {version}")
        # 确保版本足够新
        from packaging import version as pkg_version
        min_version = pkg_version.parse("0.42.0")
        current_version = pkg_version.parse(version)
        if current_version < min_version:
            print(f"⚠️ bitsandbytes版本较低 ({version})，建议更新到最新版本")
        return True
    except ImportError:
        print("⚠️ 未安装bitsandbytes库")
        print("请运行以下命令安装:")
        print("pip install bitsandbytes")
        return False
    except Exception as e:
        print(f"⚠️ 检查bitsandbytes版本时出错: {str(e)}")
        return False


def load_model_and_tokenizer(model_id, use_4bit=False, use_8bit=False, token=None, use_multi_gpu=True):
    """加载模型和分词器，支持量化和多GPU
    
    Args:
        model_id: 模型ID或本地路径
        use_4bit: 是否使用4位量化
        use_8bit: 是否使用8位量化
        token: Hugging Face访问令牌，用于访问私有模型
        use_multi_gpu: 是否使用多GPU
        
    Returns:
        tuple: (model, tokenizer) 模型和分词器的元组
    """
    import os
    
    # 检查bitsandbytes库
    check_bitsandbytes()
    
    # 检查是否为本地模型
    is_local_model = os.path.isdir(model_id)
    print(f"开始加载模型: {model_id} ({'本地模型' if is_local_model else '远程模型'})")
    
    # 基本加载参数
    basic_kwargs = {
        'token': token,
        'trust_remote_code': True,
        'low_cpu_mem_usage': True  # 总是使用低内存加载策略
    }
    
    # 设置device_map
    if use_multi_gpu:
        basic_kwargs['device_map'] = 'balanced'  # 使用balanced以更好地分布模型权重
        print("启用多GPU模式，使用balanced device map")
    else:
        basic_kwargs['device_map'] = 'auto'
        print("单GPU模式，使用auto device map")
    
    # 根据模型类型加载
    if is_local_model:
        print("加载本地模型 - 使用修复后的配置")
        
        # 检查模型分片数量
        model_files = glob.glob(os.path.join(model_id, "model-*.safetensors"))
        num_shards = len(model_files)
        print(f"检测到{num_shards}个模型分片文件")
        
        # 加载策略：尝试多种参数组合以确保成功加载
        attempts = [
            ("标准参数", basic_kwargs),
            ("基本参数", {'trust_remote_code': True, 'device_map': 'auto'}),
            ("极简参数", {'trust_remote_code': True})
        ]
        
        model = None
        error_messages = []
        
        for attempt_name, kwargs in attempts:
            try:
                print(f"尝试 {attempt_name} 加载...")
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    **kwargs
                )
                print(f"✓ {attempt_name} 加载模型成功")
                break
            except Exception as e:
                error_msg = str(e)
                error_messages.append(f"{attempt_name}: {error_msg}")
                print(f"✗ {attempt_name} 加载失败: {error_msg}")
                
                # 特殊错误处理 - bitsandbytes相关错误
                if "bitsandbytes" in error_msg.lower():
                    print("\n⚠️ 错误分析: bitsandbytes库相关问题")
                    print("请运行以下命令更新bitsandbytes库:")
                    print("pip install -U bitsandbytes")
                    raise ImportError("需要更新bitsandbytes库，请运行: pip install -U bitsandbytes") from e
                # 分析错误类型
                elif "No such file or directory" in error_msg:
                    print("错误分析: 模型文件不完整，请确保有6个分片文件")
                elif "quantization type" in error_msg or "quant_method" in error_msg:
                    print("错误分析: 量化配置问题，请确保quant_method已正确设置")
        
        if model is None:
            print("\n所有加载尝试都失败了！")
            print("错误详情:")
            for msg in error_messages:
                print(f"- {msg}")
            
            print("\n建议解决方案:")
            print("1. 确保bitsandbytes库已更新: pip install -U bitsandbytes")
            print("2. 确保模型文件完整（应有6个分片文件）")
            print("3. 确认配置文件中的quant_method已设置为'bitsandbytes_8bit'")
            print("4. 尝试更新transformers库到最新版本")
            print("5. 检查CUDA内存是否足够")
            raise RuntimeError("无法加载模型，请检查上述建议")
            
    else:
        # 对于远程模型，根据参数添加量化配置
        print("加载远程模型")
        remote_kwargs = basic_kwargs.copy()
        remote_kwargs['torch_dtype'] = torch.float16  # 使用半精度
        
        # 添加量化配置
        if use_8bit:
            print("启用8位量化")
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0
            )
            remote_kwargs['quantization_config'] = quantization_config
        elif use_4bit:
            print("启用4位量化")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )
            remote_kwargs['quantization_config'] = quantization_config
        
        # 加载远程模型
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                **remote_kwargs
            )
            print("✓ 远程模型加载成功")
        except Exception as e:
            error_msg = str(e)
            print(f"✗ 远程模型加载失败: {error_msg}")
            
            # 特殊错误处理 - bitsandbytes相关错误
            if "bitsandbytes" in error_msg.lower():
                print("\n⚠️ 错误分析: bitsandbytes库相关问题")
                print("请运行以下命令更新bitsandbytes库:")
                print("pip install -U bitsandbytes")
                raise ImportError("需要更新bitsandbytes库，请运行: pip install -U bitsandbytes") from e
            
            raise
    
    # 加载分词器
    try:
        tokenizer_kwargs = {
            'token': token,
            'trust_remote_code': True,
            'padding_side': 'right'
        }
        tokenizer = AutoTokenizer.from_pretrained(model_id, **tokenizer_kwargs)
        print("✓ 分词器加载成功")
    except Exception as e:
        print(f"✗ 分词器加载失败: {str(e)}")
        raise
    
    # 确保分词器有pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("已设置分词器的pad_token为eos_token")
    
    # 准备模型用于k位训练
    model = prepare_model_for_kbit_training(model)
    
    print("\n✅ 模型和分词器加载完成！")
    return model, tokenizer


def configure_lora(model, r=8, lora_alpha=16, lora_dropout=0.05):
    """配置LoRA参数并创建Peft模型
    
    Args:
        model: 基础模型
        r: LoRA注意力维度
        lora_alpha: LoRA alpha参数
        lora_dropout: LoRA dropout概率
        
    Returns:
        PeftModel: 配置好LoRA的模型
    """
    # LoRA配置
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=[
            "q_proj", "v_proj", "k_proj", "o_proj",  # 注意力层投影
            "gate_proj", "up_proj", "down_proj"  # MLP层
        ],
        bias="none",
        lora_dropout=lora_dropout,
        task_type=TaskType.CAUSAL_LM,
        # 对于密集连接层也应用LoRA
        modules_to_save=None
    )
    
    # 创建Peft模型
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    
    return peft_model


def setup_training_args(output_dir, per_device_train_batch_size=2, per_device_eval_batch_size=2,
                       gradient_accumulation_steps=4, learning_rate=2e-4, max_steps=1000,
                       logging_steps=50, save_steps=200, eval_steps=100, warmup_ratio=0.03,
                       use_multi_gpu=True):
    """设置训练参数
    
    Args:
        output_dir: 输出目录
        per_device_train_batch_size: 每个设备的训练批量大小
        per_device_eval_batch_size: 每个设备的评估批量大小
        gradient_accumulation_steps: 梯度累积步数
        learning_rate: 学习率
        max_steps: 最大训练步数
        logging_steps: 日志记录步数
        save_steps: 模型保存步数
        eval_steps: 评估步数
        warmup_ratio: 预热比例
        
    Returns:
        TrainingArguments: 训练参数对象
    """
    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        max_steps=max_steps,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        evaluation_strategy="steps",
        warmup_ratio=warmup_ratio,
        lr_scheduler_type="cosine",
        fp16=True,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="tensorboard",  # 可选：使用TensorBoard记录训练过程
        # 分布式训练设置
        distributed_training=True if use_multi_gpu else False,
        # 多GPU优化
        ddp_find_unused_parameters=False,
        # 梯度检查点（节省内存）
        gradient_checkpointing=True
    )