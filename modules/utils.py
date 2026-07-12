#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
工具模块
用于生成推理脚本和依赖文件等辅助功能
"""

import os

def generate_inference_script(output_dir):
    """生成一个用于推理的简单脚本
    
    Args:
        output_dir: 输出目录
    """
    inference_script = '''
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MedGemma-27B-IT-8bit LoRA 推理脚本
用于药物-药物相互作用(DDI)关系识别任务
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

class DDIInference:
    def __init__(self, base_model_id, peft_model_path, token=None):
        """初始化推理模型
        
        Args:
            base_model_id: 基础模型ID
            peft_model_path: LoRA适配器路径
            token: Hugging Face访问令牌，用于访问私有模型
        """
        # 准备模型加载参数
        model_kwargs = {
            "torch_dtype": torch.float16,
            "device_map": "auto",
            "trust_remote_code": True
        }
        
        # 如果提供了token，则添加到参数中
        if token:
            model_kwargs["token"] = token
            print("使用提供的token访问模型")
        
        # 加载基础模型
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_id, **model_kwargs)
        self.model = AutoModelForCausalLM.from_pretrained(base_model_id, **model_kwargs)
        
        # 加载LoRA适配器
        self.model = PeftModel.from_pretrained(
            self.model,
            peft_model_path,
            torch_dtype=torch.float16
        )
        
        # 设置为评估模式
        self.model.eval()
    
    def generate_response(self, input_text, max_new_tokens=20):
        """生成模型响应
        
        Args:
            input_text: 输入文本
            max_new_tokens: 最大生成长度
            
        Returns:
            生成的文本
        """
        # 准备聊天模板
        messages = [
            {"role": "user", "content": input_text}
        ]
        
        # 应用聊天模板
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 编码输入
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # 生成输出
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.1,
                top_p=0.95,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        # 解码输出
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        return response.strip()

# 使用示例
if __name__ == "__main__":
    # 初始化推理模型
    # 注意：google/medgemma-27b-it-8bit 可能不是公共可用的模型
    # 您可以使用以下选项之一：
    # 1. 使用公共可用的MedGemma模型
    # 2. 使用本地模型路径
    # 3. 如果是私有模型，添加token参数
    
    # 选项1：使用本地MedGemma模型（推荐）
    base_model_id = "models/google/medgemma-27b-it-8bit"
    
    # 选项2（备选）：使用其他本地模型路径
    # base_model_id = "./other_local_model_path"
    
    # 选项3（备选）：使用私有模型并提供token
    # token = "your_huggingface_token"
    # ddi_inference = DDIInference(
    #     base_model_id="google/medgemma-27b-it-8bit",
    #     peft_model_path="{output_dir}/final_model",
    #     token=token
    # )
    
    ddi_inference = DDIInference(
        base_model_id=base_model_id,
        peft_model_path="{output_dir}/final_model"
    )
    
    # 示例输入
    sample_input = """Task: Determine the DDI relationship type between <e1> and <e2> in the text. Only output the label (false/mechanism/effect/advise/int).
Text: The concomitant administration of <e1>rifampin</e1> and <e2>warfarin</e2> resulted in the need for an unusually high maintenance dose of warfarin."""
    
    # 生成响应
    result = ddi_inference.generate_response(sample_input)
    print(f"\nInput: {sample_input}")
    print(f"\nPredicted DDI relationship: {result}")
    
    # 可以批量处理更多输入
    # more_samples = [...]  # 更多输入样本
    # for sample in more_samples:
    #     result = ddi_inference.generate_response(sample)
    #     print(f"\nInput: {sample}")
    #     print(f"Predicted DDI relationship: {result}")
'''
    
    # 替换output_dir占位符
    inference_script = inference_script.format(output_dir=output_dir)
    
    script_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "inference_medgemma_lora.py")
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(inference_script)
    
    print(f"\nInference script generated: {script_path}")


def create_requirements_file():
    """创建requirements.txt文件，列出所需的依赖包"""
    requirements = """
# 核心依赖
transformers==4.36.0
torch==2.1.0
accelerate==0.25.0
peft==0.7.0
trl==0.7.10
datasets==2.15.0
pandas==2.1.4
numpy==1.24.3

# 可选依赖（用于量化）
bitsandbytes==0.41.3

# 可视化
tensorboard==2.15.1
matplotlib==3.8.2
seaborn==0.12.2
    """
    
    requirements_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "requirements.txt")
    with open(requirements_path, "w", encoding="utf-8") as f:
        f.write(requirements)
    
    print(f"\nrequirements.txt generated: {requirements_path}")