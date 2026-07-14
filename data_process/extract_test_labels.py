"""
从 DDIcorpus 测试集（rbert_test.csv）提取五种标签类型的数据，生成对应的 JSON 文件

标签类型：
- mechanism: 直接表示机制的文本
- effect: 说明现象的文本
- int: 只展示药物间存在相互作用的文本
- advise: 对药物用法提出建议的文本
- false: 药物无关系的文本
"""

import re
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

# Allow running this file directly via: python data_process/extract_test_labels.py
_HERE = Path(__file__).resolve()
if str(_HERE.parent.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent.parent))

from data_process.paths import DATA_LABELS, RBERT_TEST_CSV, ensure_dir


def extract_test_labels(input_file, output_dir):
    """
    从测试集提取五种标签类型的数据并分别输出
    
    Args:
        input_file: 输入的测试集文件路径（rbert_test.csv）
        output_dir: 输出目录
    """
    # 目标标签
    target_labels = {'false', 'effect', 'mechanism', 'int', 'advise'}
    
    # 数据结构：label -> 唯一句子 -> 关系列表
    data_by_label = defaultdict(lambda: defaultdict(list))
    
    # 提取实体的正则表达式
    e1_pattern = re.compile(r'<e1>(.*?)</e1>')
    e2_pattern = re.compile(r'<e2>(.*?)</e2>')
    
    print(f"读取测试集文件: {input_file}")
    
    if not os.path.exists(input_file):
        print(f"错误：输入文件不存在: {input_file}")
        return
    
    # 尝试多种编码
    encodings = ['utf-8', 'latin-1', 'gbk', 'utf-16']
    lines = None
    
    for encoding in encodings:
        try:
            with open(input_file, 'r', encoding=encoding) as f:
                lines = f.readlines()
            print(f"使用编码: {encoding}")
            break
        except UnicodeDecodeError:
            continue
    
    if lines is None:
        print("错误：无法以任何编码读取文件")
        return
    
    count_total = 0
    count_processed = 0
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        count_total += 1
        
        # 按 tab 分割
        parts = line.split('\t')
        
        if len(parts) < 2:
            print(f"警告：跳过格式错误的行 (无 tab 分割): {line[:50]}...")
            continue
        
        label = parts[0].strip()
        text = parts[1].strip()
        
        if label not in target_labels:
            print(f"警告：跳过未知标签 '{label}'")
            continue
        
        # 提取实体
        e1_match = e1_pattern.search(text)
        e2_match = e2_pattern.search(text)
        
        if not e1_match or not e2_match:
            print(f"警告：跳过无实体标记的行: {text[:50]}...")
            continue
        
        e1 = e1_match.group(1)
        e2 = e2_match.group(1)
        
        # 标准化文本：移除标记
        normalized_text = text.replace('<e1>', '').replace('</e1>', '')
        normalized_text = normalized_text.replace('<e2>', '').replace('</e2>', '')
        
        # 存储
        data_by_label[label][normalized_text].append({
            'e1': e1,
            'e2': e2,
            'label': label
        })
        count_processed += 1
    
    print(f"总行数: {count_total}")
    print(f"处理成功: {count_processed}")
    
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 为每个标签输出文件
    total_entries = 0
    for label in sorted(target_labels):
        output_file = os.path.join(output_dir, f"{label}_test.json")
        
        output_list = []
        sentences = data_by_label[label]
        
        for text, relations in sentences.items():
            entry = {
                "text": text,
                "relations": relations
            }
            output_list.append(entry)
        
        total_entries += len(output_list)
        
        print(f"输出: {len(output_list):>4} 条唯一句子 -> {output_file}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_list, f, indent=2, ensure_ascii=False)
    
    print(f"\n总计生成: {total_entries} 条测试数据")
    print(f"已保存到: {output_dir}")


def main():
    """主函数：从测试集提取标签数据"""
    # 输入：data/raw/rbert_test.csv；输出：data/labels/{label}_test.json
    input_path = str(RBERT_TEST_CSV)
    output_dir = str(ensure_dir(DATA_LABELS))

    print("=" * 60)
    print("从 DDIcorpus 测试集提取五种标签类型的数据")
    print("=" * 60)

    extract_test_labels(input_path, output_dir)


if __name__ == "__main__":
    main()
