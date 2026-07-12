import pandas as pd
import numpy as np
import re
import random
from collections import defaultdict
import hashlib

# Set random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Configuration parameters
INPUT_CSV_PATH = "../data/rbert_test.csv"
OUTPUT_CSV_PATH = "../data/test_augmented_data.csv"
HIGH_REPEAT_THRESHOLD = 3  # Threshold for high repetition text
LABEL_TARGET_NUM = {  # Target number for each label (for balance)
    "false": 1600,
    "mechanism": 320,
    "effect": 280,
    "advise": 200,
    "int": 90
}
# Label priority (smaller categories first)
LABEL_PRIORITY = ["int", "advise", "mechanism", "effect", "false"]

def extract_entities(text):
    """Extract current entity pair from text"""
    e1_match = re.search(r'<e1>(.*?)</e1>', text)
    e2_match = re.search(r'<e2>(.*?)</e2>', text)
    if e1_match and e2_match:
        return e1_match.group(1), e2_match.group(1)
    return None, None

def remove_entity_tags(text):
    """Remove entity tags from text"""
    text = re.sub(r'</?e1>', '', text)
    text = re.sub(r'</?e2>', '', text)
    return text

def generate_text_id(text):
    """Generate unique ID for text"""
    base_text = remove_entity_tags(text)
    return hashlib.md5(base_text.encode()).hexdigest()[:10]

def main():
    print("=== Step 1: Data Preprocessing - Split 'Text-Entity Pair-Label' Relationships ===")
    # Read data
    df = pd.read_csv(INPUT_CSV_PATH, sep='\t', header=None, names=['label', 'text'])
    print(f"Original data rows: {len(df)}")
    print(f"Label distribution:")
    print(df['label'].value_counts())
    
    # 1. Structured data storage with detailed entity pair information
    text_entity_dict = defaultdict(list)  # {text_id: [(e1, e2, label, original_text)]}
    text_id_mapping = {}  # {text_id: base_text_without_tags}
    
    for _, row in df.iterrows():
        text = row['text']
        label = row['label']
        text_id = generate_text_id(text)
        e1, e2 = extract_entities(text)
        
        # Store base text without tags
        if text_id not in text_id_mapping:
            text_id_mapping[text_id] = remove_entity_tags(text)
        
        # Store entity pair information
        text_entity_dict[text_id].append((e1, e2, label, text))
    
    print(f"\nNumber of unique text templates: {len(text_entity_dict)}")
    
    # 2. Classify texts based on entity pair repetition
    low_repeat_texts = []  # Texts with ≤ HIGH_REPEAT_THRESHOLD entity pairs
    high_repeat_texts = []  # Texts with > HIGH_REPEAT_THRESHOLD entity pairs
    
    for text_id, entity_pairs in text_entity_dict.items():
        if len(entity_pairs) <= HIGH_REPEAT_THRESHOLD:
            low_repeat_texts.append((text_id, entity_pairs))
        else:
            high_repeat_texts.append((text_id, entity_pairs))
    
    print(f"Low repetition texts (≤{HIGH_REPEAT_THRESHOLD} entity pairs): {len(low_repeat_texts)}")
    print(f"High repetition texts (> {HIGH_REPEAT_THRESHOLD} entity pairs): {len(high_repeat_texts)}")
    
    print("\n=== Step 2: Core Processing - Text Template Deduplication + Entity Pair Coverage ===")
    # Process low repeat texts - keep all entity pairs
    filtered_data = []
    for text_id, entity_pairs in low_repeat_texts:
        for e1, e2, label, original_text in entity_pairs:
            filtered_data.append((original_text, label))
    
    # Process high repeat texts - select representative entity pairs
    for text_id, entity_pairs in high_repeat_texts:
        # Count entity occurrences for prioritization
        entity_counts = defaultdict(int)
        for e1, e2, _, _ in entity_pairs:
            entity_counts[e1] += 1
            entity_counts[e2] += 1
        
        # Sort entity pairs by combined entity frequency
        sorted_pairs = sorted(entity_pairs, key=lambda x: entity_counts[x[0]] + entity_counts[x[1]], reverse=True)
        
        # Keep top 5 entity pairs (or all if less than 5)
        top_pairs = sorted_pairs[:5]
        for e1, e2, label, original_text in top_pairs:
            filtered_data.append((original_text, label))
    
    print(f"After initial filtering, total data rows: {len(filtered_data)}")
    
    print("\n=== Step 3: Category Balance Adjustment ===")
    # Group filtered data by label
    filtered_by_label = defaultdict(list)
    for text, label in filtered_data:
        filtered_by_label[label].append((text, label))
    
    # Group original data by label for additional sampling if needed
    original_by_label = defaultdict(list)
    for _, row in df.iterrows():
        original_by_label[row['label']].append((row['text'], row['label']))
    
    # Adjust category balance based on target numbers
    final_filtered_data = []
    label_count = {label: 0 for label in LABEL_TARGET_NUM}
    
    # First pass: take as much as possible from filtered data
    for label in LABEL_PRIORITY:
        if label in filtered_by_label:
            take_count = min(LABEL_TARGET_NUM[label], len(filtered_by_label[label]))
            selected = random.sample(filtered_by_label[label], take_count)
            final_filtered_data.extend(selected)
            label_count[label] = take_count
    
    # Second pass: supplement from original data if needed
    for label in LABEL_PRIORITY:
        need = LABEL_TARGET_NUM[label] - label_count[label]
        if need > 0 and label in original_by_label:
            # Take random samples from original data
            additional_samples = random.sample(original_by_label[label], min(need, len(original_by_label[label])))
            final_filtered_data.extend(additional_samples)
            label_count[label] += len(additional_samples)
    
    # Final label distribution check
    print(f"\nFinal processed label counts:")
    final_counts = defaultdict(int)
    for text, label in final_filtered_data:
        final_counts[label] += 1
    
    for label in LABEL_PRIORITY:
        print(f"{label}: {final_counts.get(label, 0)}/{LABEL_TARGET_NUM.get(label, 0)}")
    
    print("\n=== Step 4: Save Augmented Data ===")
    # Generate simple output format with only two columns: label and text (in this order)
    augmented_data = []
    for text, label in final_filtered_data:
        augmented_data.append({
            "label": label,
            "text": text
        })
    
    # Convert to DataFrame
    df_augmented = pd.DataFrame(augmented_data)
    
    # Shuffle data
    df_augmented = df_augmented.sample(frac=1, random_state=RANDOM_SEED)
    
    # Save to CSV with tab delimiter, no header row
    df_augmented.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8", sep='\t', header=False)
    
    print(f"\n=== Data Augmentation Completed ===")
    print(f"Original data rows: {len(df)}")
    print(f"Augmented data rows: {len(df_augmented)}")
    print(f"Final label distribution:")
    print(df_augmented['label'].value_counts())
    print(f"Results saved to: {OUTPUT_CSV_PATH}")
    print(f"Output format: CSV with no header, columns label and text, tab-delimited")

if __name__ == "__main__":
    main()