import re
import json
import os
from collections import defaultdict

def process_rbert_data(input_file, output_dir):
    # Target labels
    target_labels = {'false', 'effect', 'mechanism', 'int', 'advise'}
    
    # Data structure: label -> normalized_text -> list of relations
    data_by_label = defaultdict(lambda: defaultdict(list))
    
    # Regex to find tags and content
    e1_pattern = re.compile(r'<e1>(.*?)</e1>')
    e2_pattern = re.compile(r'<e2>(.*?)</e2>')
    
    print(f"Reading from {input_file}...")
    
    if not os.path.exists(input_file):
        print(f"Error: Input file not found at {input_file}")
        return

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        print("Utf-8 decode error, trying latin-1")
        with open(input_file, 'r', encoding='latin-1') as f:
            lines = f.readlines()

    count = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Split by tab as requested
        parts = line.split('\t')
        
        # If strict splitting by tab fails to produce at least 2 parts, skip or warn
        if len(parts) < 2:
            print(f"Skipping line (no tab separator found): {line[:50]}...")
            continue
            
        label = parts[0].strip()
        # The text is the rest of the line (in case there are tabs in the text, though unlikely for this format)
        # Usually it is label \t text. 
        # But if there are extra tabs, we might want to join them or take the second part.
        # Let's assume the format is strictly LABEL \t SENTENCE
        text = parts[1].strip()
        
        if label not in target_labels:
            continue
            
        # Extract entities
        e1_match = e1_pattern.search(text)
        e2_match = e2_pattern.search(text)
        
        if not e1_match or not e2_match:
            print(f"Skipping line without entity tags: {text[:50]}...")
            continue
            
        e1 = e1_match.group(1)
        e2 = e2_match.group(1)
        
        # Normalize text: remove tags
        # We replace <e1>X</e1> with X
        normalized_text = text.replace('<e1>', '').replace('</e1>', '')
        normalized_text = normalized_text.replace('<e2>', '').replace('</e2>', '')
        
        # Store
        data_by_label[label][normalized_text].append({
            'e1': e1,
            'e2': e2,
            'label': label
        })
        count += 1

    print(f"Processed {count} lines.")

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Write output files
    for label in target_labels:
        output_file = os.path.join(output_dir, f"{label}.json")
        
        output_list = []
        sentences = data_by_label[label]
        
        for text, relations in sentences.items():
            entry = {
                "text": text,
                "relations": relations
            }
            output_list.append(entry)
            
        print(f"Writing {len(output_list)} unique sentences to {output_file}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_list, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    # Calculate paths relative to the script location
    # Script is in data_process/, so we go up one level to root, then into data/
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    input_path = os.path.join(project_root, "data", "standardized_generated_rbert_train_random.csv")
    output_dir = os.path .join(project_root, "data")
    
    process_rbert_data(input_path, output_dir)
