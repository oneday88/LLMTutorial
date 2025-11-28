import os
import json
### load data
def load_jsonl(filepath):
    """Loads a JSONL (JSON Lines) file and returns a list of dictionaries."""
    data = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    data.append(item)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON on line: {line.strip()}")
                    print(f"Error details: {e}")
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return []
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return []
    return data


def fix_data_types(data_list):
    """修复数据类型问题"""
    fixed_data = []
    for item in data_list:
        fixed_item = {}
        for key, value in item.items():
            # 检查并转换数据类型
            if isinstance(value, float):
                # 如果应该是字符串，转换为字符串
                fixed_item[key] = str(value)
            elif isinstance(value, (list, tuple)) and len(value) > 0 and isinstance(value[0], float):
                # 如果是浮点数列表，根据需求处理
                fixed_item[key] = [str(v) for v in value]  # 或者保持为float，但需要统一
            else:
                fixed_item[key] = value
        fixed_data.append(fixed_item)
    return fixed_data
