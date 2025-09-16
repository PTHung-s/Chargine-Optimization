import json

def count_objects_in_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    count = sum(len(objects) for objects in data.values())
    return count

# Thay 'data.json' bằng đường dẫn file JSON của bạn
file_path = 'final_gop.json'
num_objects = count_objects_in_json(file_path)
print(f'Tổng số object trong file JSON: {num_objects}')
