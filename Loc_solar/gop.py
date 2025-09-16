import json
from collections import defaultdict

def group_by_date(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    grouped_data = defaultdict(list)
    
    for entry in data:
        date = entry["time"].split(":")[0][:8]  # Lấy phần ngày từ chuỗi "time"
        grouped_data[date].append(entry)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(grouped_data, f, indent=4)

# Ví dụ sử dụng
input_file = "solar.json"
output_file = "gop_solar.json"
group_by_date(input_file, output_file)