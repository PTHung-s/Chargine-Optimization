import json
from collections import defaultdict
from datetime import datetime

def group_by_connection_date(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    grouped_data = defaultdict(list)
    
    for entry in data:
        date_str = entry["connectionTime"].split(", ")[1].split(" ")[0:3]  # Lấy phần ngày
        date = "-".join(date_str)  # Định dạng thành "25-Apr-2018"
        grouped_data[date].append(entry)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(grouped_data, f, indent=4)

# Ví dụ sử dụng
input_file = "1.json"
output_file = "final_gop.json"
group_by_connection_date(input_file, output_file)
