import json
from collections import defaultdict

# Đọc dữ liệu từ file JSON
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

# Phân tích dữ liệu
def analyze_charging_sessions(data):
    daily_counts = defaultdict(int)
    
    for date, sessions in data.items():
        daily_counts[date] = len(sessions)
    
    min_day = min(daily_counts, key=daily_counts.get)
    max_day = max(daily_counts, key=daily_counts.get)
    avg_sessions = sum(daily_counts.values()) / len(daily_counts)
    
    return min_day, daily_counts[min_day], max_day, daily_counts[max_day], avg_sessions

# Đường dẫn đến file JSON
file_path = 'final_gop.json'  # Thay thế bằng đường dẫn thực tế

data = load_json(file_path)
min_day, min_count, max_day, max_count, avg_sessions = analyze_charging_sessions(data)

print(f"Ngày có ít xe nhất: {min_day} với {min_count} xe")
print(f"Ngày có nhiều xe nhất: {max_day} với {max_count} xe")
print(f"Giá trị trung bình xe vào sạc mỗi ngày: {avg_sessions:.2f}")
