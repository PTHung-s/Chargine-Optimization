import pandas as pd
import json

# Đọc file CSV
file_path = "1.csv"  # Đổi thành đường dẫn file thực tế
df = pd.read_csv(file_path)

# Kiểm tra đầu vào để đảm bảo dữ liệu đã được tách đúng
print(df.head())

# Tạo cấu trúc JSON
result = {}
for _, row in df.iterrows():
    # Sử dụng _id làm key cho mỗi object
    _id = row["_id"]
    
    # Lấy các giá trị cần thiết từ các cột khác
    session_data = {
        "connectionTime": row["connectionTime"],
        "disconnectionTime": row["disconnectTime"],
        "kWhDelivered": row["kWhDelivered"],
        "doneChargingTime": row["doneChargingTime"]
    }
    
    # Thêm thông tin vào kết quả với key là _id
    result[_id] = session_data

# Chuyển sang JSON
json_output = json.dumps(result, indent=4)

# Ghi vào file JSON
output_file = "charging_sessions_filtered.json"
with open(output_file, "w") as f:
    f.write(json_output)

print(f"Dữ liệu đã được lưu vào {output_file}")
