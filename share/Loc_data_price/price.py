import pandas as pd
import json

# Đọc file CSV với dấu phân cách là dấu phẩy (,) 
file_path = "Spain.csv"  # Đổi thành đường dẫn file thực tế
df = pd.read_csv(file_path, delimiter=',')  # Đọc tệp CSV với dấu phân cách là dấu phẩy

# Kiểm tra đầu vào để đảm bảo dữ liệu đã được tách đúng
print(df.head())

# Chuyển đổi cột thời gian sang dạng datetime
df["Datetime (UTC)"] = pd.to_datetime(df["Datetime (UTC)"])

# Lọc dữ liệu theo khoảng thời gian yêu cầu
start_date = "2018-04-25 00:00:00"
end_date = "2019-12-31 23:00:00"
filtered_df = df[(df["Datetime (UTC)"] >= start_date) & (df["Datetime (UTC)"] <= end_date)]

# Tạo cấu trúc JSON
result = {}
for _, row in filtered_df.iterrows():
    date_str = row["Datetime (UTC)"].strftime("%Y-%m-%d")
    hour = row["Datetime (UTC)"].hour
    price = row["Price (EUR/MWhe)"]
    
    if date_str not in result:
        result[date_str] = [None] * 24  # Tạo một mảng 24 phần tử ban đầu là None
    
    result[date_str][hour] = price  # Gán giá trị giá điện vào đúng vị trí giờ

# Chuyển sang JSON
json_output = json.dumps(result, indent=4)

# Ghi vào file JSON
output_file = "price.json"
with open(output_file, "w") as f:
    f.write(json_output)

print(f"Dữ liệu đã được lưu vào {output_file}")
