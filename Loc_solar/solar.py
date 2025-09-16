import json

# Đọc file JSON gốc
input_file = 'output.json'  # File JSON đã tạo trước đó
output_file = 'solar.json'  # File JSON mới sẽ được tạo

try:
    with open(input_file, 'r') as file:
        data = json.load(file)

    # Tạo dữ liệu mới với R(i) = 80 * G(i) * 0.18
    new_data = []
    for item in data:
        g_value = item['G(i)']
        r_value = (80 * g_value * 0.2) / 1000  # Tính R(i)
        new_data.append({
            'time': item['time'],
            'R(i)': r_value
        })

    # Chuyển thành chuỗi JSON với định dạng đẹp
    json_data = json.dumps(new_data, indent=2)

    # In kết quả để kiểm tra
    print(json_data)

    # Lưu vào file mới
    with open(output_file, 'w') as file:
        file.write(json_data)
    print(f"Đã tạo file mới: {output_file}")

except FileNotFoundError:
    print(f"Không tìm thấy file: {input_file}")
except Exception as e:
    print(f"Lỗi: {e}")