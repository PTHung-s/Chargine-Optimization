import csv
import json

data = []
file_path = 'Madrid_Spain.csv'  # Thay bằng tên file thực tế, ví dụ: 'data.csv'

try:
    with open(file_path, 'r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            # Kiểm tra dữ liệu trước khi xử lý
            if 'time' in row and 'G(i)' in row:
                try:
                    # Làm tròn thời gian về giờ chẵn
                    time = row['time']
                    date_time = time.split(':')
                    rounded_time = f"{date_time[0]}:{date_time[1][-4:-2]}00"
                    
                    # Chuyển G(i) thành float
                    g_value = float(row['G(i)']) if row['G(i)'] else 0.0
                    
                    data.append({
                        'time': rounded_time,
                        'G(i)': g_value
                    })
                except (ValueError, IndexError) as e:
                    print(f"Dòng lỗi: {row} - Lỗi: {e}")
                    continue
            else:
                print(f"Dòng thiếu cột cần thiết: {row}")
                continue

    # Chuyển thành JSON nếu có dữ liệu
    if data:
        json_data = json.dumps(data, indent=2)
        print(json_data)
        
        with open('output.json', 'w') as json_file:
            json_file.write(json_data)
    else:
        print("Không có dữ liệu hợp lệ để xử lý")

except FileNotFoundError:
    print(f"Không tìm thấy file: {file_path}")
except Exception as e:
    print(f"Lỗi không xác định: {e}")