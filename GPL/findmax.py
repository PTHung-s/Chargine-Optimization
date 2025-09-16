import json
from datetime import datetime

# Định dạng ngày giờ trong dữ liệu
DATE_FORMAT = '%a, %d %b %Y %H:%M:%S GMT'

# File JSON
input_file = '1.json'  # File JSON đầu vào
output_file = 'max.json'  # File JSON đầu ra

try:
    # Đọc file JSON
    with open(input_file, 'r') as file:
        data = json.load(file)

    max_rate = float('-inf')  # Giá trị lớn nhất ban đầu là âm vô cực
    max_object = None
    filtered_data = []  # Dữ liệu sau khi lọc

    for obj in data:
        try:
            # Kiểm tra xem các trường thời gian có tồn tại và không phải None
            if ('connectionTime' in obj and obj['connectionTime'] and 
                'doneChargingTime' in obj and obj['doneChargingTime'] and 
                'kWhDelivered' in obj):
                
                # Chuyển đổi thời gian từ chuỗi sang datetime
                connect_time = datetime.strptime(obj['connectionTime'], DATE_FORMAT)
                done_time = datetime.strptime(obj['doneChargingTime'], DATE_FORMAT)
                
                # Tính thời gian chênh lệch (theo giờ)
                time_diff = (done_time - connect_time).total_seconds() / 3600
                
                # Tính kWhDelivered / thời gian (kW trung bình)
                if time_diff > 0:
                    rate = obj['kWhDelivered'] / time_diff
                    doubled_rate = rate * 2
                    
                    # Nếu doubled_rate <= 100, giữ lại object
                    if doubled_rate <= 100:
                        filtered_data.append(obj)
                    
                    # Cập nhật max_rate nếu lớn hơn
                    if rate > max_rate:
                        max_rate = rate
                        max_object = obj
            else:
                print(f"Object thiếu dữ liệu cần thiết: {obj}")
                # Vẫn giữ lại object nếu không tính được rate
                filtered_data.append(obj)

        except (KeyError, ValueError) as e:
            print(f"Lỗi khi xử lý object: {obj} - {e}")
            # Giữ lại object nếu có lỗi (tùy bạn có muốn xóa không)
            filtered_data.append(obj)
            continue

    if max_rate != float('-inf'):
        # Nhân đôi giá trị lớn nhất
        doubled_max_rate = max_rate * 2
        
        # Tạo dữ liệu đầu ra theo cấu trúc yêu cầu
        output_data = {
            'max_rate': max_rate,
            'doubled_max_rate': doubled_max_rate,
            'original_object': max_object
        }
        
        # Lưu thông tin max vào max.json
        json_data = json.dumps(output_data, indent=2)
        with open(output_file, 'w') as file:
            file.write(json_data)
        
        # Lưu lại dữ liệu đã lọc vào 1.json
        with open(input_file, 'w') as file:
            json.dump(filtered_data, file, indent=2)
        
        print(f"Giá trị lớn nhất: {max_rate}")
        print(f"Giá trị nhân đôi: {doubled_max_rate}")
        print(f"Đã lưu max vào {output_file}")
        print(f"Đã cập nhật file {input_file} (xóa các object có doubled_rate > 100)")
    else:
        print("Không tìm thấy giá trị hợp lệ nào trong dữ liệu")

except FileNotFoundError:
    print(f"Không tìm thấy file: {input_file}")
except Exception as e:
    print(f"Lỗi không xác định: {e}")