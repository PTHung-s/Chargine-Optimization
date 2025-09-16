import json
from datetime import datetime
from collections import defaultdict

def calculate_monthly_electricity_cost(json_file_path):
    # Đọc file JSON
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    
    # Tạo dictionary để lưu tổng chi phí theo tháng
    monthly_costs = defaultdict(float)
    
    # Giả sử mỗi giờ tiêu thụ 1kWh (có thể điều chỉnh theo thực tế)
    consumption_per_hour = 1.0  # kWh
    
    # Duyệt qua từng ngày trong dữ liệu
    for date_str, hourly_prices in data.items():
        # Chuyển chuỗi ngày thành đối tượng datetime
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        # Lấy năm và tháng dưới dạng chuỗi (ví dụ: "2019-12")
        month_key = date_obj.strftime('%Y-%m')
        
        # Kiểm tra xem có đúng 24 giá trị không
        if len(hourly_prices) != 24:
            print(f"Cảnh báo: Ngày {date_str} không có đủ 24 giá trị giờ")
            continue
        
        # Tính tổng chi phí cho ngày đó
        daily_cost = sum(price * consumption_per_hour for price in hourly_prices)
        
        # Cộng vào tổng chi phí tháng
        monthly_costs[month_key] += daily_cost
    
    # In kết quả
    print("Tổng tiền điện theo tháng:")
    for month, total_cost in sorted(monthly_costs.items()):
        print(f"Tháng {month}: {total_cost:.2f} (đơn vị tiền tệ)")
    
    return monthly_costs

# Ví dụ sử dụng
if __name__ == "__main__":
    # Giả sử file của bạn tên là 'electricity_prices.json'
    try:
        json_file_path = 'price.json'
        monthly_costs = calculate_monthly_electricity_cost(json_file_path)
    except FileNotFoundError:
        print("Không tìm thấy file JSON. Vui lòng kiểm tra đường dẫn file.")
    except json.JSONDecodeError:
        print("Lỗi định dạng JSON. Vui lòng kiểm tra file dữ liệu.")