import json
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np  # Import numpy

# --------------------------
# 🔹 Hàm đọc dữ liệu từ JSON
# --------------------------
def load_json(filename):
    """Đọc file JSON và trả về dữ liệu dưới dạng dict (key: month, value: giá trị)"""
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file {filename}")
        return {} # Trả về dict rỗng nếu file không tồn tại
    except json.JSONDecodeError:
        print(f"Lỗi: File {filename} không phải là JSON hợp lệ.")
        return {}

# --------------------------
# 🔹 Đọc dữ liệu từ các file JSON đã lưu trước đó
# --------------------------
# Thay đổi đường dẫn nếu cần
path_optimized = "../2data_optimized/"
path_fcfs = "../2data_fcfs/"

monthly_consumption_with = load_json(f"{path_optimized}monthly_cost_with_solar.json")
monthly_consumption_without = load_json(f"{path_optimized}monthly_cost_no_solar.json")
monthly_consumption_A = load_json(f"{path_fcfs}monthly_with_solar_cost.json")
monthly_consumption_B = load_json(f"{path_fcfs}monthly_no_solar_cost.json")

# --------------------------
# 🔹 Chuẩn bị dữ liệu cho biểu đồ
# --------------------------
# Lấy danh sách tháng đã sắp xếp từ tất cả các file (chỉ lấy keys từ các dict không rỗng)
all_keys = set()
for data in [monthly_consumption_with, monthly_consumption_without, monthly_consumption_A, monthly_consumption_B]:
    if data: # Chỉ thêm keys nếu dict không rỗng
        all_keys.update(data.keys())

# Cần đảm bảo thứ tự tháng là hợp lý (ví dụ: Jan, Feb, Mar...)
# Nếu keys là "1", "2", ..., "12", sắp xếp theo số
try:
    months_numeric = sorted([int(m) for m in all_keys])
    months = [str(m) for m in months_numeric]
except ValueError:
    # Nếu keys không phải số (ví dụ: "Jan", "Feb"), sắp xếp theo alphabet hoặc logic khác nếu cần
    months = sorted(list(all_keys))
    # Ví dụ: Nếu bạn muốn thứ tự cụ thể:
    # month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    # months = [m for m in month_order if m in all_keys]


print(f"Tháng được sử dụng cho biểu đồ: {months}")

# Chuẩn bị dữ liệu cho biểu đồ, dùng get(month, 0) để xử lý tháng bị thiếu
consumption_with_values = [monthly_consumption_with.get(month, 0) for month in months]
consumption_without_values = [monthly_consumption_without.get(month, 0) for month in months]
consumption_A_values = [monthly_consumption_A.get(month, 0) for month in months]
consumption_B_values = [monthly_consumption_B.get(month, 0) for month in months]

# --------------------------
# 🔹 Vẽ biểu đồ cột ghép
# --------------------------
x = np.arange(len(months))  # Vị trí số cho các nhóm tháng trên trục x
bar_width = 0.2  # Độ rộng của mỗi cột trong nhóm

fig, ax = plt.subplots(figsize=(15, 7)) # Tăng kích thước để dễ nhìn hơn

# Vẽ các cột, dịch chuyển vị trí x cho mỗi bộ dữ liệu
rects1 = ax.bar(x - 1.5 * bar_width, consumption_with_values, bar_width, label="With Solar (Optimal)", color="green")
rects2 = ax.bar(x - 0.5 * bar_width, consumption_without_values, bar_width, label="Without Solar (Optimal)", color="red")
rects3 = ax.bar(x + 0.5 * bar_width, consumption_A_values, bar_width, label="With Solar (FCFS)", color="purple")
rects4 = ax.bar(x + 1.5 * bar_width, consumption_B_values, bar_width, label="Without Solar (FCFS)", color="black")

# --- Cấu hình biểu đồ ---
ax.set_xlabel("Months")
ax.set_ylabel("Electricity Cost (EUR)")
ax.set_title("Monthly Electricity Cost Comparison (Optimal vs FCFS, With/Without Solar)") # Cập nhật tiêu đề
ax.set_xticks(x)  # Đặt vị trí các nhãn tháng ở giữa các nhóm cột
ax.set_xticklabels(months, rotation=45, ha="right") # Đặt nhãn tháng và xoay
ax.legend()
ax.grid(True, axis='y', linestyle='--', alpha=0.7) # Chỉ hiển thị lưới ngang
ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))

plt.tight_layout() # Tự động điều chỉnh bố cục cho vừa vặn
plt.show()