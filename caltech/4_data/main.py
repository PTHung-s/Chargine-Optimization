import json
import matplotlib.pyplot as plt
from matplotlib import ticker

# --------------------------
# 🔹 Hàm đọc dữ liệu từ JSON
# --------------------------
def load_json(filename):
    """Đọc file JSON và trả về dữ liệu dưới dạng dict (key: month, value: giá trị)"""
    with open(filename, "r") as f:
        return json.load(f)

# --------------------------
# 🔹 Đọc dữ liệu từ các file JSON đã lưu trước đó
# --------------------------
# Giả sử các file JSON của bạn có tên như sau:
# - "../2data_optimized/with_solar_S_plus.json": Chi phí điện khi có năng lượng mặt trời (Optimal)
# - "../2data_optimized/no_solar_S_plus.json": Chi phí điện khi không có năng lượng mặt trời (Optimal)
# - "../2data_fcfs/with_solar_S_plus.json": Chi phí điện khi có năng lượng mặt trời (FCFS)
# - "../2data_fcfs/no_solar_S_plus.json": Chi phí điện khi không có năng lượng mặt trời (FCFS)

monthly_consumption_with = load_json("../2data_optimized/monthly_cost_with_solar.json")
monthly_consumption_without = load_json("../2data_optimized/monthly_cost_no_solar.json")
monthly_consumption_A = load_json("../2data_fcfs/monthly_with_solar_cost.json")
monthly_consumption_B = load_json("../2data_fcfs/monthly_no_solar_cost.json")

# --------------------------
# 🔹 Chuẩn bị dữ liệu cho biểu đồ
# --------------------------
# Lấy danh sách tháng đã sắp xếp từ tất cả các file
months = sorted(set(monthly_consumption_with.keys()) | set(monthly_consumption_without.keys()) | 
                set(monthly_consumption_A.keys()) | set(monthly_consumption_B.keys()))

# Chuẩn bị dữ liệu cho biểu đồ
consumption_with_values = [monthly_consumption_with.get(month, 0) for month in months]
consumption_without_values = [monthly_consumption_without.get(month, 0) for month in months]
consumption_A_values = [monthly_consumption_A.get(month, 0) for month in months]
consumption_B_values = [monthly_consumption_B.get(month, 0) for month in months]

# --------------------------
# 🔹 Vẽ biểu đồ tổng lượng điện tiêu thụ theo tháng
# --------------------------
plt.figure(figsize=(10, 5))
# plt.plot(months, consumption_without_values, label="Without Solar Energy (Optimal)", color="red", linestyle="--", marker="o")
plt.plot(months, consumption_with_values, label="With Solar Energy (Optimal)", color="green", linestyle="-", marker="s")
# plt.plot(months, consumption_A_values, label="With Solar Energy (FCFS)", color="purple", linestyle="-.", marker="d")
plt.plot(months, consumption_B_values, label="Without Solar Energy (FCFS)", color="black", linestyle=":", marker="x")
plt.xlabel("Months")
plt.ylabel("Electricity Cost (EUR)")  # Thay đổi nhãn vì dữ liệu là chi phí
plt.title("Electricity Cost by Months with Our Model in Caltech")  # Thay đổi tiêu đề
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
plt.tight_layout()
plt.show()