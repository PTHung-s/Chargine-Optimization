import json
import matplotlib.pyplot as plt
from matplotlib import ticker

# --------------------------
# 🔹 Hàm đọc dữ liệu từ JSON
# --------------------------
def load_json(filename):
    """Đọc file JSON và trả về dữ liệu dưới dạng dict (key: date, value: danh sách 24 giá trị)"""
    with open(filename, "r") as f:
        return json.load(f)

# --------------------------
# 🔹 Hàm tính tổng (hoặc trung bình) theo tháng
# --------------------------
def aggregate_monthly(data):
    """
    Giả sử data là dict với key là ngày ("YYYY-MM-DD") và value là list 24 giá trị.
    Tính tổng hàng ngày (sum của 24 giờ) rồi nhóm theo tháng.
    """
    monthly = {}
    for date_str, hourly_values in data.items():
        daily_total = sum(hourly_values)
        # Lấy phần "YYYY-MM" của ngày
        month = date_str[:7]
        monthly[month] = monthly.get(month, 0) + daily_total
    return monthly

# --------------------------
# 🔹 Đọc dữ liệu từ các file JSON đã lưu trước đó
# --------------------------
consumption_with_renewable = load_json("with_solar_S_plus.json")
consumption_without_renewable = load_json("no_solar_S_plus.json")
cost_with_renewable = load_json("with_solar_cost.json")
cost_without_renewable = load_json("no_solar_cost.json")

# --------------------------
# 🔹 Tính tổng theo tháng
# --------------------------
monthly_consumption_with = aggregate_monthly(consumption_with_renewable)
monthly_consumption_without = aggregate_monthly(consumption_without_renewable)
monthly_cost_with = aggregate_monthly(cost_with_renewable)
monthly_cost_without = aggregate_monthly(cost_without_renewable)

# Lấy danh sách tháng đã sắp xếp (ví dụ: "2018-05", "2018-06", …)
months = sorted(monthly_consumption_with.keys())

# Chuẩn bị dữ liệu cho biểu đồ
consumption_with_values = [monthly_consumption_with[month] for month in months]
consumption_without_values = [monthly_consumption_without.get(month, 0) for month in months]
cost_with_values = [monthly_cost_with[month] for month in months]
cost_without_values = [monthly_cost_without.get(month, 0) for month in months]

# --------------------------
# 🔹 Vẽ biểu đồ tổng lượng điện tiêu thụ theo tháng
# --------------------------
plt.figure(figsize=(10, 5))
plt.plot(months, consumption_without_values, label="Without Solar energy", color="red", linestyle="--", marker="o")
plt.plot(months, consumption_with_values, label="With Solar energy", color="green", linestyle="-", marker="s")
plt.xlabel("Months")
plt.ylabel("Total electricity consumption (kWh)")
plt.title("Electricity consumption by months with Our Model in Caltech")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
plt.tight_layout()
plt.show()

# --------------------------
# 🔹 Vẽ biểu đồ tổng chi phí điện theo tháng
# --------------------------
plt.figure(figsize=(10, 5))
plt.plot(months, cost_without_values, label="Without Solar energy", color="orange", linestyle="--", marker="o")
plt.plot(months, cost_with_values, label="With Solar energy", color="blue", linestyle="-", marker="s")
plt.xlabel("Months")
plt.ylabel("Total Cost (EURO)")
plt.title("The cost of electricity over the months with Our Model in Caltech")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
plt.tight_layout()
plt.show()
