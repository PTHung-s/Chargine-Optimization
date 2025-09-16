import json
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib import ticker

def read_jsonl(filename, key_name):
    """Đọc file JSONL và tổng hợp dữ liệu theo tháng."""
    monthly_data = defaultdict(float)
    
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                date_range = entry["date_range"]  # "YYYY-MM-DD to YYYY-MM-DD" hoặc "YYYY-MM-DD"
                value = entry.get(key_name, None)
                if value is None:
                    continue
                first_date = date_range.split(" to ")[0]
                month_str = first_date[:7]
                monthly_data[month_str] += value
            except json.JSONDecodeError:
                print("Lỗi đọc dòng:", line.strip())
    return monthly_data

# Đọc dữ liệu từ JSONL
optimized_data = read_jsonl("results.jsonl", "objective_value")
fcfs_data = read_jsonl("FCFS.jsonl", "fcfs_cost")

# Danh sách tất cả các tháng
all_months = sorted(set(optimized_data.keys()).union(set(fcfs_data.keys())))

# Dữ liệu theo tháng
optimized_costs = [optimized_data.get(month, 0) for month in all_months]
fcfs_costs = [fcfs_data.get(month, 0) for month in all_months]

# Tạo dữ liệu cho biểu đồ
max_costs = [max(opt, fcfs) for opt, fcfs in zip(optimized_costs, fcfs_costs)]
lower_values = [min(opt, fcfs) for opt, fcfs in zip(optimized_costs, fcfs_costs)]
upper_values = [abs(opt - fcfs) for opt, fcfs in zip(optimized_costs, fcfs_costs)]

# Vẽ biểu đồ cột
plt.figure(figsize=(10, 6))
plt.bar(all_months, lower_values, color="green", label="With ACN")
plt.bar(all_months, upper_values, bottom=lower_values, color="orange", label="Without ACN")

plt.xlabel("Months")
plt.ylabel("Total Cost (EURO)")
plt.title("Electricity Costs: With vs. Without ACN by Month")
plt.xticks(rotation=45)
plt.legend()
plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
plt.tight_layout()
plt.show()
