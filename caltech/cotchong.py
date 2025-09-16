import json
import os
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
from matplotlib import ticker

def read_jsonl(filename, key_name):
    """Đọc file JSONL và tổng hợp dữ liệu theo tháng (YYYY-MM)."""
    monthly_data = defaultdict(float)
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                date_range = entry.get("date_range", "")
                value = entry.get(key_name, None)
                if value is None:
                    continue
                # "YYYY-MM-DD to YYYY-MM-DD" hoặc "YYYY-MM-DD"
                first_date = date_range.split(" to ")[0]
                month_str = first_date[:7]
                monthly_data[month_str] += float(value)
            except json.JSONDecodeError:
                print("Lỗi đọc dòng:", line.strip())
            except Exception as e:
                print("Lỗi xử lý dòng:", e)
    return dict(monthly_data)

# Khai báo các series cần vẽ: (nhãn, file, key)
series_specs = [
    ("LP (Optimized)", "results.jsonl", "objective_value"),
    ("PV-first", "PVFIRST.jsonl", "pvfirst_cost"),
    ("DRL (DQN)", "DRL.jsonl", "drl_cost"),
]

# Chỉ giữ những series có file tồn tại
series_data = []
for label, fname, key in series_specs:
    if os.path.exists(fname):
        data = read_jsonl(fname, key)
        if data:
            series_data.append((label, data))
    else:
        print(f"⚠️  Không tìm thấy file: {fname} (bỏ qua)")

if not series_data:
    raise RuntimeError("Không có dữ liệu nào để vẽ. Hãy kiểm tra các file JSONL đầu vào.")

# Tập hợp tất cả các tháng
all_months = sorted(set().union(*[set(d.keys()) for _, d in series_data]))
index = np.arange(len(all_months))

# Tạo mảng chi phí theo tháng cho từng series
values_per_series = []
for label, data in series_data:
    values = [data.get(m, 0.0) for m in all_months]
    values_per_series.append((label, values))

# Vẽ biểu đồ cột nhóm động theo số series
num_series = len(values_per_series)
bar_width = 0.8 / num_series  # dồn vừa trong 80% bề rộng
offsets = (np.arange(num_series) - (num_series - 1) / 2.0) * bar_width

plt.figure(figsize=(12, 6))

# Màu sắc tùy chọn (sẽ lặp nếu thiếu)
palette = ["#2ca02c", "#ff7f0e", "#1f77b4", "#d62728", "#9467bd", "#8c564b"]

for i, (label, vals) in enumerate(values_per_series):
    plt.bar(index + offsets[i], vals, bar_width, label=label, color=palette[i % len(palette)])

# Định dạng
plt.xlabel("Tháng")
plt.ylabel("Tổng chi phí (EUR)")
plt.title("So sánh chi phí điện theo tháng: LP vs PV-first vs DRL")
plt.xticks(index, all_months, rotation=45)
plt.legend()
plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
plt.tight_layout()
plt.show()
