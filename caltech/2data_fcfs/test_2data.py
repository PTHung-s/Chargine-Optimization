import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib import ticker
import os
import copy
import numpy as np

# -------------------------------------------------
# 1) Đọc dữ liệu từ các file JSON
# -------------------------------------------------
def read_json_files():
    with open("../../Loc_data_price/price.json", "r") as f:
        price_data = json.load(f)
    with open("../../Loc_solar/gop_solar.json", "r") as f:
        solar_data = json.load(f)
    with open("../final_gop.json", "r") as f:
        final_gop_data = json.load(f)
    with open("../max.json", "r") as f:
        max_data = json.load(f)
    return price_data, solar_data, final_gop_data, max_data

# -------------------------------------------------
# 2) Hàm chuyển đổi chuỗi thời gian sang datetime
# -------------------------------------------------
def parse_time(time_str):
    return datetime.strptime(time_str.replace("GMT", "").strip(), "%a, %d %b %Y %H:%M:%S")

# -------------------------------------------------
# 3) Lấy các ngày từ final_gop_data trong khoảng thời gian
# -------------------------------------------------
def parse_final_gop_dates(final_gop_data, start_date, end_date):
    mapping = {}
    for key in final_gop_data:
        try:
            dt = datetime.strptime(key, "%d-%b-%Y")
        except Exception:
            continue
        if start_date <= dt <= end_date:
            mapping[dt] = key
    sorted_dates = sorted(mapping.keys())
    return sorted_dates, mapping

# -------------------------------------------------
# 4) Tạo các block overlapping (mỗi block chứa 2 ngày liên tiếp)
# -------------------------------------------------
def create_overlapping_blocks(sorted_dates, final_gop_data, mapping):
    blocks = []
    pending = []
    if len(sorted_dates) < 2:
        block = {"dates": [sorted_dates[0]], 
                 "sessions": final_gop_data[mapping[sorted_dates[0]]]}
        blocks.append(block)
        return blocks

    for i in range(1, len(sorted_dates)):
        day_prev = sorted_dates[i-1]
        day_curr = sorted_dates[i]
        block_dates = [day_prev, day_curr]
        block_start = datetime.combine(day_prev, datetime.min.time())
        block_end = datetime.combine(day_curr, datetime.min.time()) + timedelta(days=1)

        sessions_prev = final_gop_data.get(mapping[day_prev], [])
        sessions_curr = final_gop_data.get(mapping[day_curr], [])
        block_sessions = pending + sessions_prev + sessions_curr

        new_block_sessions = []
        new_pending = []
        for session in block_sessions:
            dtime = parse_time(session["disconnectTime"])
            if dtime >= block_end:
                new_pending.append(session)
            else:
                new_block_sessions.append(session)
        block = {"dates": block_dates, "sessions": new_block_sessions}
        blocks.append(block)
        pending = new_pending

    if pending:
        last_date = sorted_dates[-1]
        block_start = datetime.combine(last_date, datetime.min.time())
        block_end = block_start + timedelta(days=1)
        final_sessions = []
        new_pending = []
        for session in pending:
            dtime = parse_time(session["disconnectTime"])
            if dtime >= block_end:
                new_pending.append(session)
            else:
                final_sessions.append(session)
        block = {"dates": [last_date], "sessions": final_sessions}
        blocks.append(block)
    return blocks

# -------------------------------------------------
# 5) Chuẩn bị dữ liệu cho block
# -------------------------------------------------
def get_block_data_from_block(block, price_data, solar_data, max_data):
    block_dates = block["dates"]
    sessions = block["sessions"]

    block_start = datetime.combine(block_dates[0], datetime.min.time())
    if len(block_dates) > 1:
        block_end = datetime.combine(block_dates[-1], datetime.min.time()) + timedelta(days=1)
    else:
        block_end = block_start + timedelta(days=1)

    T = int((block_end - block_start).total_seconds() / 3600)

    # Giá điện lưới (p_grid)
    p_grid = []
    current = block_start
    while current < block_end:
        day_str = current.strftime("%Y-%m-%d")
        hour_index = current.hour
        price = price_data.get(day_str, [0])[hour_index] / 1000  # VNĐ/kWh
        p_grid.append(price)
        current += timedelta(hours=1)

    # Năng lượng mặt trời (R)
    R_list = []
    current = block_start
    while current < block_end:
        solar_key = current.strftime("%Y%m%d")
        hour_index = current.hour
        R_val = solar_data.get(solar_key, [{"R(i)": 0}])[hour_index]["R(i)"]
        R_list.append(R_val)
        current += timedelta(hours=1)

    # Sắp xếp session theo thời gian kết nối
    sessions_sorted = sorted(sessions, key=lambda s: parse_time(s["connectionTime"]))

    # Xây dựng ma trận A[i][t] và các thông số khác
    A = []
    L_req = []
    conn_times = []
    for session in sessions_sorted:
        conn = parse_time(session["connectionTime"])
        disc = parse_time(session["disconnectTime"])
        session_start = max(conn, block_start)
        session_end = min(disc, block_end)
        availability = []
        for t in range(T):
            slot_start = block_start + timedelta(hours=t)
            slot_end = slot_start + timedelta(hours=1)
            eff_start = max(slot_start, session_start)
            eff_end = min(slot_end, session_end)
            fraction = (eff_end - eff_start).total_seconds() / 3600.0 if eff_end > eff_start else 0
            fraction = min(fraction, 1)
            availability.append(fraction)
        A.append(availability)
        L_req.append(session["kWhDelivered"])
        conn_times.append(conn)

    s = max_data["doubled_max_rate"]

    data = {
        "T": T,
        "N": len(sessions_sorted),
        "sessions_sorted": sessions_sorted,
        "A": A,
        "L_req": L_req,
        "conn_times": conn_times,
        "p_grid": p_grid,
        "R": R_list,
        "s": s,
        "eta": 0.9,
        "C_grid": 300,
        "delta_t": 1
    }
    return data

# -------------------------------------------------
# 6) Hàm FCFS: Tính S_plus[t] và cost[t] mỗi giờ
# -------------------------------------------------
def fcfs_per_block(data, use_solar=True):
    T = data["T"]
    N = data["N"]
    A = data["A"]
    L_req = data["L_req"]
    s = data["s"]
    eta = data["eta"]
    p_grid = data["p_grid"]
    R_list = data["R"][:]
    C_grid = data["C_grid"]
    conn_times = data["conn_times"]

    if not use_solar:
        R_list = [0] * T

    needed = [(L_req[i] / eta) for i in range(N)]
    X = [[0.0] * T for _ in range(N)]
    STATION_LIMIT = 80

    for t in range(T):
        grid_leftover = C_grid
        station_count = 0
        present_sessions = [i for i in range(N) if needed[i] > 1e-9 and A[i][t] > 0]
        present_sessions.sort(key=lambda i: conn_times[i])

        for i in present_sessions:
            if station_count < STATION_LIMIT and grid_leftover > 1e-9:
                deliverable = min(s * A[i][t], needed[i], grid_leftover)
                X[i][t] = deliverable
                needed[i] -= deliverable
                grid_leftover -= deliverable
                station_count += 1

    S_plus = [0.0] * T
    cost = [0.0] * T
    for t in range(T):
        total_load_t = sum(X[i][t] for i in range(N))
        used_R_t = min(R_list[t], total_load_t) if use_solar else 0
        grid_needed = total_load_t - used_R_t
        S_plus[t] = grid_needed
        cost[t] = p_grid[t] * grid_needed

    return S_plus, cost

# -------------------------------------------------
# 7) Main: Chạy FCFS cho 2 kịch bản và tổng hợp kết quả
# -------------------------------------------------
def main():
    start_date = datetime.strptime("25-04-2018", "%d-%m-%Y")
    end_date = datetime.strptime("31-12-2019", "%d-%m-%Y")
    
    price_data, solar_data, final_gop_data, max_data = read_json_files()

    # Lọc dữ liệu theo khoảng thời gian
    filtered_price = {k: v for k, v in price_data.items() 
                     if start_date <= datetime.strptime(k, "%Y-%m-%d") <= end_date}
    price_data = filtered_price

    filtered_solar = {k: v for k, v in solar_data.items() 
                     if start_date <= datetime.strptime(k, "%Y%m%d") <= end_date}
    solar_data = filtered_solar

    # Tạo các block
    sorted_dates, mapping = parse_final_gop_dates(final_gop_data, start_date, end_date)
    if not sorted_dates:
        print("Không có dữ liệu trong khoảng ngày được chọn!")
        return

    blocks = create_overlapping_blocks(sorted_dates, final_gop_data, mapping)
    print(f"Tổng số block cần xử lý: {len(blocks)}")

    # Tổng hợp kết quả theo tháng
    monthly_with_solar_cost = {}
    monthly_no_solar_cost = {}
    monthly_with_solar_splus = {}
    monthly_no_solar_splus = {}

    for idx, block in enumerate(blocks, 1):
        block_dates = block["dates"]
        rep_day = block_dates[0]
        month_str = rep_day.strftime("%Y-%m")
        print(f"\nĐang xử lý block {idx}: {', '.join(d.strftime('%Y-%m-%d') for d in block_dates)}")

        data_block = get_block_data_from_block(block, price_data, solar_data, max_data)
        if data_block["N"] == 0:
            print(" - Block không có phiên sạc, bỏ qua.")
            continue

        # Chạy FCFS cho 2 kịch bản
        splus_with, cost_with = fcfs_per_block(copy.deepcopy(data_block), use_solar=True)
        splus_no, cost_no = fcfs_per_block(copy.deepcopy(data_block), use_solar=False)

        # Tổng hợp kết quả cho block
        block_cost_with = sum(cost_with)
        block_cost_no = sum(cost_no)
        block_splus_with = sum(splus_with)
        block_splus_no = sum(splus_no)

        # Cộng dồn theo tháng
        monthly_with_solar_cost[month_str] = monthly_with_solar_cost.get(month_str, 0) + block_cost_with
        monthly_no_solar_cost[month_str] = monthly_no_solar_cost.get(month_str, 0) + block_cost_no
        monthly_with_solar_splus[month_str] = monthly_with_solar_splus.get(month_str, 0) + block_splus_with
        monthly_no_solar_splus[month_str] = monthly_no_solar_splus.get(month_str, 0) + block_splus_no

    # Lưu kết quả vào 4 file JSON
    with open("monthly_with_solar_cost.json", "w", encoding="utf-8") as f:
        json.dump(monthly_with_solar_cost, f, ensure_ascii=False, indent=4)
    with open("monthly_no_solar_cost.json", "w", encoding="utf-8") as f:
        json.dump(monthly_no_solar_cost, f, ensure_ascii=False, indent=4)
    with open("monthly_with_solar_splus.json", "w", encoding="utf-8") as f:
        json.dump(monthly_with_solar_splus, f, ensure_ascii=False, indent=4)
    with open("monthly_no_solar_splus.json", "w", encoding="utf-8") as f:
        json.dump(monthly_no_solar_splus, f, ensure_ascii=False, indent=4)

    print("\nĐã lưu kết quả vào 4 file JSON!")

    # Vẽ biểu đồ cột ghép
    months = sorted(monthly_with_solar_cost.keys())
    costs_with = [monthly_with_solar_cost[m] for m in months]
    costs_no = [monthly_no_solar_cost[m] for m in months]
    splus_with = [monthly_with_solar_splus[m] for m in months]
    splus_no = [monthly_no_solar_splus[m] for m in months]

    # Biểu đồ chi phí
    plt.figure(figsize=(12, 6))
    bar_width = 0.35
    index = np.arange(len(months))
    plt.bar(index, costs_with, bar_width, label='Có NLMT', color='green')
    plt.bar(index + bar_width, costs_no, bar_width, label='Không NLMT', color='orange')
    plt.xlabel("Tháng")
    plt.ylabel("Chi phí điện (VNĐ)")
    plt.title("So sánh chi phí điện lưới theo tháng")
    plt.xticks(index + bar_width / 2, months, rotation=45)
    plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Biểu đồ sản lượng điện
    plt.figure(figsize=(12, 6))
    plt.bar(index, splus_with, bar_width, label='Có NLMT', color='blue')
    plt.bar(index + bar_width, splus_no, bar_width, label='Không NLMT', color='red')
    plt.xlabel("Tháng")
    plt.ylabel("Sản lượng điện mua từ lưới (kWh)")
    plt.title("So sánh sản lượng điện mua từ lưới theo tháng")
    plt.xticks(index + bar_width / 2, months, rotation=45)
    plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()