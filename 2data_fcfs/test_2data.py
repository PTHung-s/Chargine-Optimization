import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib import ticker
import os

# -------------------------------------------------
# 1) Đọc dữ liệu
# -------------------------------------------------
def read_json_files():
    with open("../Loc_data_price/price.json", "r") as f:
        price_data = json.load(f)
    with open("../Loc_solar/gop_solar.json", "r") as f:
        solar_data = json.load(f)
    with open("../Loc_data_EV_1/final_gop.json", "r") as f:
        final_gop_data = json.load(f)
    with open("../Loc_data_EV_1/max.json", "r") as f:
        max_data = json.load(f)
    return price_data, solar_data, final_gop_data, max_data

# -------------------------------------------------
# 2) Hàm chuyển đổi chuỗi time sang datetime
# -------------------------------------------------
def parse_time(time_str):
    # "Thu, 26 Apr 2018 00:02:16 GMT"
    return datetime.strptime(time_str.replace("GMT", "").strip(), "%a, %d %b %Y %H:%M:%S")

# -------------------------------------------------
# 3) Lấy các ngày từ final_gop_data trong khoảng
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
# 4) Tạo block overlapping (mỗi block chứa 2 ngày liên tiếp)
# -------------------------------------------------
def create_overlapping_blocks(sorted_dates, final_gop_data, mapping):
    blocks = []
    pending = []
    if len(sorted_dates) < 2:
        # Chỉ 1 ngày -> block 24h
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

    # Nếu pending còn, tạo block cho ngày cuối
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
        pending = new_pending
    return blocks

# -------------------------------------------------
# 5) Chuẩn bị dữ liệu cho block (tạo ma trận A, R, p_grid,...)
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

    # p_grid
    p_grid = []
    current = block_start
    while current < block_end:
        day_str = current.strftime("%Y-%m-%d")
        hour_index = current.hour
        if day_str in price_data:
            price = price_data[day_str][hour_index]
        else:
            price = 0
        p_grid.append(price)
        current += timedelta(hours=1)

    # R
    R_list = []
    current = block_start
    while current < block_end:
        solar_key = current.strftime("%Y%m%d")
        hour_index = current.hour
        if solar_key in solar_data:
            R_val = solar_data[solar_key][hour_index]["R(i)"]
        else:
            R_val = 0
        R_list.append(R_val)
        current += timedelta(hours=1)

    # Sắp xếp session
    sessions_sorted = sorted(sessions, key=lambda s: parse_time(s["connectionTime"]))

    # Xây dựng A[i][t]
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
            if eff_end > eff_start:
                fraction = (eff_end - eff_start).total_seconds()/3600.0
            else:
                fraction = 0
            availability.append(fraction)
        A.append(availability)
        L_req.append(session["kWhDelivered"])
        conn_times.append(conn)

    s = max_data["doubled_max_rate"]

    data = {
        "T": T,
        "N": len(sessions_sorted),
        "sessions_sorted": sessions_sorted,
        "A": A,                  # Ma trận [N][T]
        "L_req": L_req,         # Năng lượng (kWh) cần nạp
        "conn_times": conn_times,
        "p_grid": p_grid,
        "R": R_list,
        "s": s,
        "eta": 0.9,             # Hiệu suất sạc
        "C_grid": 300,          # Giới hạn lưới (kW) cho mỗi giờ
        "delta_t": 1
    }
    return data

# -------------------------------------------------
# 6) Hàm FCFS: Tính S_plus[t] và cost[t] mỗi giờ
# -------------------------------------------------
def fcfs_per_block(data, use_solar=True):
    """
    FCFS với tối đa 80 trạm.
    Nếu use_solar=False, ta ép R_list=0 để xem không có NLMT.
    Trả về:
      S_plus[t] (lượng điện mua từ lưới mỗi giờ)
      cost[t] (chi phí điện giờ t)
    """
    T = data["T"]
    N = data["N"]
    A = data["A"]            
    L_req = data["L_req"]    
    s = data["s"]            
    eta = data["eta"]        
    p_grid = data["p_grid"]  
    R_list = data["R"][:]    # copy
    C_grid = data["C_grid"]  
    conn_times = data["conn_times"]

    # Nếu không dùng solar, ép R=0
    if not use_solar:
        R_list = [0]*T

    needed = [(L_req[i] / eta) for i in range(N)]  # kWh cần (đã chia hiệu suất)
    X = [[0.0]*T for _ in range(N)]                # Ma trận sạc theo giờ
    STATION_LIMIT = 80

    for t in range(T):
        grid_leftover = C_grid
        station_count = 0

        # Tìm các phiên có mặt giờ t và còn cần
        present_sessions = []
        for i in range(N):
            if needed[i] > 1e-9 and A[i][t] > 0:
                present_sessions.append(i)

        # FCFS theo connectionTime
        present_sessions.sort(key=lambda i_: conn_times[i_])

        for i in present_sessions:
            if station_count < STATION_LIMIT and grid_leftover > 1e-9:
                deliverable = min(s*A[i][t], needed[i], grid_leftover)
                X[i][t] = deliverable
                needed[i] -= deliverable
                grid_leftover -= deliverable
                station_count += 1
            else:
                # Hết trạm hoặc hết lưới
                break

    # Tính S_plus[t] và cost[t] cho mỗi giờ
    S_plus = [0.0]*T
    cost = [0.0]*T

    for t in range(T):
        total_load_t = sum(X[i][t] for i in range(N))  # kWh sạc trong giờ t
        used_R_t = min(R_list[t], total_load_t)        # lấy từ NLMT trước
        grid_needed = total_load_t - used_R_t          # phần còn lại mua lưới
        S_plus[t] = grid_needed
        cost[t] = p_grid[t] * grid_needed

    return S_plus, cost

# -------------------------------------------------
# 7) Main: Chỉ chạy FCFS cho 2 kịch bản, xuất 4 file JSON
# -------------------------------------------------
def main():
    # Khoảng ngày
    start_date = datetime.strptime("25-04-2018", "%d-%m-%Y")
    end_date = datetime.strptime("07-06-2019", "%d-%m-%Y")
    
    price_data, solar_data, final_gop_data, max_data = read_json_files()

    # Lọc dữ liệu
    filtered_price = {}
    for k, v in price_data.items():
        try:
            dt = datetime.strptime(k, "%Y-%m-%d")
            if start_date <= dt <= end_date:
                filtered_price[k] = v
        except:
            pass
    price_data = filtered_price

    filtered_solar = {}
    for k, v in solar_data.items():
        try:
            dt = datetime.strptime(k, "%Y%m%d")
            if start_date <= dt <= end_date:
                filtered_solar[k] = v
        except:
            pass
    solar_data = filtered_solar

    # Tạo các block
    sorted_dates, mapping = parse_final_gop_dates(final_gop_data, start_date, end_date)
    if not sorted_dates:
        print("Không có dữ liệu trong khoảng ngày được chọn!")
        return

    blocks = create_overlapping_blocks(sorted_dates, final_gop_data, mapping)
    print(f"Tổng số block cần xử lý: {len(blocks)}")

    # Các dictionary lưu kết quả (mỗi ngày -> list 24 giá trị)
    with_solar_S_plus_results = {}
    with_solar_cost_results = {}
    no_solar_S_plus_results = {}
    no_solar_cost_results = {}

    # Xử lý từng block
    for idx, block in enumerate(blocks, 1):
        block_dates = block["dates"]
        # Chọn ngày đại diện (ngày đầu block)
        rep_day = block_dates[0]
        day_str = rep_day.strftime("%Y-%m-%d")

        print("\n========================================")
        print(f"Đang xử lý block {idx}: {', '.join(d.strftime('%Y-%m-%d') for d in block_dates)}")

        data_block = get_block_data_from_block(block, price_data, solar_data, max_data)
        T = data_block["T"]
        N = data_block["N"]
        if N == 0:
            print(" - Block không có phiên sạc, bỏ qua.")
            # Gán 24 giá trị 0
            with_solar_S_plus_results[day_str] = [0]*24
            with_solar_cost_results[day_str] = [0]*24
            no_solar_S_plus_results[day_str] = [0]*24
            no_solar_cost_results[day_str] = [0]*24
            continue

        print(f" - T (số giờ) = {T}, N (số phiên) = {N}")

        # FCFS - có solar
        splus_with, cost_with = fcfs_per_block(data_block, use_solar=True)

        # FCFS - không solar
        splus_no, cost_no = fcfs_per_block(data_block, use_solar=False)

        # Ghi lại 24 giờ đầu cho ngày đại diện
        # (nếu block có 48h, 24h đầu thuộc ngày rep_day)
        with_solar_S_plus_results[day_str] = splus_with[:24]
        with_solar_cost_results[day_str] = cost_with[:24]
        no_solar_S_plus_results[day_str] = splus_no[:24]
        no_solar_cost_results[day_str] = cost_no[:24]

    # Lưu ra file JSON
    with open("with_solar_S_plus.json", "w", encoding="utf-8") as f:
        json.dump(with_solar_S_plus_results, f, ensure_ascii=False, indent=4)
    with open("no_solar_S_plus.json", "w", encoding="utf-8") as f:
        json.dump(no_solar_S_plus_results, f, ensure_ascii=False, indent=4)
    with open("with_solar_cost.json", "w", encoding="utf-8") as f:
        json.dump(with_solar_cost_results, f, ensure_ascii=False, indent=4)
    with open("no_solar_cost.json", "w", encoding="utf-8") as f:
        json.dump(no_solar_cost_results, f, ensure_ascii=False, indent=4)

    print("\nĐã lưu kết quả FCFS cho 2 kịch bản (có / không có NLMT) vào 4 file JSON!")

if __name__ == "__main__":
    main()
