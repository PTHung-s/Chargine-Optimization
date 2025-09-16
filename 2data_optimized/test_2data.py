import json
import pulp
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib import ticker  # Import ticker để định dạng trục y
import os

# ---------------------------
# Hàm đọc dữ liệu từ file với đường dẫn mới
# ---------------------------
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

# ---------------------------
# Hàm chuyển đổi chuỗi thời gian sang datetime
# ---------------------------
def parse_time(time_str):
    # Định dạng: "Thu, 26 Apr 2018 00:02:16 GMT"
    return datetime.strptime(time_str.replace("GMT", "").strip(), "%a, %d %b %Y %H:%M:%S")

# ---------------------------
# Lấy các ngày từ final_gop_data nằm trong khoảng ngày cần xử lý
# ---------------------------
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

# ---------------------------
# Tạo các block theo kiểu overlapping: mỗi block gồm 2 ngày liên tiếp.
# Block thứ i sẽ gồm ngày i-1 và ngày i (ví dụ: block1: [Ngày1, Ngày2], block2: [Ngày2, Ngày3], ...)
# Nếu trong block có phiên nào của ngày hôm nay có disconnectTime vượt qua 00:00 của ngày sau,
# thì phiên đó sẽ không xử lý trong block hiện tại mà được chuyển sang pending cho block sau.
# ---------------------------
def create_overlapping_blocks(sorted_dates, final_gop_data, mapping):
    blocks = []
    pending = []  # Các phiên kéo dài qua block, từ block trước
    # Nếu chỉ có 1 ngày, tạo block 24h
    if len(sorted_dates) < 2:
        block = {"dates": [sorted_dates[0]], "sessions": final_gop_data[mapping[sorted_dates[0]]]}
        blocks.append(block)
        return blocks

    # Duyệt từ ngày thứ 2 đến ngày cuối cùng để tạo block gồm (ngày trước, ngày hiện tại)
    for i in range(1, len(sorted_dates)):
        day_prev = sorted_dates[i-1]
        day_curr = sorted_dates[i]
        block_dates = [day_prev, day_curr]
        block_start = datetime.combine(day_prev, datetime.min.time())
        # Block kết thúc vào 00:00 của ngày sau day_curr (tức 24 giờ của day_curr)
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
        pending = new_pending  # Cập nhật pending cho block kế tiếp
    # Nếu còn pending sau khi duyệt hết, tạo block cuối cùng (24h) với ngày cuối cùng
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

# ---------------------------
# Xây dựng dữ liệu đầu vào cho LP theo block từ overlapping block đã tạo
# ---------------------------
def get_block_data_from_block(block, price_data, solar_data, max_data):
    block_dates = block["dates"]
    sessions = block["sessions"]
    # Xác định block_start là 00:00 của ngày đầu block,
    # block_end là 00:00 của ngày sau ngày cuối block (nếu có 2 ngày => 48 tiếng)
    block_start = datetime.combine(block_dates[0], datetime.min.time())
    if len(block_dates) > 1:
        block_end = datetime.combine(block_dates[-1], datetime.min.time()) + timedelta(days=1)
    else:
        block_end = block_start + timedelta(days=1)
    T = int((block_end - block_start).total_seconds() / 3600)  # Số giờ trong block

    # Tạo danh sách giá điện theo giờ (p_grid)
    p_grid = []
    current = block_start
    while current < block_end:
        day_str = current.strftime("%Y-%m-%d")
        if day_str in price_data:
            hour_index = current.hour
            price = price_data[day_str][hour_index]
        else:
            price = 0
        p_grid.append(price)
        current += timedelta(hours=1)

    # Tạo danh sách năng lượng tái tạo theo giờ (R)
    R = []
    current = block_start
    while current < block_end:
        solar_key = current.strftime("%Y%m%d")
        if solar_key in solar_data:
            hour_index = current.hour
            R_val = solar_data[solar_key][hour_index]["R(i)"]
        else:
            R_val = 0
        R.append(R_val)
        current += timedelta(hours=1)

    # Xây dựng ma trận A (theo phần trăm thời gian có mặt của EV) và danh sách L_req (năng lượng cần nhận)
    A_matrix = []
    L_req = []
    for session in sessions:
        conn = parse_time(session["connectionTime"])
        disc = parse_time(session["disconnectTime"])
        # Giới hạn phiên trong block hiện tại
        session_start = max(conn, block_start)
        session_end = min(disc, block_end)
        availability = []
        for t in range(T):
            slot_start = block_start + timedelta(hours=t)
            slot_end = slot_start + timedelta(hours=1)
            # Xác định khoảng thời gian giao nhau giữa khung giờ [slot_start, slot_end]
            # và phiên [session_start, session_end]
            effective_start = max(slot_start, session_start)
            effective_end = min(slot_end, session_end)
            if effective_end > effective_start:
                fraction = (effective_end - effective_start).total_seconds() / 3600.0
                fraction = min(fraction, 1)  # Đảm bảo không vượt quá 1
            else:
                fraction = 0
            availability.append(fraction)
        A_matrix.append(availability)
        L_req.append(session["kWhDelivered"])
    
    # Lấy công suất sạc tối đa từ max.json (sử dụng doubled_max_rate)
    s = max_data["doubled_max_rate"]

    data = {
        "T": T,
        "N": len(sessions),
        "delta_t": 1,
        "eta": 0.9,
        "p_grid": p_grid,
        "R": R,
        "s": s,
        "L_req": L_req,
        "C_grid": 300,
        "A": A_matrix
    }
    return data

# ---------------------------
# Xây dựng mô hình LP theo dữ liệu truyền vào
# ---------------------------
def build_model(data):
    T = data["T"]
    N = data["N"]
    delta_t = data["delta_t"]
    eta = data["eta"]
    p_grid = data["p_grid"]
    R_list = data["R"]
    s = data["s"]
    L_req = data["L_req"]
    C_grid = data["C_grid"]
    A = data["A"]

    problem = pulp.LpProblem("EV_Charging_Optimization", pulp.LpMinimize)

    # Biến quyết định: Y_{i,t} - công suất sạc cho phiên i tại giờ t
    Y = pulp.LpVariable.dicts("Y", ((i, t) for i in range(N) for t in range(T)),
                                lowBound=0, cat=pulp.LpContinuous)
    # Biến S_plus[t]: điện mua từ lưới tại giờ t
    S_plus = pulp.LpVariable.dicts("S_plus", (t for t in range(T)),
                                   lowBound=0, cat=pulp.LpContinuous)
    # Biến R_used[t]: năng lượng tái tạo sử dụng tại giờ t
    R_used = pulp.LpVariable.dicts("R_used", (t for t in range(T)),
                                   lowBound=0, cat=pulp.LpContinuous)

    # Hàm mục tiêu: tối thiểu hóa chi phí điện lưới
    problem += pulp.lpSum([p_grid[t] * S_plus[t] * delta_t for t in range(T)]), "Minimize_Cost"

    # Ràng buộc năng lượng tối thiểu cho mỗi phiên EV
    for i in range(N):
        T_i = [t for t in range(T) if A[i][t] > 0]  # chỉ xét những giờ EV có mặt
        problem += eta * pulp.lpSum([Y[(i, t)] * delta_t for t in T_i]) >= L_req[i], f"EnergyReq_EV_{i}"

    # Ràng buộc công suất sạc tối đa và chỉ sạc khi EV có mặt
    for i in range(N):
        for t in range(T):
            problem += Y[(i, t)] <= s, f"MaxPower_EV_{i}_t_{t}"
            problem += Y[(i, t)] <= s * A[i][t], f"Presence_EV_{i}_t_{t}"

    # Ràng buộc điện lưới và năng lượng tái tạo
    for t in range(T):
        total_load = pulp.lpSum([Y[(i, t)] for i in range(N)])
        problem += total_load - R_used[t] <= C_grid, f"GridLimit_t_{t}"
        problem += S_plus[t] >= total_load - R_used[t], f"SplusPositivity_t_{t}"
        problem += R_used[t] <= R_list[t], f"RenewableLimit_t_{t}"

    return problem, Y, S_plus, R_used

# ---------------------------
# Giải bài toán LP
# ---------------------------
def solve_model(problem):
    problem.solve(pulp.PULP_CBC_CMD(msg=0))
    status = pulp.LpStatus[problem.status]
    obj_val = pulp.value(problem.objective)
    return status, obj_val

# ---------------------------
# Main: xử lý dữ liệu theo block, tính toán tối ưu cho cả 2 trường hợp (có và không có năng lượng mặt trời)
# và lưu kết quả ra 4 file JSON như yêu cầu.
# ---------------------------
def main():
    # Đặt khoảng ngày cần xử lý: từ 25/04/2018 đến 07/06/2019
    start_date = datetime.strptime("25-04-2018", "%d-%m-%Y")
    end_date = datetime.strptime("07-06-2019", "%d-%m-%Y")
    
    price_data, solar_data, final_gop_data, max_data = read_json_files()
    
    # Lọc dữ liệu price_data theo khoảng ngày (key dạng "YYYY-MM-DD")
    filtered_price = {}
    for k, v in price_data.items():
        try:
            dt = datetime.strptime(k, "%Y-%m-%d")
            if start_date <= dt <= end_date:
                filtered_price[k] = v
        except Exception:
            continue
    price_data = filtered_price

    # Lọc dữ liệu solar_data (key dạng "YYYYMMDD")
    filtered_solar = {}
    for k, v in solar_data.items():
        try:
            dt = datetime.strptime(k, "%Y%m%d")
            if start_date <= dt <= end_date:
                filtered_solar[k] = v
        except Exception:
            continue
    solar_data = filtered_solar

    # Lấy các ngày từ final_gop_data nằm trong khoảng xử lý
    sorted_dates, mapping = parse_final_gop_dates(final_gop_data, start_date, end_date)
    if not sorted_dates:
        print("Không có dữ liệu trong khoảng ngày được chọn!")
        return

    # Tạo các block overlapping: mỗi block gồm (ngày trước, ngày hôm nay)
    blocks = create_overlapping_blocks(sorted_dates, final_gop_data, mapping)
    print(f"Tổng số block cần xử lý: {len(blocks)}")
    
    # (Có thể vẫn ghi log kết quả vào file results.jsonl nếu cần)
    jsonl_file = "results.jsonl"
    f_out = open(jsonl_file, "a", encoding="utf-8")
    
    monthly_results = {}

    # Khai báo các dictionary lưu kết quả theo ngày cho 2 trường hợp:
    solar_case_S_plus_results = {}
    solar_case_cost_results = {}
    no_solar_case_S_plus_results = {}
    no_solar_case_cost_results = {}

    # Xử lý từng block
    for idx, block in enumerate(blocks, 1):
        block_dates = block["dates"]
        # Chọn ngày đại diện cho block: sử dụng ngày đầu tiên của block
        rep_day = block_dates[0]
        day_str = rep_day.strftime("%Y-%m-%d")
        
        if len(block_dates) == 2:
            date_range_str = f"{block_dates[0].strftime('%Y-%m-%d')} to {block_dates[1].strftime('%Y-%m-%d')}"
            hours_to_extract = 24  # chỉ lấy 24 giờ của ngày đầu
        else:
            date_range_str = f"{block_dates[0].strftime('%Y-%m-%d')}"
            hours_to_extract = 24  # giả sử block 1 ngày luôn có 24 giờ

        print("\n================================================")
        print(f"Đang xử lý block {idx}: {date_range_str}")
        
        # Xây dựng dữ liệu đầu vào cho block hiện tại
        data_with_solar = get_block_data_from_block(block, price_data, solar_data, max_data)
        T = data_with_solar["T"]
        print(f" - Số giờ trong block (T): {T}")
        print(f" - Số phiên EV (N): {data_with_solar['N']}")
        if data_with_solar["p_grid"]:
            print(f" - Giá điện mẫu (p_grid): {data_with_solar['p_grid'][:5]} ...")
        if data_with_solar["R"]:
            print(f" - Năng lượng tái tạo mẫu (R): {data_with_solar['R'][:5]} ...")
        print(f" - Năng lượng yêu cầu (L_req): {data_with_solar['L_req']}")
        
        # Nếu không có phiên sạc nào trong block thì gán giá trị 0 cho 24 giờ
        if data_with_solar["N"] == 0:
            print("Không có phiên sạc nào trong block này, bỏ qua tối ưu!")
            solar_S_plus = [0]*hours_to_extract
            solar_cost = [0]*hours_to_extract
            no_solar_S_plus = [0]*hours_to_extract
            no_solar_cost = [0]*hours_to_extract

            result_obj = {
                "date_range": date_range_str,
                "objective_value": None,
                "status": "No session"
            }
            f_out.write(json.dumps(result_obj) + "\n")
            f_out.flush()
        else:
            # ----- Trường hợp có năng lượng mặt trời -----
            model_solar, Y_solar, S_plus_solar, R_used_solar = build_model(data_with_solar)
            status_solar, obj_val_solar = solve_model(model_solar)
            solar_S_plus_full = [S_plus_solar[t].varValue for t in range(T)]
            p_grid = data_with_solar["p_grid"]
            # Lấy 24 giờ ứng với ngày đại diện (ngày đầu block)
            solar_S_plus = solar_S_plus_full[:hours_to_extract]
            solar_cost = [p_grid[t] * solar_S_plus_full[t] for t in range(hours_to_extract)]
            print(f" - Kết quả solver (có mặt trời): {status_solar} với objective value = {obj_val_solar}")
            
            # ----- Trường hợp không có năng lượng mặt trời -----
            # Tạo bản sao dữ liệu và gán R = 0 cho mọi giờ
            data_no_solar = data_with_solar.copy()
            data_no_solar["R"] = [0]*T
            model_no, Y_no, S_plus_no, R_used_no = build_model(data_no_solar)
            status_no, obj_val_no = solve_model(model_no)
            no_solar_S_plus_full = [S_plus_no[t].varValue for t in range(T)]
            p_grid_no = data_no_solar["p_grid"]  # giống nhau
            no_solar_S_plus = no_solar_S_plus_full[:hours_to_extract]
            no_solar_cost = [p_grid_no[t] * no_solar_S_plus_full[t] for t in range(hours_to_extract)]
            print(f" - Kết quả solver (không có mặt trời): {status_no} với objective value = {obj_val_no}")

            result_obj = {
                "date_range": date_range_str,
                "objective_value_with_solar": obj_val_solar,
                "status_with_solar": status_solar,
                "objective_value_no_solar": obj_val_no,
                "status_no_solar": status_no
            }
            f_out.write(json.dumps(result_obj) + "\n")
            f_out.flush()

        # Ghi kết quả của ngày đại diện (chỉ ghi 24 giờ đầu của block)
        solar_case_S_plus_results[day_str] = solar_S_plus
        solar_case_cost_results[day_str] = solar_cost
        no_solar_case_S_plus_results[day_str] = no_solar_S_plus
        no_solar_case_cost_results[day_str] = no_solar_cost

        # Tổng hợp theo tháng (dựa trên ngày đại diện của block)
        month_str = rep_day.strftime("%Y-%m")
        monthly_results[month_str] = monthly_results.get(month_str, 0) + (obj_val_solar if obj_val_solar is not None else 0)
    
    f_out.close()

    print("\nTổng hợp chi phí theo tháng (có mặt trời):")
    for month, cost in sorted(monthly_results.items()):
        print(f" - {month}: {cost}")

    # Vẽ biểu đồ thống kê theo tháng (có mặt trời)
    months = sorted(monthly_results.keys())
    costs = [monthly_results[m] for m in months]
    
    plt.figure()
    plt.bar(months, costs)
    plt.xlabel("Tháng")
    plt.ylabel("Tổng Chi Phí Điện Lưới Tối Thiểu")
    plt.title("Thống kê chi phí điện lưới theo tháng (có mặt trời)")
    plt.xticks(rotation=45)
    plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
    plt.tight_layout()
    plt.show()

    # --------- Lưu kết quả ra các file JSON ---------
    with open("with_solar_S_plus.json", "w", encoding="utf-8") as f:
        json.dump(solar_case_S_plus_results, f, ensure_ascii=False, indent=4)
    with open("no_solar_S_plus.json", "w", encoding="utf-8") as f:
        json.dump(no_solar_case_S_plus_results, f, ensure_ascii=False, indent=4)
    with open("with_solar_cost.json", "w", encoding="utf-8") as f:
        json.dump(solar_case_cost_results, f, ensure_ascii=False, indent=4)
    with open("no_solar_cost.json", "w", encoding="utf-8") as f:
        json.dump(no_solar_case_cost_results, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()
