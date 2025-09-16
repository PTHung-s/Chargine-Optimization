import json
from datetime import datetime, timedelta

def parse_time(time_str):
    """
    Chuyển đổi chuỗi thời gian dạng "Thu, 26 Apr 2018 00:02:16 GMT" thành đối tượng datetime.
    """
    return datetime.strptime(time_str.replace("GMT", "").strip(), "%a, %d %b %Y %H:%M:%S")

def get_free_minutes_for_day(sessions, day):
    """
    Với một ngày cụ thể (day là datetime, đại diện cho 00:00 của ngày đó),
    trả về một danh sách 1440 phần tử (cho 1440 phút) với True nếu phút đó không có phiên sạc nào,
    và False nếu có phiên sạc đang hoạt động.
    """
    free = [True] * 1440
    start_of_day = datetime.combine(day, datetime.min.time())
    end_of_day = start_of_day + timedelta(days=1)
    
    for session in sessions:
        conn = parse_time(session["connectionTime"])
        disc = parse_time(session["disconnectTime"])
        # Giới hạn phiên sạc trong ngày: chỉ xét phần giao nhau với [start_of_day, end_of_day)
        session_start = max(conn, start_of_day)
        session_end = min(disc, end_of_day)
        if session_start < session_end:
            start_min = int((session_start - start_of_day).total_seconds() // 60)
            end_min = int((session_end - start_of_day).total_seconds() // 60)
            for i in range(start_min, end_min):
                free[i] = False
    return free

def find_common_free_intervals(final_gop_data, start_date, end_date):
    """
    Lọc các ngày từ final_gop_data nằm trong khoảng [start_date, end_date].
    Với mỗi ngày, tính mảng free (1440 phút). Sau đó, lấy giao của tất cả các mảng free,
    và tìm các khoảng liên tục các phút chung rảnh.
    """
    # Lọc các key của final_gop_data theo định dạng "26-Apr-2018" và nằm trong khoảng
    mapping = {}
    days = []
    for key in final_gop_data.keys():
        try:
            dt = datetime.strptime(key, "%d-%b-%Y")
            if start_date <= dt <= end_date:
                mapping[dt] = key
                days.append(dt)
        except Exception as e:
            continue
    days = sorted(days)
    
    if not days:
        print("Không có dữ liệu nào trong khoảng ngày yêu cầu!")
        return []
    
    # Khởi tạo common_free là danh sách 1440 phần tử True
    common_free = [True] * 1440
    for day in days:
        day_key = mapping[day]
        sessions = final_gop_data[day_key]
        free = get_free_minutes_for_day(sessions, day)
        for i in range(1440):
            common_free[i] = common_free[i] and free[i]
    
    # Tìm các khoảng liên tục trong common_free (theo phút)
    intervals = []
    i = 0
    while i < 1440:
        if common_free[i]:
            start = i
            while i < 1440 and common_free[i]:
                i += 1
            intervals.append((start, i))
        else:
            i += 1
    return intervals

def minutes_to_time_str(minutes):
    """
    Chuyển đổi số phút thành định dạng HH:MM.
    """
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours:02d}:{mins:02d}"

def main():
    start_date = datetime.strptime("25-04-2018", "%d-%m-%Y")
    end_date = datetime.strptime("07-06-2019", "%d-%m-%Y")
    
    with open("./Loc_data_EV_1/final_gop.json", "r") as f:
        final_gop_data = json.load(f)
    
    common_intervals = find_common_free_intervals(final_gop_data, start_date, end_date)
    
    print("Các khoảng trống thời gian chung (theo HH:MM) trên các ngày:")
    if common_intervals:
        for (start_min, end_min) in common_intervals:
            duration = end_min - start_min
            print(f"{minutes_to_time_str(start_min)} đến {minutes_to_time_str(end_min)} (khoảng {duration} phút)")
    else:
        print("Không tìm thấy khoảng trống chung nào!")
    
if __name__ == "__main__":
    main()

