import requests
import pandas as pd
import json
import os
import time

# ========= THÔNG SỐ =========
API_KEY = 'tcrEGnoAniIGc3RdKBGchDIHH8PqeX2GpIFQn7nT9b8'
SITE_ID = 'caltech'
START_PAGE_DEFAULT = 614    # Trang bắt đầu nếu chưa có file last_page_downloaded.txt
MAX_PAGES = 10000         # Giới hạn số trang (tránh vòng lặp vô tận)
TIMEOUT = 100                # Timeout request
SLEEP_SECONDS = 5           # Chờ 5s giữa mỗi lần retry
CSV_FILE = 'ev_sessions.csv'

# File JSON lines (mỗi object 1 dòng)
JSONL_FILE = 'ev_sessions.jsonl'

# File JSON (một mảng) - chỉ tạo cuối cùng sau khi tải xong
FINAL_JSON_FILE = 'ev_sessions.json'

# File lưu trang cuối đã tải
PAGE_RECORD_FILE = 'last_page_downloaded.txt'


# ========= HÀM TIỆN ÍCH =========

def read_last_page_downloaded():
    """
    Đọc trang cuối đã tải từ file PAGE_RECORD_FILE.
    Nếu không có hoặc lỗi parse, trả về START_PAGE_DEFAULT.
    """
    if not os.path.exists(PAGE_RECORD_FILE):
        return START_PAGE_DEFAULT
    try:
        with open(PAGE_RECORD_FILE, 'r', encoding='utf-8') as f:
            return int(f.read().strip())
    except:
        return START_PAGE_DEFAULT

def write_last_page_downloaded(page_number):
    """
    Ghi trang cuối cùng đã tải vào PAGE_RECORD_FILE.
    """
    with open(PAGE_RECORD_FILE, 'w', encoding='utf-8') as f:
        f.write(str(page_number))

def append_to_csv(list_of_dicts):
    """
    Ghi (append) danh sách dict vào CSV_FILE, mỗi dict là 1 record.
    """
    if not list_of_dicts:
        return
    df = pd.DataFrame(list_of_dicts)
    file_exists = os.path.isfile(CSV_FILE)
    df.to_csv(CSV_FILE, mode='a', index=False, header=not file_exists, encoding='utf-8')

def append_to_jsonl(list_of_dicts):
    """
    Ghi (append) mỗi dict trong list_of_dicts dưới dạng JSON Lines (mỗi object một dòng).
    """
    if not list_of_dicts:
        return
    with open(JSONL_FILE, 'a', encoding='utf-8') as f:
        for obj in list_of_dicts:
            line = json.dumps(obj, ensure_ascii=False)
            f.write(line + '\n')

def merge_jsonl_to_single_array(jsonl_file, output_json):
    """
    Đọc file .jsonl (mỗi dòng là 1 object),
    gộp lại thành 1 list [obj1, obj2, ...], rồi ghi ra file JSON.
    """
    all_objects = []
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                all_objects.append(obj)

    # Ghi ra file JSON dạng 1 mảng
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(all_objects, f, ensure_ascii=False, indent=4)

    print(f"Đã gộp {len(all_objects)} object vào file '{output_json}'.")


# ========= CODE CHÍNH =========

def main():
    # current_page = read_last_page_downloaded()
    # page_count = 0

    # while page_count < MAX_PAGES:
    #     if page_count > MAX_PAGES:
    #         print(current_page)
    #         break
    #     url = f'https://ev.caltech.edu/api/v1/sessions/{SITE_ID}?page={current_page}'

    #     print(f"\nĐang tải trang {current_page} ...")

    #     try:
    #         response = requests.get(
    #             url,
    #             headers={'Authorization': f'Bearer {API_KEY}'},
    #             timeout=TIMEOUT
    #         )
    #         if response.status_code == 200:
    #             data = response.json()
    #             if '_items' not in data:
    #                 # Không tìm thấy _items => có lẽ hết data => dừng
    #                 print(f"Trang {current_page} không có '_items', dừng.")
    #                 break

    #             items = data['_items']
    #             if items:
    #                 print(f"  -> Tải thành công {len(items)} records.")
    #                 # Ghi CSV
    #                 append_to_csv(items)
    #                 # Ghi JSON Lines (mỗi record một dòng)
    #                 append_to_jsonl(items)
    #             else:
    #                 print(f"  -> Trang {current_page} có 0 records. Có thể đã hết.")

    #             # Ghi lại trang này vào file last_page
    #             write_last_page_downloaded(current_page)

    #             # Kiểm tra link next
    #             links = data.get('_links', {})
    #             next_link = links.get('next')
    #             if not next_link:
    #                 print("Không có next_link => Hết data để tải, dừng.")
    #                 break

    #             current_page += 1
    #             page_count += 1

    #         else:
    #             print(f"Lỗi HTTP {response.status_code} tại trang {current_page}.")
    #             write_last_page_downloaded(current_page)
    #             time.sleep(SLEEP_SECONDS)
    #             # retry cùng trang => không tăng page_count
    #     except requests.exceptions.RequestException as e:
    #         print(f"Lỗi kết nối/timeout tại trang {current_page}: {e}")
    #         write_last_page_downloaded(current_page)
    #         time.sleep(SLEEP_SECONDS)
    #         # retry => không tăng page_count

    # print("\nHOÀN TẤT hoặc đã đạt giới hạn MAX_PAGES.")
    # print(f"File CSV đang ở: {CSV_FILE}")
    # print(f"File JSON Lines đang ở: {JSONL_FILE}")
    # print(f"Nếu muốn có 1 file JSON dạng mảng, hãy chạy hàm merge_jsonl_to_single_array()")

    # -- Nếu muốn tự động gộp JSON Lines => 1 mảng JSON, uncomment dòng dưới:
    merge_jsonl_to_single_array(JSONL_FILE, FINAL_JSON_FILE)


if __name__ == "__main__":
    main()
