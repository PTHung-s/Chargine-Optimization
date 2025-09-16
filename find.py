import requests
import json

api_key = 'GXnuC_Pu-jqwLtxp28l55DV4E6rJFSnNioGMBA2XN7k'
site_id = 'caltech'
start_page = 721  # Bắt đầu từ trang 600
target_id = "5cafdc24f9af8b3a92caa8f3"

# URL API với tham số page
url = f'https://ev.caltech.edu/api/v1/sessions/{site_id}?page={start_page}'

headers = {
    'Authorization': f'Bearer {api_key}'
}

page_count = 0
max_pages = 1000  # Giới hạn số trang quét để tránh lặp vô hạn
found = False

while url and page_count < max_pages:
    print(f"Đang tải trang {start_page + page_count}...")
    try:
        response = requests.get(url, headers=headers, timeout=100)
        if response.status_code == 200:
            data = response.json()
            if '_items' in data:
                for item in data['_items']:
                    if item.get('_id') == target_id:
                        print(f"Đã tìm thấy _id {target_id} tại trang {start_page + page_count}")
                        found = True
                        break  # Thoát khỏi vòng lặp khi tìm thấy
            else:
                print(f"Lỗi: Không tìm thấy dữ liệu '_items' tại trang {start_page + page_count}")
                break

            if found:
                break  # Dừng toàn bộ vòng lặp nếu tìm thấy

            # Lấy URL trang tiếp theo nếu có
            next_page = data['_links'].get('next', None)
            if next_page:
                url = f"https://ev.caltech.edu/api/v1/{next_page['href']}"
            else:
                url = None
        else:
            print(f'Yêu cầu không thành công: {response.status_code}')
            break
    except requests.exceptions.RequestException as e:
        print(f"Lỗi kết nối: {e}")
        break

    page_count += 1

if not found:
    print(f"Không tìm thấy _id {target_id} sau {page_count} trang đã quét.")
