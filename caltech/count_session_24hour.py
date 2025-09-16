import json

def count_objects_for_key(file_path, date_key):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    # Kiểm tra nếu key tồn tại trong data
    if date_key in data:
        return len(data[date_key])
    else:
        # Trường hợp key không tồn tại
        return 0

# Ví dụ cách dùng
if __name__ == "__main__":
    file_path = "final_gop.json"  # thay bằng đường dẫn của bạn
    date_key = "06-Sep-2018" # thay bằng date key bạn muốn
    date_key2 = "07-Sep-2018" # thay bằng date key bạn muốn
    result = count_objects_for_key(file_path, date_key) + count_objects_for_key(file_path, date_key2)
    print(f"Số object cho key '{date_key}' là: {result}")
