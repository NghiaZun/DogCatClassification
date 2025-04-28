# Sử dụng image Python chính thức
FROM python:3.9-slim

# Đặt thư mục làm việc
WORKDIR /app

# Sao chép requirements.txt và cài đặt các phụ thuộc
COPY requirements.txt .  
# Chỉ copy file requirements.txt trước để tận dụng layer caching
RUN pip install --no-cache-dir -r requirements.txt  
# --no-cache-dir giúp giảm dung lượng image

# Sao chép mã nguồn vào container
COPY . .

# Mở port mà ứng dụng sẽ chạy trên
EXPOSE 8080  
# Đây là port mặc định khi chạy Flask trên Cloud Run

# Chạy ứng dụng Flask
CMD ["python", "app.py"]  
# Đảm bảo app.py là tên file chính của bạn
