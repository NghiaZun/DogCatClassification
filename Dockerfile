# Sử dụng image Python chính thức
FROM python:3.9-slim

# Đặt thư mục làm việc
WORKDIR /app

# Sao chép requirements.txt và cài đặt các phụ thuộc
COPY requirements.txt .
RUN pip install -r requirements.txt

# Sao chép mã nguồn vào container
COPY . .

# Mở port mà ứng dụng sẽ chạy trên
EXPOSE 8080

# Chạy ứng dụng Flask
CMD ["python", "app.py"]
