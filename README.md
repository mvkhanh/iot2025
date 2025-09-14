# Face Detection & Recognition on Raspberry Pi

Ứng dụng nhận diện và phát hiện khuôn mặt chạy trực tiếp trên Raspberry Pi Zero/2 W hoặc các dòng Pi khác. Bật LED tương ứng với từng người đã enroll, và chế độ enroll từ camera.

## 1. Cài đặt

### Trên Raspberry Pi
```
sudo apt-get install python3-picamera2
pip install numpy opencv-contrib-python
```

## 2. Chạy chương trình

#### Enroll người mới từ camera:
```
python server.py recognition --enroll-from-camera <new_person_name>
```

#### Chạy chế độ nhận diện
```
python server.py recognition
```


#### Hoặc chế độ detection đơn giản
```
python server.py detection
```

🗂 Cấu trúc DB

Thư mục faces_db/ chứa mô hình, ảnh xám đã crop và ảnh gốc của từng người đã enroll.
