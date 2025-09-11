# Face Detection & Recognition on Raspberry Pi

Ứng dụng nhận diện và phát hiện khuôn mặt chạy trực tiếp trên Raspberry Pi Zero/2 W hoặc các dòng Pi khác. Hỗ trợ streaming ảnh đến máy tính qua TCP/ImageZMQ, bật LED tương ứng với từng người đã enroll, và chế độ enroll từ camera.

- Client (Pi): Captures video (PiCam), chạy detect/recognize → gửi JPEG qua ImageZMQ.
- Server (PC): Nhận JPEG và hiển thị real-time.

## 1. Cài đặt

### Trên Raspberry Pi
```
sudo apt-get install python3-opencv python3-picamera2
pip install imagezmq numpy
```

### Trên PC
```
pip install imagezmq opencv-python numpy
```

## 2. Chạy chương trình

### Server (PC):
```
python server.py --bind 0.0.0.0 --port 9009
```

### Client (Raspberry Pi):

#### Chạy chế độ nhận diện
```
python client.py recognition --server <server_ip>
```

#### Enroll người mới từ camera:
```
python client.py recognition --server <server_ip> --enroll-from-camera <new_person_name>
```

#### Hoặc chế độ detection đơn giản
```
python client.py detection --server <server_ip>
```

🗂 Cấu trúc DB

Thư mục faces_db/ chứa embeddings JSON và ảnh gốc của từng người đã enroll.
