# Fire Detection with YOLOv8

## Mô tả
Dự án **Fire Detection with YOLOv8** triển khai hệ thống phát hiện đám cháy sử dụng mô hình YOLOv8 từ Ultralytics. Hệ thống tích hợp camera IP qua giao thức **RTSP** để giám sát thời gian thực và gửi tin nhắn SMS cảnh báo khi phát hiện đám cháy thông qua [Android SMS Gateway](https://github.com/capcom6/android-sms-gateway). Dự án phù hợp cho các ứng dụng như giám sát an toàn, theo dõi cháy rừng, hoặc hệ thống cảnh báo cháy tự động.

![SYSTEM](image.png)
## Tính năng
- Suy luận trên ảnh, video, webcam, hoặc luồng RTSP từ camera IP để phát hiện đám cháy.
- Hiển thị kết quả với khung bao quanh vùng cháy.
- Tích hợp camera IP sử dụng giao thức RTSP để giám sát thời gian thực.
- Gửi tin nhắn SMS cảnh báo khi phát hiện đám cháy qua [Android SMS Gateway](https://github.com/capcom6/android-sms-gateway).
- Hỗ trợ tùy chỉnh để cải thiện hiệu suất trên các nguồn dữ liệu khác nhau.

## Yêu cầu
- Python 3.8+
- Thư viện Ultralytics YOLOv8
- OpenCV (`opencv-python`) để xử lý luồng RTSP và webcam
- NumPy
- Thư viện `requests` để giao tiếp với Android SMS Gateway
- Thiết bị Android với ứng dụng **Android SMS Gateway** được cài đặt và cấu hình
- Camera IP hỗ trợ giao thức RTSP

## Cài đặt
1. **Tải kho lưu trữ**:
   ```bash
   git clone https://github.com/23hoangkt/fire-detection-with-yolov8.git
   cd fire-detection-with-yolov8
   ```

2. **Cài đặt môi trường**:
   Tạo môi trường ảo và cài đặt các thư viện cần thiết:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/MacOS
   venv\Scripts\activate      # Windows
   pip install -r requirements.txt
   ```

3. **Cấu hình camera IP (RTSP)**:
   - Lấy URL RTSP từ camera IP (thường có dạng: `rtsp://username:password@ip_address:port/stream`).
   - Cập nhật URL RTSP trong tệp `config.yaml` hoặc trực tiếp trong mã (ví dụ: `fire_detect.py` hoặc `sms.py`).
   - Đảm bảo camera IP và hệ thống chạy trên cùng mạng hoặc cấu hình mạng phù hợp (ví dụ: mở cổng 554 cho RTSP).
   - Kiểm tra kết nối RTSP bằng công cụ như VLC hoặc lệnh:
     ```bash
     python rtsp.py
     ```

4. **Cấu hình Android SMS Gateway**:
   - Tải và cài đặt ứng dụng Android SMS Gateway từ [kho lưu trữ](https://github.com/capcom6/android-sms-gateway).
   - Chạy ứng dụng trên thiết bị Android và ghi chú URL API (mặc định: `http://<device-ip>:8080`).
   - Cập nhật thông tin API (URL, số điện thoại nhận cảnh báo) trong tệp `config.yaml` hoặc trực tiếp trong mã (ví dụ: `sms.py`).
   - Đảm bảo thiết bị Android kết nối cùng mạng với hệ thống.

## Hướng dẫn sử dụng
### 1. Phát hiện trên ảnh/video/luồng RTSP/webcam
Phát hiện đám cháy trên ảnh, video, webcam, hoặc luồng RTSP từ camera IP:
- **Trên ảnh**:
  ```bash
  python valid.py 
  ```
- **Trên video hoặc webcam**:
  ```bash
  python fire_detect.py 
  ```
  
- **Trên camera IP (RTSP)**:
  ```bash
  python main.py 
  ```

### 2. Gửi cảnh báo SMS
Khi phát hiện đám cháy, hệ thống tự động gửi tin nhắn SMS qua Android SMS Gateway:
- Chạy hệ thống tích hợp camera IP và gửi SMS:
  ```bash
  python sms.py 
  ```
- Đảm bảo ứng dụng Android SMS Gateway đang chạy và URL API được cấu hình đúng trong  `sms.py`.


### 3. Kết quả
- Kết quả suy luận được lưu trong thư mục `runs/detect/exp/` với khung bao quanh vùng cháy.

## Đóng góp
Chúng tôi hoan nghênh mọi đóng góp! Để đóng góp:
1. Fork kho lưu trữ.
2. Tạo nhánh mới: `git checkout -b feature/ten-tinh-nang`.
3. Commit thay đổi: `git commit -m 'Thêm tính năng XYZ'`.
4. Push lên nhánh: `git push origin feature/ten-tinh-nang`.
5. Tạo Pull Request.

## Liên hệ
Nếu bạn có câu hỏi hoặc cần hỗ trợ, hãy mở issue trên GitHub hoặc liên hệ qua email: [hoangkimtruong2003@gmail.com].

## Giấy phép
Dự án được cấp phép theo [MIT License](LICENSE).
