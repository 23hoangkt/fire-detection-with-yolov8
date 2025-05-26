import cv2
import time
from ultralytics import YOLO

# Tải mô hình YOLOv8
model = YOLO(r"C:\Users\hoang\Desktop\fire-detection-with-yolov8\best.pt")

# Khởi tạo webcam
cap = cv2.VideoCapture(0)

# Biến để theo dõi thời gian lửa xuất hiện
fire_start_time = None
fire_detected = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Thực hiện dự đoán với ngưỡng confidence và NMS
    results = model.predict(frame, conf=0.5, iou=0.7)

    # Kiểm tra nếu lửa được phát hiện
    fire_in_frame = False
    for result in results:
        for box in result.boxes:
            confidence = box.conf.item()
            label = result.names[int(box.cls)]
            if label == "fire" and confidence > 0.8:
                fire_in_frame = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Cập nhật thời gian phát hiện lửa
    if fire_in_frame:
        if fire_start_time is None:
            fire_start_time = time.time()
        elif time.time() - fire_start_time > 3:
            fire_detected = True
    else:
        fire_start_time = None
        fire_detected = False

    # In thông báo
    print("Fire!" if fire_detected else "No fire")

    # Hiển thị frame
    cv2.imshow("Fire Detection", frame)

    # Thoát nếu nhấn 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng webcam và đóng cửa sổ
cap.release()
cv2.destroyAllWindows()