from ultralytics import YOLO

# Tạo một đối tượng mô hình YOLOv8
model = YOLO('best (2).pt')

# Gọi phương thức predict trên đối tượng mô hình với các đối số đã chỉ định
model.predict(source="images.jpg" , imgsz=640, conf=0.6, save = True)



