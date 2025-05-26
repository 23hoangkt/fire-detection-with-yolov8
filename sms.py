import cv2
import time
import ffmpeg
import numpy as np
import threading
from ultralytics import YOLO

import requests
from requests.auth import HTTPBasicAuth

import requests
from requests.auth import HTTPBasicAuth

def send_fire_alert():
    url = ""
    username = "sms"
    password = ""

    payload = {
        "message": "Có lửa kìa chạy mau",
        "phoneNumbers": [""],
    }

    try:
        response = requests.post(url, json=payload, auth=HTTPBasicAuth(username, password))
        print(f" Đã gửi SMS cảnh báo (API nội bộ): {response.status_code}")
    except Exception as e:
        print(f" Gửi SMS thất bại: {e}")


# === [2] Lớp đọc luồng RTSP bằng thread ===
class RTSPReader(threading.Thread):
    def __init__(self, cam_source):
        super().__init__()
        self.cam_source = cam_source
        self.frame = None
        self.running = True
        self.lock = threading.Lock()

        args = {"rtsp_transport": "udp"}
        probe = ffmpeg.probe(cam_source)
        cap_info = next(x for x in probe["streams"] if x["codec_type"] == "video")
        self.width = cap_info["width"]
        self.height = cap_info["height"]
        self.process = (
            ffmpeg.input(cam_source, **args)
            .output("pipe:", format="rawvideo", pix_fmt="rgb24")
            .overwrite_output()
            .run_async(pipe_stdout=True)
        )

    def run(self):
        while self.running:
            in_bytes = self.process.stdout.read(self.width * self.height * 3)
            if not in_bytes:
                continue
            frame = np.frombuffer(in_bytes, np.uint8).reshape([self.height, self.width, 3])
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame = cv2.resize(frame, (640, 360))
            frame = cv2.GaussianBlur(frame, (5, 5), 0)
            with self.lock:
                self.frame = frame

    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False
        self.process.kill()


def main(cam_source):
    model = YOLO(r"C:\Users\hoang\Desktop\fire-detection-with-yolov8\best.pt")

    rtsp_reader = RTSPReader(cam_source)
    rtsp_reader.start()

    fire_start_time = None
    fire_detected = False
    alert_sent = False

    try:
        while True:
            frame = rtsp_reader.get_frame()
            if frame is None:
                continue

            results = model.predict(frame, conf=0.6, iou=0.7)

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

            if fire_in_frame:
                if fire_start_time is None:
                    fire_start_time = time.time()
                elif time.time() - fire_start_time > 1:
                    fire_detected = True
                    if not alert_sent:
                        send_fire_alert()
                        alert_sent = True
            else:
                fire_start_time = None
                fire_detected = False
                alert_sent = False

            print(" Fire detected!" if fire_detected else " No fire")

            cv2.imshow("Fire Detection ", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        rtsp_reader.stop()
        rtsp_reader.join()
        cv2.destroyAllWindows()

# === [4] Chạy chương trình ===
if __name__ == "__main__":
    cam_source = "your link rtsp"
    main(cam_source)
