import threading
import ffmpeg
import numpy as np
import cv2
import time
from twilio.rest import Client
from ultralytics import YOLO

# Cấu hình Twilio
account_sid = ''  # Thay bằng Account SID của bạn
auth_token = ''  # Thay bằng Auth Token của bạn
twilio_phone_number = ''  # Thay bằng số điện thoại Twilio của bạn
target_phone_number = ''  # Thay bằng số điện thoại nhận tin nhắn

# Khởi tạo Twilio Client
twilio_client = Client(account_sid, auth_token)

def send_sms(body):
    try:
        message = twilio_client.messages.create(
            body=body,
            from_=twilio_phone_number,
            to=target_phone_number
        )
        print(f"Tin nhắn đã được gửi thành công. SID: {message.sid}")
    except Exception as e:
        print(f"Đã xảy ra lỗi: {e}")

# Biến chung để chia sẻ khung hình giữa các luồng
shared_frame = None
frame_lock = threading.Lock()
stop_event = threading.Event()  # Cờ để dừng các luồng

# Hàm để xử lý video bằng ffmpeg
def run_ffmpeg(cam_source):
    global shared_frame
    args = {"rtsp_transport": "udp"}
    probe = ffmpeg.probe(cam_source)
    cap_info = next(x for x in probe["streams"] if x["codec_type"] == "video")
    print("fps: {}".format(cap_info["r_frame_rate"]))
    width = cap_info["width"]
    height = cap_info["height"]
    up, down = str(cap_info["r_frame_rate"]).split("/")
    fps = eval(up) / eval(down)
    print("fps: {}".format(fps))
    process1 = (
        ffmpeg.input(cam_source, **args)
        .output("pipe:", format="rawvideo", pix_fmt="rgb24")
        .overwrite_output()
        .run_async(pipe_stdout=True)
    )
    while not stop_event.is_set():
        in_bytes = process1.stdout.read(width * height * 3)
        if not in_bytes:
            break
        in_frame = np.frombuffer(in_bytes, np.uint8).reshape([height, width, 3])
        frame = cv2.resize(in_frame, (384, 384))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Cập nhật khung hình chung
        with frame_lock:
            shared_frame = frame

        # cv2.imshow("ffmpeg", frame)
        # if cv2.waitKey(1) == ord("q"):
        #     stop_event.set()
        #     break
    process1.kill()

# Hàm để phát hiện lửa và gửi SMS
def run_fire_detection():
    global shared_frame
    model = YOLO("best (2).pt")

    fire_start_time = None
    fire_detected = False
    fire_sms_sent = False

    while not stop_event.is_set():
        with frame_lock:
            if shared_frame is not None:
                frame = shared_frame.copy()
            else:
                continue

        results = model.predict(frame)
        fire_in_frame = False
        for result in results:
            for box in result.boxes:
                label = result.names[int(box.cls)]
                if label == "fire":
                    fire_in_frame = True
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        if fire_in_frame:
            if fire_start_time is None:
                fire_start_time = time.time()
            elif time.time() - fire_start_time > 3:
                fire_detected = True
        else:
            fire_start_time = None
            fire_detected = False
            fire_sms_sent = False

        if fire_detected and not fire_sms_sent:
            send_sms("Có lửa!")
            fire_sms_sent = True
        elif not fire_detected:
            print("No fire")

        # Hiển thị frame (tuỳ chọn)
        cv2.imshow("Fire Detection", frame)
        if cv2.waitKey(1) == ord('q'):
            stop_event.set()
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    cam_source = ""
    
    thread1 = threading.Thread(target=run_ffmpeg, args=(cam_source,))
    thread2 = threading.Thread(target=run_fire_detection)

    thread1.start()
    thread2.start()

    thread1.join()
    thread2.join()
