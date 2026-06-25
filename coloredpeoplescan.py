import cv2
import numpy as np
import threading
import time
from ultralytics import YOLO

model = YOLO(
    r"C:\Users\Monster\Downloads\yolo\drone_yolo_project\runs\detect\human_drone_yolo11n_640\weights\best.pt"
)

PERSON_CLASS_ID = 0

CAMERA_INDEX = 0

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

YOLO_CONF = 0.55

YOLO_IMGSZ = 640

ONLY_DRAW_RED_BLUE = True

MIN_BOX_AREA_RATIO = 0.003

MIN_ASPECT_RATIO = 0.20
MAX_ASPECT_RATIO = 5.00

cap = cv2.VideoCapture(CAMERA_INDEX)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

latest_frame = None
lock = threading.Lock()
running = True


def camera_thread():
    global latest_frame, running

    while running:
        ret, frame = cap.read()

        if ret:
            with lock:
                latest_frame = frame.copy()
        else:
            time.sleep(0.01)


def detect_clothing_color(person_crop):
    h, w = person_crop.shape[:2]

    if h < 20 or w < 10:
        return None, 0.0, 0.0

    y1 = int(h * 0.10)
    y2 = int(h * 0.90)
    x1 = int(w * 0.10)
    x2 = int(w * 0.90)

    crop = person_crop[y1:y2, x1:x2]

    if crop.size == 0:
        return None, 0.0, 0.0

    crop = cv2.GaussianBlur(crop, (5, 5), 0)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([90, 60, 40])
    upper_blue = np.array([135, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    lower_red1 = np.array([0, 70, 40])
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([170, 70, 40])
    upper_red2 = np.array([180, 255, 255])

    red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.add(red_mask1, red_mask2)

    colored_mask = cv2.inRange(
        hsv,
        np.array([0, 50, 40]),
        np.array([180, 255, 255])
    )

    kernel = np.ones((3, 3), np.uint8)

    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    colored_mask = cv2.morphologyEx(colored_mask, cv2.MORPH_OPEN, kernel)

    blue_pixels = cv2.countNonZero(blue_mask)
    red_pixels = cv2.countNonZero(red_mask)
    colored_pixels = cv2.countNonZero(colored_mask)

    if colored_pixels < 15:
        return None, 0.0, 0.0

    blue_ratio = blue_pixels / colored_pixels
    red_ratio = red_pixels / colored_pixels

    min_ratio = 0.35
    dominance_ratio = 1.35

    if red_ratio >= min_ratio and red_ratio > blue_ratio * dominance_ratio:
        return "red", red_ratio, blue_ratio

    if blue_ratio >= min_ratio and blue_ratio > red_ratio * dominance_ratio:
        return "blue", red_ratio, blue_ratio

    return None, red_ratio, blue_ratio


if not cap.isOpened():
    print("Kamera açılamadı.")
    exit()

thread = threading.Thread(target=camera_thread, daemon=True)
thread.start()

time.sleep(1)

prev_time = time.time()

try:
    while True:
        with lock:
            if latest_frame is None:
                continue

            frame = latest_frame.copy()

        results = model(
            frame,
            conf=YOLO_CONF,
            imgsz=YOLO_IMGSZ,
            iou=0.45,
            verbose=False
        )

        h_frame, w_frame = frame.shape[:2]
        frame_area = h_frame * w_frame

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])

                if cls_id != PERSON_CLASS_ID:
                    continue

                confidence = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w_frame, x2)
                y2 = min(h_frame, y2)

                if x2 <= x1 or y2 <= y1:
                    continue

                box_w = x2 - x1
                box_h = y2 - y1
                box_area = box_w * box_h

                area_ratio = box_area / frame_area
                aspect_ratio = box_w / box_h if box_h > 0 else 0

                if area_ratio < MIN_BOX_AREA_RATIO:
                    continue

                if aspect_ratio < MIN_ASPECT_RATIO or aspect_ratio > MAX_ASPECT_RATIO:
                    continue

                person_crop = frame[y1:y2, x1:x2]

                if person_crop.size == 0:
                    continue

                color, red_ratio, blue_ratio = detect_clothing_color(person_crop)

                if color == "red":
                    box_color = (0, 0, 255)
                    label = f"INSAN {confidence:.2f} RED:{red_ratio:.2f}"

                elif color == "blue":
                    box_color = (255, 0, 0)
                    label = f"INSAN {confidence:.2f} BLUE:{blue_ratio:.2f}"

                else:
                    if ONLY_DRAW_RED_BLUE:
                        continue

                    box_color = (180, 180, 180)
                    label = f"DIGER INSAN {confidence:.2f}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)

                cv2.putText(
                    frame,
                    label,
                    (x1, max(y1 - 10, 30)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    box_color,
                    2
                )

        now = time.time()
        fps = 1 / (now - prev_time)
        prev_time = now

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow("Drone Mavi / Kirmizi Kiyafetli Insan Tespiti", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    running = False
    time.sleep(0.1)
    cap.release()
    cv2.destroyAllWindows()