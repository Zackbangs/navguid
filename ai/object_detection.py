import os
import base64
import cv2
import numpy as np
from ultralytics import YOLO

# 🔥 USE STRONGER MODEL
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "yolov8m.pt")
MODEL_PATH = os.path.abspath(MODEL_PATH)

model = YOLO(MODEL_PATH)

TARGET_CLASSES = {
    "person": ("warning", "Person"),
    "bicycle": ("warning", "Bicycle"),
    "motorcycle": ("warning", "Motorcycle"),
    "car": ("warning", "Car"),
    "bus": ("urgent", "Bus"),
    "truck": ("urgent", "Truck"),
    "chair": ("caution", "Obstacle"),
    "bench": ("caution", "Obstacle"),
}

CONF_THRESHOLD = 0.3

def decode_base64_image(data_url):
    encoded = data_url.split(",")[1]
    img_bytes = base64.b64decode(encoded)
    arr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def get_direction(x, width):
    center = width / 2
    if x < center * 0.7:
        return "left"
    elif x > center * 1.3:
        return "right"
    return "ahead"

def get_distance(area):
    if area > 0.2:
        return "very close"
    elif area > 0.1:
        return "close"
    elif area > 0.03:
        return "ahead"
    return "far"

def detect_hazard_from_frame(frame):
    h, w = frame.shape[:2]

    results = model.predict(
        source=frame,
        conf=CONF_THRESHOLD,
        imgsz=640,
        verbose=False
    )

    best = None
    best_score = 0

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            name = model.names[cls_id]

            if name not in TARGET_CLASSES:
                continue

            x1, y1, x2, y2 = box.xyxy[0]
            area = ((x2 - x1) * (y2 - y1)) / (w * h)

            direction = get_direction((x1 + x2) / 2, w)
            distance = get_distance(area)

            severity, label = TARGET_CLASSES[name]

            score = conf + area
            if distance == "very close":
                score += 1

            if score > best_score:
                best_score = score
                best = {
                    "hazard_detected": True,
                    "hazard_type": name,
                    "severity": severity,
                    "message": f"{label} {distance} {direction}",
                    "distance_label": distance,
                    "direction_label": direction
                }

    if best:
        return best

    return {
        "hazard_detected": False,
        "message": "No immediate hazard"
    }

def detect_hazard_from_base64(data_url):
    frame = decode_base64_image(data_url)
    return detect_hazard_from_frame(frame)