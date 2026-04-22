import os
import base64
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel

torch.serialization.add_safe_globals([DetectionModel])

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Render-safe model choice:
# Use yolov8n.pt in deployment. Ultralytics can auto-download it at runtime.
MODEL_PATH = os.path.join(BASE_DIR, "yolov8n.pt")

model = None
TARGET_CLASS_IDS = None

TARGET_CLASSES = {
    "person": ("warning", "Person"),
    "bicycle": ("warning", "Bicycle"),
    "motorcycle": ("warning", "Motorcycle"),
    "car": ("warning", "Car"),
    "bus": ("urgent", "Bus"),
    "truck": ("urgent", "Truck"),
    "bench": ("caution", "Obstacle"),
    "chair": ("caution", "Obstacle"),
    "potted plant": ("caution", "Obstacle"),
    "fire hydrant": ("warning", "Pole-like obstacle"),
    "stop sign": ("caution", "Sign or pole"),
    "parking meter": ("warning", "Pole-like obstacle"),
    "traffic light": ("caution", "Traffic light"),
}

CONFIDENCE_THRESHOLD = 0.22
IOU_THRESHOLD = 0.50
PREDICT_IMGSZ = 512
MAX_RETURNED_DETECTIONS = 6

CLASS_MIN_CONFIDENCE = {
    "person": 0.20,
    "bicycle": 0.24,
    "motorcycle": 0.26,
    "car": 0.22,
    "bus": 0.24,
    "truck": 0.24,
    "bench": 0.30,
    "chair": 0.28,
    "potted plant": 0.30,
    "fire hydrant": 0.22,
    "stop sign": 0.22,
    "parking meter": 0.20,
    "traffic light": 0.20,
}

CLASS_MIN_AREA_RATIO = {
    "person": 0.0012,
    "bicycle": 0.0020,
    "motorcycle": 0.0020,
    "car": 0.0025,
    "bus": 0.0035,
    "truck": 0.0035,
    "bench": 0.0035,
    "chair": 0.0025,
    "potted plant": 0.0028,
    "fire hydrant": 0.0010,
    "stop sign": 0.0010,
    "parking meter": 0.0009,
    "traffic light": 0.0009,
}


def get_model():
    global model
    if model is None:
        model = YOLO(MODEL_PATH)
    return model


def _build_target_class_ids():
    global TARGET_CLASS_IDS
    if TARGET_CLASS_IDS is not None:
        return TARGET_CLASS_IDS

    names = get_model().names
    target_ids = []

    for cls_id, class_name in names.items():
        if class_name in TARGET_CLASSES:
            target_ids.append(cls_id)

    TARGET_CLASS_IDS = target_ids
    return TARGET_CLASS_IDS


def decode_base64_image(data_url: str):
    if not data_url or "," not in data_url:
        return None

    encoded = data_url.split(",", 1)[1]
    image_bytes = base64.b64decode(encoded)
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return image


def preprocess_frame(frame):
    if frame is None:
        return None

    enhanced = frame.copy()
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)

    lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)

    merged = cv2.merge((l_channel, a_channel, b_channel))
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    return enhanced


def make_result(
    hazard_detected,
    hazard_type,
    severity,
    message,
    distance_label=None,
    direction_label=None,
    bbox=None,
    detections=None
):
    return {
        "hazard_detected": hazard_detected,
        "hazard_type": hazard_type,
        "severity": severity,
        "message": message,
        "distance_label": distance_label,
        "direction_label": direction_label,
        "bbox": bbox,
        "detections": detections or []
    }


def _box_area_ratio(x1, y1, x2, y2, width, height):
    box_area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
    frame_area = float(width * height)
    if frame_area <= 0:
        return 0.0
    return box_area / frame_area


def _box_bottom_ratio(y2, height):
    return float(y2) / max(float(height), 1.0)


def _get_direction_label(x1, x2, width):
    cx = (x1 + x2) / 2.0
    left_boundary = width * 0.35
    right_boundary = width * 0.65

    if cx < left_boundary:
        return "left"
    if cx > right_boundary:
        return "right"
    return "center"


def _get_distance_label(area_ratio, bottom_ratio):
    if area_ratio >= 0.18 or bottom_ratio >= 0.92:
        return "very_close"
    if area_ratio >= 0.08 or bottom_ratio >= 0.84:
        return "close"
    if area_ratio >= 0.02 or bottom_ratio >= 0.68:
        return "ahead"
    return "far"


def _distance_phrase(distance_label):
    phrases = {
        "very_close": "very close",
        "close": "close",
        "ahead": "ahead",
        "far": "far ahead"
    }
    return phrases.get(distance_label, "ahead")


def _direction_phrase(direction_label):
    phrases = {
        "left": "on your left",
        "center": "ahead",
        "right": "on your right"
    }
    return phrases.get(direction_label, "ahead")


def _is_relevant_box(class_name, conf, x1, y1, x2, y2, width, height):
    min_conf = CLASS_MIN_CONFIDENCE.get(class_name, CONFIDENCE_THRESHOLD)
    if conf < min_conf:
        return False

    area_ratio = _box_area_ratio(x1, y1, x2, y2, width, height)
    min_area_ratio = CLASS_MIN_AREA_RATIO.get(class_name, 0.003)
    if area_ratio < min_area_ratio:
        return False

    box_width = max(1.0, x2 - x1)
    box_height = max(1.0, y2 - y1)
    aspect_ratio = box_height / box_width

    if class_name == "person" and aspect_ratio < 0.45:
        return False

    return True


def _severity_weight(severity):
    if severity == "urgent":
        return 0.65
    if severity == "warning":
        return 0.38
    return 0.20


def _center_path_bonus(direction_label, distance_label):
    bonus = 0.0
    if direction_label == "center":
        bonus += 0.25
    if distance_label == "very_close":
        bonus += 0.75
    elif distance_label == "close":
        bonus += 0.42
    elif distance_label == "ahead":
        bonus += 0.18
    else:
        bonus += 0.08
    return bonus


def _build_object_message(class_name, distance_label, direction_label, severity):
    friendly_name = TARGET_CLASSES.get(class_name, ("caution", class_name))[1]
    distance_text = _distance_phrase(distance_label)
    direction_text = _direction_phrase(direction_label)

    prefix = "Caution"
    if severity == "warning":
        prefix = "Warning"
    elif severity == "urgent":
        prefix = "Urgent"

    if direction_label == "center":
        return f"{prefix}. {friendly_name} {distance_text}."

    return f"{prefix}. {friendly_name} {distance_text} {direction_text}."


def _build_detection_entry(class_name, severity, conf, x1, y1, x2, y2, width, height):
    area_ratio = _box_area_ratio(x1, y1, x2, y2, width, height)
    bottom_ratio = _box_bottom_ratio(y2, height)
    distance_label = _get_distance_label(area_ratio, bottom_ratio)
    direction_label = _get_direction_label(x1, x2, width)

    box_center_x = (x1 + x2) / 2.0
    horizontal_penalty = abs(box_center_x - width / 2.0) / max(width / 2.0, 1.0)

    score = (
        (conf * 1.9)
        + (area_ratio * 3.2)
        + _severity_weight(severity)
        + _center_path_bonus(direction_label, distance_label)
        - (horizontal_penalty * 0.9)
    )

    return {
        "hazard_detected": True,
        "hazard_type": class_name.replace(" ", "_"),
        "severity": severity,
        "message": _build_object_message(class_name, distance_label, direction_label, severity),
        "distance_label": distance_label,
        "direction_label": direction_label,
        "bbox": {
            "x1": int(x1),
            "y1": int(y1),
            "x2": int(x2),
            "y2": int(y2)
        },
        "conf": round(conf, 3),
        "score": score
    }


def detect_hazard_from_frame(frame):
    if frame is None:
        return make_result(False, None, None, "No valid frame available.", detections=[])

    frame = preprocess_frame(frame)
    height, width = frame.shape[:2]

    target_ids = _build_target_class_ids()

    results = get_model().predict(
        source=frame,
        verbose=False,
        conf=CONFIDENCE_THRESHOLD,
        iou=IOU_THRESHOLD,
        imgsz=PREDICT_IMGSZ,
        max_det=16,
        classes=target_ids,
        device="cpu"
    )

    candidates = []

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue

        for box in boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            class_name = get_model().names.get(cls_id, str(cls_id))

            if class_name not in TARGET_CLASSES:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            if not _is_relevant_box(class_name, conf, x1, y1, x2, y2, width, height):
                continue

            severity, _friendly_name = TARGET_CLASSES[class_name]
            candidates.append(
                _build_detection_entry(class_name, severity, conf, x1, y1, x2, y2, width, height)
            )

    candidates.sort(key=lambda item: item["score"], reverse=True)
    top_detections = candidates[:MAX_RETURNED_DETECTIONS]

    if top_detections:
        best = top_detections[0]
        return make_result(
            True,
            best["hazard_type"],
            best["severity"],
            best["message"],
            distance_label=best["distance_label"],
            direction_label=best["direction_label"],
            bbox=best["bbox"],
            detections=top_detections
        )

    return make_result(False, None, None, "No immediate hazard detected.", detections=[])


def detect_hazard_from_base64(data_url):
    frame = decode_base64_image(data_url)
    return detect_hazard_from_frame(frame)