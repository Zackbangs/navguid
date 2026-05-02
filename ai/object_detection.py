import os
import base64
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel


"""
NavGuid Hazard Detection Module
-------------------------------

Render-safe hazard detection for NavGuid.

Current model:
- Primary: yolov8m.pt
- Fallback: yolov8n.pt if yolov8m.pt is missing

This keeps detection stronger than yolov8n while still protecting Render stability.
"""


torch.serialization.add_safe_globals([DetectionModel])

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

PRIMARY_MODEL_PATH = os.path.join(BASE_DIR, "yolov8m.pt")
FALLBACK_MODEL_PATH = os.path.join(BASE_DIR, "yolov8n.pt")

MODEL_PATH = PRIMARY_MODEL_PATH if os.path.exists(PRIMARY_MODEL_PATH) else FALLBACK_MODEL_PATH

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
    "backpack": ("caution", "Bag or obstacle"),
    "suitcase": ("caution", "Bag or obstacle"),
    "umbrella": ("caution", "Obstacle"),
    "sports ball": ("caution", "Small obstacle"),

    "fire hydrant": ("warning", "Pole-like obstacle"),
    "stop sign": ("caution", "Sign or pole"),
    "parking meter": ("warning", "Pole-like obstacle"),
    "traffic light": ("caution", "Traffic light"),

    "dog": ("warning", "Dog"),
}


CONFIDENCE_THRESHOLD = 0.20
IOU_THRESHOLD = 0.48

# Good balance for yolov8m on Render Starter.
PREDICT_IMGSZ = 576

MAX_RETURNED_DETECTIONS = 8


CLASS_MIN_CONFIDENCE = {
    "person": 0.18,
    "bicycle": 0.22,
    "motorcycle": 0.22,
    "car": 0.20,
    "bus": 0.22,
    "truck": 0.22,

    "bench": 0.25,
    "chair": 0.24,
    "potted plant": 0.25,
    "backpack": 0.22,
    "suitcase": 0.22,
    "umbrella": 0.24,
    "sports ball": 0.24,

    "fire hydrant": 0.20,
    "stop sign": 0.20,
    "parking meter": 0.20,
    "traffic light": 0.20,
    "dog": 0.22,
}


CLASS_MIN_AREA_RATIO = {
    "person": 0.0008,
    "bicycle": 0.0014,
    "motorcycle": 0.0015,
    "car": 0.0018,
    "bus": 0.0028,
    "truck": 0.0028,

    "bench": 0.0025,
    "chair": 0.0018,
    "potted plant": 0.0020,
    "backpack": 0.0014,
    "suitcase": 0.0015,
    "umbrella": 0.0015,
    "sports ball": 0.0012,

    "fire hydrant": 0.0007,
    "stop sign": 0.0007,
    "parking meter": 0.0006,
    "traffic light": 0.0006,
    "dog": 0.0014,
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
    TARGET_CLASS_IDS = [
        cls_id for cls_id, class_name in names.items()
        if class_name in TARGET_CLASSES
    ]

    return TARGET_CLASS_IDS


def decode_base64_image(data_url: str):
    if not data_url or "," not in data_url:
        return None

    try:
        encoded = data_url.split(",", 1)[1]
        image_bytes = base64.b64decode(encoded)
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        return cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    except Exception:
        return None


def preprocess_frame(frame):
    if frame is None:
        return None

    enhanced = frame.copy()

    lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)

    merged = cv2.merge((l_channel, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


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
        "success": True,
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
    box_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    frame_area = float(width * height)

    if frame_area <= 0:
        return 0.0

    return box_area / frame_area


def _box_bottom_ratio(y2, height):
    return float(y2) / max(float(height), 1.0)


def _get_direction_label(x1, x2, width):
    cx = (x1 + x2) / 2.0

    if cx < width * 0.34:
        return "left"

    if cx > width * 0.66:
        return "right"

    return "center"


def _get_distance_label(area_ratio, bottom_ratio):
    if area_ratio >= 0.16 or bottom_ratio >= 0.91:
        return "very_close"

    if area_ratio >= 0.065 or bottom_ratio >= 0.82:
        return "close"

    if area_ratio >= 0.012 or bottom_ratio >= 0.62:
        return "ahead"

    return "far"


def _distance_phrase(distance_label):
    return {
        "very_close": "very close",
        "close": "close",
        "ahead": "ahead",
        "far": "far ahead"
    }.get(distance_label, "ahead")


def _direction_phrase(direction_label):
    return {
        "left": "on your left",
        "center": "ahead",
        "right": "on your right"
    }.get(direction_label, "ahead")


def _is_relevant_box(class_name, conf, x1, y1, x2, y2, width, height):
    min_conf = CLASS_MIN_CONFIDENCE.get(class_name, CONFIDENCE_THRESHOLD)

    if conf < min_conf:
        return False

    area_ratio = _box_area_ratio(x1, y1, x2, y2, width, height)
    min_area_ratio = CLASS_MIN_AREA_RATIO.get(class_name, 0.002)

    if area_ratio < min_area_ratio:
        return False

    box_width = max(1.0, x2 - x1)
    box_height = max(1.0, y2 - y1)
    aspect_ratio = box_height / box_width

    if class_name == "person" and aspect_ratio < 0.35:
        return False

    return True


def _severity_weight(severity):
    if severity == "urgent":
        return 0.75

    if severity == "warning":
        return 0.45

    return 0.22


def _class_priority(class_name):
    priorities = {
        "person": 0.35,
        "car": 0.32,
        "truck": 0.42,
        "bus": 0.42,
        "motorcycle": 0.34,
        "bicycle": 0.30,
        "dog": 0.28,
        "fire hydrant": 0.22,
        "parking meter": 0.22,
        "stop sign": 0.16,
        "traffic light": 0.16,
        "chair": 0.18,
        "bench": 0.18,
        "backpack": 0.16,
        "suitcase": 0.18,
        "umbrella": 0.14,
        "sports ball": 0.12,
        "potted plant": 0.14,
    }

    return priorities.get(class_name, 0.12)


def _center_path_bonus(direction_label, distance_label):
    bonus = 0.0

    if direction_label == "center":
        bonus += 0.35

    if distance_label == "very_close":
        bonus += 0.85
    elif distance_label == "close":
        bonus += 0.50
    elif distance_label == "ahead":
        bonus += 0.24
    else:
        bonus += 0.06

    return bonus


def _build_object_message(class_name, distance_label, direction_label, severity):
    friendly_name = TARGET_CLASSES.get(class_name, ("caution", class_name))[1]
    distance_text = _distance_phrase(distance_label)
    direction_text = _direction_phrase(direction_label)

    if severity == "urgent":
        prefix = "Careful"
    elif severity == "warning":
        prefix = "Watch out"
    else:
        prefix = "Caution"

    if direction_label == "center":
        return f"{prefix}. {friendly_name} {distance_text}."

    return f"{prefix}. {friendly_name} {distance_text}, {direction_text}."


def _build_detection_entry(class_name, severity, conf, x1, y1, x2, y2, width, height):
    area_ratio = _box_area_ratio(x1, y1, x2, y2, width, height)
    bottom_ratio = _box_bottom_ratio(y2, height)
    distance_label = _get_distance_label(area_ratio, bottom_ratio)
    direction_label = _get_direction_label(x1, x2, width)

    box_center_x = (x1 + x2) / 2.0
    horizontal_penalty = abs(box_center_x - width / 2.0) / max(width / 2.0, 1.0)

    score = (
        (conf * 2.1)
        + (area_ratio * 3.6)
        + _severity_weight(severity)
        + _class_priority(class_name)
        + _center_path_bonus(direction_label, distance_label)
        - (horizontal_penalty * 0.75)
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
        "score": round(score, 4)
    }


def detect_hazard_from_frame(frame):
    if frame is None:
        return make_result(False, None, None, "No valid frame available.", detections=[])

    frame = preprocess_frame(frame)

    if frame is None:
        return make_result(False, None, None, "No valid frame available.", detections=[])

    height, width = frame.shape[:2]
    target_ids = _build_target_class_ids()

    try:
        results = get_model().predict(
            source=frame,
            verbose=False,
            conf=CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD,
            imgsz=PREDICT_IMGSZ,
            max_det=24,
            classes=target_ids,
            device="cpu"
        )
    except Exception as error:
        return {
            "success": False,
            "hazard_detected": False,
            "hazard_type": None,
            "severity": None,
            "message": f"Hazard detection error: {str(error)}",
            "distance_label": None,
            "direction_label": None,
            "bbox": None,
            "detections": []
        }

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
                _build_detection_entry(
                    class_name,
                    severity,
                    conf,
                    x1,
                    y1,
                    x2,
                    y2,
                    width,
                    height
                )
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