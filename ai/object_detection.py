import os
import base64
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel

# Allowlist Ultralytics DetectionModel for torch weights loading
torch.serialization.add_safe_globals([DetectionModel])

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "yolov8n.pt")
MODEL_PATH = os.path.abspath(MODEL_PATH)

model = YOLO(MODEL_PATH)

TARGET_CLASSES = {
    "person": ("warning", "Person"),
    "bicycle": ("warning", "Bicycle"),
    "motorcycle": ("warning", "Motorcycle"),
    "car": ("warning", "Car"),
    "bus": ("urgent", "Large vehicle"),
    "truck": ("urgent", "Large vehicle"),
    "bench": ("caution", "Obstacle"),
    "chair": ("caution", "Obstacle"),
    "potted plant": ("caution", "Obstacle"),
    "fire hydrant": ("warning", "Pole-like obstacle"),
    "stop sign": ("caution", "Sign or pole"),
}

CONFIDENCE_THRESHOLD = 0.40
MIN_BOX_AREA_RATIO = 0.012
CENTER_ZONE_WEIGHT = 1.15
LOWER_SCREEN_BONUS = 1.10
MAX_RETURNED_DETECTIONS = 5
PREDICT_IMGSZ = 416

CLASS_MIN_CONFIDENCE = {
    "person": 0.38,
    "bicycle": 0.40,
    "motorcycle": 0.42,
    "car": 0.42,
    "bus": 0.40,
    "truck": 0.40,
    "bench": 0.45,
    "chair": 0.45,
    "potted plant": 0.45,
    "fire hydrant": 0.42,
    "stop sign": 0.42,
}

TARGET_CLASS_IDS = None


def _build_target_class_ids():
    global TARGET_CLASS_IDS
    if TARGET_CLASS_IDS is not None:
        return TARGET_CLASS_IDS

    names = model.names
    target_ids = []

    for cls_id, class_name in names.items():
        if class_name in TARGET_CLASSES:
            target_ids.append(cls_id)

    TARGET_CLASS_IDS = target_ids
    return TARGET_CLASS_IDS


def decode_base64_image(data_url: str):
    if "," not in data_url:
        return None

    encoded = data_url.split(",", 1)[1]
    image_bytes = base64.b64decode(encoded)
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return image


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


def center_priority(x1, y1, x2, y2, width, height):
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    horizontal_penalty = abs(cx - width / 2) / max(width / 2, 1)
    vertical_bonus = cy / max(height, 1)
    return (vertical_bonus * LOWER_SCREEN_BONUS) - (horizontal_penalty * CENTER_ZONE_WEIGHT)


def detect_step_or_curb(gray_frame, height, width):
    lower_strip = gray_frame[
        int(height * 0.72):int(height * 0.92),
        int(width * 0.12):int(width * 0.88)
    ]

    edges = cv2.Canny(lower_strip, 60, 150)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=60,
        minLineLength=80,
        maxLineGap=18
    )

    if lines is None:
        return None

    horizontal_count = 0
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(y2 - y1) < 18:
            horizontal_count += 1

    if horizontal_count >= 6:
        return make_result(
            True,
            "step_or_curb",
            "urgent",
            "Urgent. Possible step or curb very close ahead. Slow down and check your path.",
            distance_label="very_close",
            direction_label="center"
        )

    if horizontal_count >= 3:
        return make_result(
            True,
            "step_or_curb",
            "warning",
            "Warning. Possible change in ground level ahead.",
            distance_label="ahead",
            direction_label="center"
        )

    return None


def detect_low_visibility(gray_frame):
    brightness = gray_frame.mean()

    if brightness < 35:
        return make_result(
            True,
            "low_visibility",
            "warning",
            "Warning. Very low visibility ahead. Please slow down.",
            distance_label="ahead",
            direction_label="center"
        )

    if brightness < 50:
        return make_result(
            True,
            "low_visibility",
            "caution",
            "Caution. Low visibility ahead.",
            distance_label="ahead",
            direction_label="center"
        )

    return None


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
    left_boundary = width * 0.36
    right_boundary = width * 0.64

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
    if area_ratio >= 0.03 or bottom_ratio >= 0.72:
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
    if area_ratio < MIN_BOX_AREA_RATIO:
        return False

    box_width = max(1.0, x2 - x1)
    box_height = max(1.0, y2 - y1)
    aspect_ratio = box_height / box_width

    if class_name == "person" and aspect_ratio < 0.8:
        return False

    return True


def _build_object_message(class_name, distance_label, direction_label, severity):
    object_name = class_name.replace("_", " ")
    distance_text = _distance_phrase(distance_label)
    direction_text = _direction_phrase(direction_label)

    prefix = "Caution"
    if severity == "warning":
        prefix = "Warning"
    elif severity == "urgent":
        prefix = "Urgent"

    if direction_label == "center":
        return f"{prefix}. {object_name.capitalize()} {distance_text}."

    return f"{prefix}. {object_name.capitalize()} {distance_text} {direction_text}."


def _severity_weight(severity):
    if severity == "urgent":
        return 0.55
    if severity == "warning":
        return 0.32
    return 0.18


def _build_detection_entry(class_name, severity, conf, x1, y1, x2, y2, width, height):
    area_ratio = _box_area_ratio(x1, y1, x2, y2, width, height)
    bottom_ratio = _box_bottom_ratio(y2, height)
    distance_label = _get_distance_label(area_ratio, bottom_ratio)
    direction_label = _get_direction_label(x1, x2, width)
    position_score = center_priority(x1, y1, x2, y2, width, height)

    score = (
        (conf * 1.8)
        + position_score
        + (area_ratio * 2.4)
        + _severity_weight(severity)
    )

    if distance_label == "very_close":
        score += 0.65
    elif distance_label == "close":
        score += 0.35
    elif distance_label == "ahead":
        score += 0.15

    if direction_label == "center":
        score += 0.18

    message = _build_object_message(class_name, distance_label, direction_label, severity)

    return {
        "hazard_detected": True,
        "hazard_type": class_name.replace(" ", "_"),
        "severity": severity,
        "message": message,
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
        return make_result(False, None, None, "No valid frame available.")

    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    visibility_result = detect_low_visibility(gray)
    if visibility_result:
        return visibility_result

    curb_result = detect_step_or_curb(gray, height, width)
    if curb_result and curb_result["severity"] == "urgent":
        return curb_result

    target_ids = _build_target_class_ids()

    results = model.predict(
        source=frame,
        verbose=False,
        conf=CONFIDENCE_THRESHOLD,
        imgsz=PREDICT_IMGSZ,
        max_det=8,
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
            class_name = model.names.get(cls_id, str(cls_id))

            if class_name not in TARGET_CLASSES:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            if not _is_relevant_box(class_name, conf, x1, y1, x2, y2, width, height):
                continue

            severity, _friendly_name = TARGET_CLASSES[class_name]
            candidate = _build_detection_entry(
                class_name, severity, conf, x1, y1, x2, y2, width, height
            )
            candidates.append(candidate)

    candidates.sort(key=lambda item: item["score"], reverse=True)
    top_detections = candidates[:MAX_RETURNED_DETECTIONS]

    if top_detections:
        best_candidate = top_detections[0]
        return make_result(
            True,
            best_candidate["hazard_type"],
            best_candidate["severity"],
            best_candidate["message"],
            distance_label=best_candidate["distance_label"],
            direction_label=best_candidate["direction_label"],
            bbox=best_candidate["bbox"],
            detections=top_detections
        )

    if curb_result:
        return curb_result

    return make_result(False, None, None, "No immediate hazard detected.", detections=[])


def detect_hazard_from_base64(data_url):
    frame = decode_base64_image(data_url)
    return detect_hazard_from_frame(frame)