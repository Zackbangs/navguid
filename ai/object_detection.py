import os
import base64
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel

# Allowlist Ultralytics DetectionModel for torch weights loading
# Needed because newer PyTorch defaults torch.load(weights_only=True)
torch.serialization.add_safe_globals([DetectionModel])

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "yolov8n.pt")
MODEL_PATH = os.path.abspath(MODEL_PATH)

model = YOLO(MODEL_PATH)

TARGET_CLASSES = {
    "person": ("warning", "Warning. Person ahead."),
    "bicycle": ("warning", "Warning. Bicycle ahead."),
    "motorcycle": ("warning", "Warning. Motorcycle ahead."),
    "car": ("warning", "Warning. Car ahead."),
    "bus": ("urgent", "Urgent. Large vehicle ahead."),
    "truck": ("urgent", "Urgent. Large vehicle ahead."),
    "bench": ("caution", "Caution. Obstacle ahead."),
    "chair": ("caution", "Caution. Obstacle ahead."),
    "potted plant": ("caution", "Caution. Obstacle ahead."),
    "fire hydrant": ("warning", "Warning. Pole-like obstacle ahead."),
    "stop sign": ("caution", "Caution. Sign or pole ahead."),
}

CONFIDENCE_THRESHOLD = 0.40

# Added tuning values for smoother and more stable hazard selection
MIN_BOX_AREA_RATIO = 0.015
CENTER_ZONE_WEIGHT = 1.25
LOWER_SCREEN_BONUS = 1.15


def decode_base64_image(data_url: str):
    if "," not in data_url:
        return None

    encoded = data_url.split(",", 1)[1]
    image_bytes = base64.b64decode(encoded)
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return image


def make_result(hazard_detected, hazard_type, severity, message):
    return {
        "hazard_detected": hazard_detected,
        "hazard_type": hazard_type,
        "severity": severity,
        "message": message
    }


def center_priority(x1, y1, x2, y2, width, height):
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    horizontal_penalty = abs(cx - width / 2) / (width / 2)
    vertical_bonus = cy / height
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
            "Urgent. Possible step or curb ahead. Slow down and check your path."
        )

    if horizontal_count >= 3:
        return make_result(
            True,
            "step_or_curb",
            "warning",
            "Warning. Possible change in ground level ahead."
        )

    return None


def detect_low_visibility(gray_frame):
    brightness = gray_frame.mean()

    if brightness < 35:
        return make_result(
            True,
            "low_visibility",
            "warning",
            "Warning. Very low visibility ahead. Please slow down."
        )

    if brightness < 50:
        return make_result(
            True,
            "low_visibility",
            "caution",
            "Caution. Low visibility ahead."
        )

    return None


def _box_area_ratio(x1, y1, x2, y2, width, height):
    box_area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
    frame_area = float(width * height)
    if frame_area <= 0:
        return 0.0
    return box_area / frame_area


def _is_relevant_box(x1, y1, x2, y2, width, height):
    area_ratio = _box_area_ratio(x1, y1, x2, y2, width, height)
    return area_ratio >= MIN_BOX_AREA_RATIO


def _build_object_result(class_name):
    severity, message = TARGET_CLASSES[class_name]
    return {
        "hazard_detected": True,
        "hazard_type": class_name.replace(" ", "_"),
        "severity": severity,
        "message": message
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

    results = model.predict(source=frame, verbose=False, conf=CONFIDENCE_THRESHOLD)

    best_candidate = None
    best_score = -999

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

            # Ignore very tiny detections to reduce noisy warnings
            if not _is_relevant_box(x1, y1, x2, y2, width, height):
                continue

            area_ratio = _box_area_ratio(x1, y1, x2, y2, width, height)
            position_score = center_priority(x1, y1, x2, y2, width, height)

            # Prefer confident, central, lower-frame, and reasonably large obstacles
            score = (conf * 1.8) + position_score + (area_ratio * 2.2)

            if score > best_score:
                best_candidate = _build_object_result(class_name)
                best_score = score

    if best_candidate:
        return best_candidate

    if curb_result:
        return curb_result

    return make_result(False, None, None, "No immediate hazard detected.")


def detect_hazard_from_base64(data_url):
    frame = decode_base64_image(data_url)
    return detect_hazard_from_frame(frame)