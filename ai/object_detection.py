import base64
import cv2
import numpy as np


def decode_base64_image(data_url):
    if "," not in data_url:
        return None

    encoded = data_url.split(",", 1)[1]
    image_bytes = base64.b64decode(encoded)
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return image


def detect_hazard_from_frame(frame):
    if frame is None:
        return {
            "hazard_detected": False,
            "hazard_type": None,
            "message": "No valid frame available."
        }

    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 1. Low visibility
    brightness = gray.mean()
    if brightness < 45:
        return {
            "hazard_detected": True,
            "hazard_type": "low_visibility",
            "message": "Low visibility ahead. Please slow down and use caution."
        }

    # 2. Front walking zone analysis
    front_zone = gray[int(height * 0.45):height, int(width * 0.2):int(width * 0.8)]

    # Edge density
    edges = cv2.Canny(front_zone, 70, 150)
    edge_ratio = np.count_nonzero(edges) / edges.size

    # 3. Strong lower edge line can simulate curb / step
    lower_strip = gray[int(height * 0.75):int(height * 0.9), int(width * 0.15):int(width * 0.85)]
    lower_edges = cv2.Canny(lower_strip, 70, 150)
    lines = cv2.HoughLinesP(lower_edges, 1, np.pi / 180, threshold=70,
                            minLineLength=80, maxLineGap=20)

    if lines is not None and len(lines) >= 4:
        return {
            "hazard_detected": True,
            "hazard_type": "step_or_curb",
            "message": "Possible step or curb ahead. Slow down and check your path."
        }

    # 4. Dense clutter in front zone
    if edge_ratio > 0.14:
        return {
            "hazard_detected": True,
            "hazard_type": "possible_obstacle",
            "message": "Possible obstacle ahead. Stop and check your path."
        }

    # 5. Strong contrast cluster in center may suggest blocked path
    center_zone = gray[int(height * 0.4):int(height * 0.85), int(width * 0.35):int(width * 0.65)]
    _, thresh = cv2.threshold(center_zone, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    white_ratio = np.count_nonzero(thresh) / thresh.size

    if white_ratio < 0.25 or white_ratio > 0.75:
        return {
            "hazard_detected": True,
            "hazard_type": "path_unclear",
            "message": "Path ahead may be unclear. Please move carefully."
        }

    return {
        "hazard_detected": False,
        "hazard_type": None,
        "message": "No immediate hazard detected."
    }


def detect_hazard_from_base64(data_url):
    frame = decode_base64_image(data_url)
    return detect_hazard_from_frame(frame)