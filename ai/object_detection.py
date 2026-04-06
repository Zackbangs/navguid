import base64
import cv2
import numpy as np


def decode_base64_image(data_url):
    """
    Convert base64 image from browser canvas into an OpenCV image.
    """
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


def detect_step_or_curb(gray_frame, height, width):
    """
    Looks for strong horizontal edges in the lower walking area.
    """
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


def detect_blocked_path(gray_frame, height, width):
    """
    Checks the main walking corridor in the center lower area.
    """
    front_zone = gray_frame[
        int(height * 0.45):height,
        int(width * 0.22):int(width * 0.78)
    ]

    edges = cv2.Canny(front_zone, 70, 160)
    edge_ratio = np.count_nonzero(edges) / edges.size

    blurred = cv2.GaussianBlur(front_zone, (5, 5), 0)
    _, thresh = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    white_ratio = np.count_nonzero(thresh) / thresh.size

    # Very cluttered front path
    if edge_ratio > 0.18:
        return make_result(
            True,
            "blocked_path",
            "urgent",
            "Urgent. Path ahead may be blocked. Stop and check your path."
        )

    if edge_ratio > 0.13:
        return make_result(
            True,
            "possible_obstacle",
            "warning",
            "Warning. Possible obstacle ahead."
        )

    # Unclear center view
    if white_ratio < 0.18 or white_ratio > 0.82:
        return make_result(
            True,
            "path_unclear",
            "caution",
            "Caution. Path ahead may be unclear."
        )

    return None


def detect_side_obstacle(gray_frame, height, width):
    """
    Compares left and right path complexity to suggest a partial blockage.
    """
    lower_zone = gray_frame[
        int(height * 0.50):height,
        int(width * 0.15):int(width * 0.85)
    ]

    zone_width = lower_zone.shape[1]
    left_zone = lower_zone[:, :zone_width // 2]
    right_zone = lower_zone[:, zone_width // 2:]

    left_edges = cv2.Canny(left_zone, 70, 150)
    right_edges = cv2.Canny(right_zone, 70, 150)

    left_ratio = np.count_nonzero(left_edges) / left_edges.size
    right_ratio = np.count_nonzero(right_edges) / right_edges.size

    difference = abs(left_ratio - right_ratio)

    if difference > 0.08:
        if left_ratio > right_ratio:
            return make_result(
                True,
                "left_side_obstacle",
                "caution",
                "Caution. Path may be more blocked on your left side."
            )
        return make_result(
            True,
            "right_side_obstacle",
            "caution",
            "Caution. Path may be more blocked on your right side."
        )

    return None


def detect_hazard_from_frame(frame):
    """
    Smarter rule-based hazard detection for NavGuid.

    Current hazard types:
    - low_visibility
    - step_or_curb
    - blocked_path
    - possible_obstacle
    - path_unclear
    - left_side_obstacle
    - right_side_obstacle
    """
    if frame is None:
        return make_result(
            False,
            None,
            None,
            "No valid frame available."
        )

    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 1. Low visibility
    result = detect_low_visibility(gray)
    if result:
        return result

    # 2. Ground-level hazards first
    result = detect_step_or_curb(gray, height, width)
    if result:
        return result

    # 3. Blocked path / front obstacle
    result = detect_blocked_path(gray, height, width)
    if result:
        return result

    # 4. Left/right side awareness
    result = detect_side_obstacle(gray, height, width)
    if result:
        return result

    return make_result(
        False,
        None,
        None,
        "No immediate hazard detected."
    )


def detect_hazard_from_base64(data_url):
    frame = decode_base64_image(data_url)
    return detect_hazard_from_frame(frame)