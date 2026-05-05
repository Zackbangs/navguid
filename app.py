from flask import Flask, render_template, request, jsonify
from services.indoor_service import find_indoor_route  #
from services.outdoor_service import get_outdoor_route
from ai.object_detection import detect_hazard_from_base64
import os


"""
NavGuid Flask Application
-------------------------

This is the main backend entry point for the NavGuid prototype.

It handles:
1. Rendering the home page
2. Rendering the navigation page
3. Indoor route requests
4. Outdoor GPS route requests
5. Camera hazard detection requests
"""


# Create the Flask application instance.
app = Flask(__name__)


@app.route("/")
def home():
    """
    Render the voice-first landing page.
    """
    return render_template("index.html")


@app.route("/navigation")
def navigation_page():
    """
    Render the main navigation interface.
    """
    return render_template("navigation.html")


@app.route("/navigate", methods=["POST"])
def navigate():
    """
    Handle indoor navigation requests.
    """

    data = request.get_json(silent=True) or {}

    building = data.get("building")
    start = data.get("start")
    destination = data.get("destination")

    # ✅ Validation
    if not building or not start or not destination:
        return jsonify({
            "success": False,
            "message": "Building, start location, and destination are required."
        }), 400

    # ✅ NEW: use indoor_service instead of old navigation_service
    result = find_indoor_route(building, start, destination)

    return jsonify(result)


@app.route("/outdoor-navigate", methods=["POST"])
def outdoor_navigate():
    """
    Handle outdoor navigation requests.
    """

    data = request.get_json(silent=True) or {}

    start_lat = data.get("start_lat")
    start_lon = data.get("start_lon")
    destination_query = data.get("destination")

    if start_lat is None or start_lon is None or not destination_query:
        return jsonify({
            "success": False,
            "message": "Current location and destination are required."
        }), 400

    try:
        start_lat = float(start_lat)
        start_lon = float(start_lon)
    except (TypeError, ValueError):
        return jsonify({
            "success": False,
            "message": "Latitude and longitude must be valid numbers."
        }), 400

    result = get_outdoor_route(start_lat, start_lon, destination_query)
    return jsonify(result)


@app.route("/detect-hazard", methods=["POST"])
def detect_hazard():
    """
    Handle live camera hazard detection.
    """

    data = request.get_json(silent=True) or {}
    frame_data = data.get("frame")

    if not frame_data:
        return jsonify({
            "success": False,
            "message": "No camera frame received."
        }), 400

    try:
        result = detect_hazard_from_base64(frame_data)

        return jsonify({
            "success": True,
            "hazard_detected": result.get("hazard_detected", False),
            "hazard_type": result.get("hazard_type"),
            "severity": result.get("severity"),
            "message": result.get("message", "No immediate hazard detected."),
            "distance_label": result.get("distance_label"),
            "direction_label": result.get("direction_label"),
            "bbox": result.get("bbox"),
            "detections": result.get("detections", [])
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Hazard detection failed: {str(e)}"
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)