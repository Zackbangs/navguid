from flask import Flask, render_template, request, jsonify
from services.navigation_service import get_route_steps
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

    This page introduces NavGuid and allows the user to start the
    voice assistant before moving to the navigation screen.
    """

    return render_template("index.html")


@app.route("/navigation")
def navigation_page():
    """
    Render the main navigation interface.

    This page contains:
    - voice assistant logic
    - indoor navigation fallback controls
    - outdoor GPS navigation
    - camera feed
    - live hazard detection overlay
    """

    return render_template("navigation.html")


@app.route("/navigate", methods=["POST"])
def navigate():
    """
    Handle indoor navigation requests.

    Expected JSON body:
    {
        "building": "torrens",
        "start": "entrance",
        "destination": "room_g12"
    }

    The actual indoor route logic is handled by services/navigation_service.py.
    """

    data = request.get_json(silent=True) or {}

    building = data.get("building")
    start = data.get("start")
    destination = data.get("destination")

    # Validate required indoor navigation inputs.
    if not building or not start or not destination:
        return jsonify({
            "success": False,
            "message": "Building, start location, and destination are required."
        }), 400

    result = get_route_steps(building, start, destination)
    return jsonify(result)


@app.route("/outdoor-navigate", methods=["POST"])
def outdoor_navigate():
    """
    Handle outdoor navigation requests.

    Expected JSON body:
    {
        "start_lat": -34.9285,
        "start_lon": 138.6007,
        "destination": "Marion Shopping Centre"
    }

    This endpoint sends the user's current GPS position and destination query
    to the outdoor navigation service, which uses Nominatim and OSRM.
    """

    data = request.get_json(silent=True) or {}

    start_lat = data.get("start_lat")
    start_lon = data.get("start_lon")
    destination_query = data.get("destination")

    # Validate that location and destination were provided.
    if start_lat is None or start_lon is None or not destination_query:
        return jsonify({
            "success": False,
            "message": "Current location and destination are required."
        }), 400

    # Convert coordinates to float so route services receive valid numeric values.
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

    Expected JSON body:
    {
        "frame": "data:image/jpeg;base64,..."
    }

    The frontend captures a camera frame, sends it here, and this endpoint
    passes the frame to the YOLO-based object detection module.

    The returned response is kept frontend-friendly so navigation.html can:
    - show hazard status
    - draw bounding boxes
    - speak hazard warnings
    """

    data = request.get_json(silent=True) or {}
    frame_data = data.get("frame")

    # Validate that a camera frame was received.
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
    """
    Local and Render-compatible startup.

    Render provides the PORT environment variable automatically.
    If PORT is not available, the app uses 10000 for local testing.
    """

    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)