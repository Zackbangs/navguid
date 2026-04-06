from flask import Flask, render_template, request, jsonify
from services.navigation_service import get_route_steps
from services.outdoor_service import get_outdoor_route
from ai.object_detection import detect_hazard_from_base64

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/navigation")
def navigation_page():
    return render_template("navigation.html")


@app.route("/navigate", methods=["POST"])
def navigate():
    data = request.get_json()

    building = data.get("building")
    start = data.get("start")
    destination = data.get("destination")

    if not building or not start or not destination:
        return jsonify({
            "success": False,
            "message": "Building, start location, and destination are required."
        }), 400

    result = get_route_steps(building, start, destination)
    return jsonify(result)


@app.route("/outdoor-navigate", methods=["POST"])
def outdoor_navigate():
    data = request.get_json()

    start_lat = data.get("start_lat")
    start_lon = data.get("start_lon")
    destination_query = data.get("destination")

    if start_lat is None or start_lon is None or not destination_query:
        return jsonify({
            "success": False,
            "message": "Current location and destination are required."
        }), 400

    result = get_outdoor_route(start_lat, start_lon, destination_query)
    return jsonify(result)


@app.route("/detect-hazard", methods=["POST"])
def detect_hazard():
    data = request.get_json()
    frame_data = data.get("frame")

    if not frame_data:
        return jsonify({
            "success": False,
            "message": "No camera frame received."
        }), 400

    result = detect_hazard_from_base64(frame_data)
    return jsonify({
        "success": True,
        "hazard_detected": result["hazard_detected"],
        "hazard_type": result["hazard_type"],
        "severity": result["severity"],
        "message": result["message"]
    })


import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)