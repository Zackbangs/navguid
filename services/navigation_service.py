import json
import os
from pathfinding import find_shortest_path, build_step_instructions


def get_building_file_path(building_name):
    # Gets the main project directory and builds the full path to the selected building JSON file.
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, "buildings", f"{building_name}.json")


def load_building_data(building_name):
    # Loads the building map data from the JSON file.
    file_path = get_building_file_path(building_name)

    # Returns None if the building file does not exist.
    if not os.path.exists(file_path):
        return None

    # Opens and reads the JSON file using UTF-8 encoding.
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def get_route_steps(building_name, start, destination):
    # Main function used to generate indoor navigation route steps.
    building_data = load_building_data(building_name)

    # Handles missing building data safely.
    if not building_data:
        return {
            "success": False,
            "message": f"Building file '{building_name}.json' not found."
        }

    # Finds the shortest path between the selected start and destination nodes.
    path = find_shortest_path(building_data, start, destination)

    # Handles cases where no valid route exists between the selected locations.
    if not path:
        return {
            "success": False,
            "message": "No path found between the selected locations."
        }

    # Converts the path into simple step-by-step navigation instructions.
    steps = build_step_instructions(building_data, path)

    # Returns the successful route result to the frontend.
    return {
        "success": True,
        "path": path,
        "steps": steps
    }