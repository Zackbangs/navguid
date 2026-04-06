import json
import os
from pathfinding import find_shortest_path, build_step_instructions


def get_building_file_path(building_name):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, "buildings", f"{building_name}.json")


def load_building_data(building_name):
    file_path = get_building_file_path(building_name)

    if not os.path.exists(file_path):
        return None

    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def get_route_steps(building_name, start, destination):
    building_data = load_building_data(building_name)

    if not building_data:
        return {
            "success": False,
            "message": f"Building file '{building_name}.json' not found."
        }

    path = find_shortest_path(building_data, start, destination)

    if not path:
        return {
            "success": False,
            "message": "No path found between the selected locations."
        }

    steps = build_step_instructions(building_data, path)

    return {
        "success": True,
        "path": path,
        "steps": steps
    }