from collections import deque

BUILDINGS = {
    "torrens": {
        "entrance": ["reception"],
        "reception": ["entrance", "hallway"],
        "hallway": ["reception", "classroom", "library", "toilet"],
        "classroom": ["hallway"],
        "library": ["hallway"],
        "toilet": ["hallway"]
    },
    "shopping": {
        "entrance": ["information desk"],
        "information desk": ["entrance", "main hallway"],
        "main hallway": ["information desk", "food court", "shops", "toilet"],
        "food court": ["main hallway"],
        "shops": ["main hallway"],
        "toilet": ["main hallway"]
    }
}


def find_indoor_route(building, start, destination):
    building = (building or "").strip().lower()
    start = (start or "").strip().lower()
    destination = (destination or "").strip().lower()

    if building not in BUILDINGS:
        return {
            "success": False,
            "message": "Building not found.",
            "steps": []
        }

    graph = BUILDINGS[building]

    if start not in graph:
        return {
            "success": False,
            "message": f"Start location '{start}' was not found.",
            "steps": []
        }

    if destination not in graph:
        return {
            "success": False,
            "message": f"Destination '{destination}' was not found.",
            "steps": []
        }

    queue = deque([[start]])
    visited = set()

    while queue:
        path = queue.popleft()
        current = path[-1]

        if current == destination:
            return {
                "success": True,
                "message": "Indoor route generated successfully.",
                "steps": build_indoor_steps(path),
                "path": path
            }

        if current not in visited:
            visited.add(current)

            for neighbour in graph[current]:
                new_path = list(path)
                new_path.append(neighbour)
                queue.append(new_path)

    return {
        "success": False,
        "message": "No indoor route found.",
        "steps": []
    }


def build_indoor_steps(path):
    steps = []

    if not path or len(path) < 2:
        return ["You are already at your destination."]

    steps.append(f"Start at {path[0]}.")

    for index in range(1, len(path)):
        steps.append(f"Move from {path[index - 1]} to {path[index]}.")

    steps.append("You have arrived at your indoor destination.")

    return steps