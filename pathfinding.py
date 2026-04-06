import heapq


def build_graph(building_data):
    graph = {}

    for node in building_data["nodes"]:
        graph[node["id"]] = []

    for edge in building_data["edges"]:
        start = edge["from"]
        end = edge["to"]
        distance = edge["distance"]

        graph[start].append((end, distance))
        graph[end].append((start, distance))

    return graph


def find_shortest_path(building_data, start, destination):
    graph = build_graph(building_data)

    if start not in graph or destination not in graph:
        return None

    priority_queue = [(0, start, [])]
    visited = set()

    while priority_queue:
        current_distance, current_node, path = heapq.heappop(priority_queue)

        if current_node in visited:
            continue

        visited.add(current_node)
        path = path + [current_node]

        if current_node == destination:
            return path

        for neighbor, weight in graph[current_node]:
            if neighbor not in visited:
                heapq.heappush(
                    priority_queue,
                    (current_distance + weight, neighbor, path)
                )

    return None


def get_node_name(building_data, node_id):
    for node in building_data["nodes"]:
        if node["id"] == node_id:
            return node["name"]
    return node_id


def build_step_instructions(building_data, path):
    steps = []

    for i in range(len(path) - 1):
        current_node = path[i]
        next_node = path[i + 1]

        current_name = get_node_name(building_data, current_node)
        next_name = get_node_name(building_data, next_node)

        steps.append(f"Move from {current_name} to {next_name}.")

    steps.append("You have arrived at your destination.")
    return steps