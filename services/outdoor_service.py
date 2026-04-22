import requests


def search_destination(destination_query):
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": destination_query,
        "format": "jsonv2",
        "limit": 1
    }
    headers = {
        "User-Agent": "NavGuid/1.0"
    }

    response = requests.get(url, params=params, headers=headers, timeout=15)
    response.raise_for_status()

    results = response.json()

    if not results:
        return None

    return {
        "name": results[0].get("display_name", destination_query),
        "lat": float(results[0]["lat"]),
        "lon": float(results[0]["lon"])
    }


def _build_spoken_instruction(step):
    maneuver = step.get("maneuver", {})
    maneuver_type = maneuver.get("type", "continue")
    modifier = maneuver.get("modifier", "")
    street_name = step.get("name", "").strip()
    distance = round(step.get("distance", 0))

    if maneuver_type == "depart":
        if street_name:
            return f"Start walking on {street_name} for {distance} metres."
        return f"Start walking straight for {distance} metres."

    if maneuver_type == "continue":
        if street_name:
            return f"Continue straight on {street_name} for {distance} metres."
        return f"Continue straight for {distance} metres."

    if maneuver_type == "turn":
        if modifier and street_name:
            return f"Turn {modifier} onto {street_name} in {distance} metres."
        if modifier:
            return f"Turn {modifier} in {distance} metres."
        return f"Turn ahead in {distance} metres."

    if maneuver_type == "fork":
        if modifier:
            return f"Keep {modifier} in {distance} metres."
        return f"Keep ahead in {distance} metres."

    if maneuver_type == "merge":
        if modifier:
            return f"Merge {modifier} in {distance} metres."
        return f"Merge ahead in {distance} metres."

    if maneuver_type == "roundabout":
        return f"Approach the roundabout in {distance} metres."

    if maneuver_type == "arrive":
        return "You have arrived at your destination."

    if street_name:
        return f"{maneuver_type.replace('_', ' ').title()} on {street_name} for {distance} metres."

    return f"{maneuver_type.replace('_', ' ').title()} for {distance} metres."


def build_osrm_steps_and_data(route_data):
    steps_output = []
    step_data = []

    routes = route_data.get("routes", [])
    if not routes:
        return steps_output, step_data

    legs = routes[0].get("legs", [])
    if not legs:
        return steps_output, step_data

    for leg in legs:
        for step in leg.get("steps", []):
            maneuver = step.get("maneuver", {})
            location = maneuver.get("location", [])
            spoken_instruction = _build_spoken_instruction(step)

            steps_output.append(spoken_instruction)

            step_data.append({
                "instruction": spoken_instruction,
                "lat": float(location[1]) if len(location) == 2 else None,
                "lon": float(location[0]) if len(location) == 2 else None,
                "maneuver_type": maneuver.get("type", "continue"),
                "modifier": maneuver.get("modifier", ""),
                "street_name": step.get("name", "").strip(),
                "distance_m": round(step.get("distance", 0))
            })

    if steps_output:
        steps_output.append("You have arrived at your destination.")
        step_data.append({
            "instruction": "You have arrived at your destination.",
            "lat": None,
            "lon": None,
            "maneuver_type": "arrive",
            "modifier": "",
            "street_name": "",
            "distance_m": 0
        })

    return steps_output, step_data


def build_route_geometry(route):
    geometry = route.get("geometry", {})
    coordinates = geometry.get("coordinates", [])

    output = []
    for coord in coordinates:
        if len(coord) == 2:
            output.append({
                "lat": float(coord[1]),
                "lon": float(coord[0])
            })

    return output


def get_outdoor_route(start_lat, start_lon, destination_query):
    try:
        destination = search_destination(destination_query)

        if not destination:
            return {
                "success": False,
                "message": "Destination not found."
            }

        dest_lat = destination["lat"]
        dest_lon = destination["lon"]

        osrm_url = (
            f"https://router.project-osrm.org/route/v1/foot/"
            f"{start_lon},{start_lat};{dest_lon},{dest_lat}"
        )

        params = {
            "overview": "full",
            "steps": "true",
            "geometries": "geojson"
        }

        response = requests.get(osrm_url, params=params, timeout=15)
        response.raise_for_status()

        route_data = response.json()

        if route_data.get("code") != "Ok" or not route_data.get("routes"):
            return {
                "success": False,
                "message": "Could not generate walking route."
            }

        route = route_data["routes"][0]
        distance_m = round(route.get("distance", 0))
        duration_s = round(route.get("duration", 0))

        steps, step_data = build_osrm_steps_and_data(route_data)
        route_geometry = build_route_geometry(route)

        return {
            "success": True,
            "destination_name": destination["name"],
            "destination": {
                "lat": dest_lat,
                "lon": dest_lon
            },
            "distance_m": distance_m,
            "duration_min": round(duration_s / 60),
            "steps": steps,
            "step_data": step_data,
            "route_geometry": route_geometry
        }

    except requests.RequestException:
        return {
            "success": False,
            "message": "Network error while contacting map services."
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Unexpected error: {str(e)}"
        }