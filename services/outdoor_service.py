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


def _clean_street_name(street_name):
    return (street_name or "").strip()


def _round_distance(step):
    return max(0, round(step.get("distance", 0)))


def _build_spoken_instruction(step):
    maneuver = step.get("maneuver", {})
    maneuver_type = maneuver.get("type", "continue")
    modifier = (maneuver.get("modifier", "") or "").replace("_", " ").strip()
    street_name = _clean_street_name(step.get("name", ""))
    distance = _round_distance(step)

    if maneuver_type == "depart":
        if street_name:
            return f"Start walking on {street_name}."
        return "Start walking now."

    if maneuver_type == "continue":
        if street_name:
            return f"Keep going on {street_name} for about {distance} metres."
        return f"Keep going straight for about {distance} metres."

    if maneuver_type == "turn":
        if modifier and street_name:
            return f"In about {distance} metres, turn {modifier} onto {street_name}."
        if modifier:
            return f"In about {distance} metres, turn {modifier}."
        return f"Turn ahead in about {distance} metres."

    if maneuver_type == "fork":
        if modifier:
            return f"In about {distance} metres, keep {modifier}."
        return f"Keep ahead in about {distance} metres."

    if maneuver_type == "merge":
        if modifier:
            return f"In about {distance} metres, merge {modifier}."
        return f"Merge ahead in about {distance} metres."

    if maneuver_type == "roundabout":
        exit_number = maneuver.get("exit")
        if exit_number:
            return f"In about {distance} metres, approach the roundabout and take exit {exit_number}."
        return f"In about {distance} metres, approach the roundabout."

    if maneuver_type == "arrive":
        return "You have arrived at your destination."

    if street_name:
        return f"{maneuver_type.replace('_', ' ').title()} on {street_name} for about {distance} metres."

    return f"{maneuver_type.replace('_', ' ').title()} for about {distance} metres."


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

            lat = float(location[1]) if len(location) == 2 else None
            lon = float(location[0]) if len(location) == 2 else None

            steps_output.append(spoken_instruction)

            step_data.append({
                "instruction": spoken_instruction,
                "lat": lat,
                "lon": lon,
                "maneuver_type": maneuver.get("type", "continue"),
                "modifier": maneuver.get("modifier", ""),
                "street_name": _clean_street_name(step.get("name", "")),
                "distance_m": _round_distance(step),
                "duration_s": round(step.get("duration", 0)),
                "bearing_before": maneuver.get("bearing_before"),
                "bearing_after": maneuver.get("bearing_after")
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
            "distance_m": 0,
            "duration_s": 0,
            "bearing_before": None,
            "bearing_after": None
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