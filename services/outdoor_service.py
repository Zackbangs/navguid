import math
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


def normalise_modifier(modifier):
    if not modifier:
        return ""
    return modifier.replace("_", " ").strip().lower()


def build_step_instruction(maneuver_type, modifier, street_name, distance):
    maneuver_type = (maneuver_type or "continue").replace("_", " ").strip().lower()
    modifier = normalise_modifier(modifier)
    distance = max(0, round(distance or 0))

    if maneuver_type == "depart":
        if street_name:
            return f"Start and continue on {street_name} for {distance} metres."
        return f"Start and continue straight for {distance} metres."

    if maneuver_type == "arrive":
        return "You have arrived at your destination."

    if maneuver_type == "turn":
        if modifier:
            if street_name:
                return f"Turn {modifier} onto {street_name} for {distance} metres."
            return f"Turn {modifier} for {distance} metres."

    if maneuver_type == "continue":
        if modifier in ["left", "right", "slight left", "slight right"]:
            if street_name:
                return f"Continue {modifier} onto {street_name} for {distance} metres."
            return f"Continue {modifier} for {distance} metres."
        if street_name:
            return f"Continue straight on {street_name} for {distance} metres."
        return f"Continue straight for {distance} metres."

    if maneuver_type == "new name":
        if street_name:
            return f"Continue onto {street_name} for {distance} metres."
        return f"Continue straight for {distance} metres."

    if maneuver_type == "merge":
        if modifier:
            if street_name:
                return f"Merge {modifier} onto {street_name} for {distance} metres."
            return f"Merge {modifier} for {distance} metres."
        if street_name:
            return f"Merge onto {street_name} for {distance} metres."
        return f"Merge ahead for {distance} metres."

    if maneuver_type == "fork":
        if modifier:
            if street_name:
                return f"Keep {modifier} onto {street_name} for {distance} metres."
            return f"Keep {modifier} for {distance} metres."
        return f"Keep ahead for {distance} metres."

    if maneuver_type == "roundabout":
        if modifier:
            if street_name:
                return f"At the roundabout, take the {modifier} exit onto {street_name} for {distance} metres."
            return f"At the roundabout, take the {modifier} exit for {distance} metres."
        return f"At the roundabout, continue for {distance} metres."

    if maneuver_type == "end of road":
        if modifier:
            if street_name:
                return f"At the end of the road, turn {modifier} onto {street_name} for {distance} metres."
            return f"At the end of the road, turn {modifier} for {distance} metres."

    if maneuver_type == "use lane":
        if modifier:
            return f"Use the lane to go {modifier} for {distance} metres."
        return f"Use the indicated lane for {distance} metres."

    if street_name:
        return f"{maneuver_type.title()} onto {street_name} for {distance} metres."

    return f"{maneuver_type.title()} for {distance} metres."


def build_osrm_steps(route_data):
    steps_output = []
    step_objects = []

    routes = route_data.get("routes", [])
    if not routes:
        return steps_output, step_objects

    legs = routes[0].get("legs", [])
    if not legs:
        return steps_output, step_objects

    for leg in legs:
        for index, step in enumerate(leg.get("steps", [])):
            maneuver = step.get("maneuver", {})
            maneuver_type = maneuver.get("type", "continue")
            modifier = maneuver.get("modifier", "")
            name = step.get("name", "").strip()
            distance = round(step.get("distance", 0))
            duration = round(step.get("duration", 0))

            maneuver_location = maneuver.get("location", [])
            step_lon = None
            step_lat = None
            if isinstance(maneuver_location, list) and len(maneuver_location) == 2:
                step_lon = float(maneuver_location[0])
                step_lat = float(maneuver_location[1])

            instruction = build_step_instruction(
                maneuver_type=maneuver_type,
                modifier=modifier,
                street_name=name,
                distance=distance
            )

            steps_output.append(instruction)
            step_objects.append({
                "index": index,
                "instruction": instruction,
                "distance_m": distance,
                "duration_s": duration,
                "street_name": name,
                "maneuver_type": maneuver_type,
                "modifier": modifier,
                "lat": step_lat,
                "lon": step_lon,
                "bearing_before": maneuver.get("bearing_before"),
                "bearing_after": maneuver.get("bearing_after")
            })

    if steps_output:
        steps_output.append("You have arrived at your destination.")

    return steps_output, step_objects


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

        steps, step_objects = build_osrm_steps(route_data)

        geometry = route.get("geometry", {}).get("coordinates", [])
        route_geometry = [
            {"lat": float(point[1]), "lon": float(point[0])}
            for point in geometry
            if isinstance(point, list) and len(point) == 2
        ]

        return {
            "success": True,
            "destination_name": destination["name"],
            "distance_m": distance_m,
            "duration_min": round(duration_s / 60),
            "steps": steps,
            "step_data": step_objects,
            "destination": {
                "lat": dest_lat,
                "lon": dest_lon
            },
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