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


def build_osrm_steps(route_data):
    steps_output = []

    routes = route_data.get("routes", [])
    if not routes:
        return steps_output

    legs = routes[0].get("legs", [])
    if not legs:
        return steps_output

    for leg in legs:
        for step in leg.get("steps", []):
            maneuver = step.get("maneuver", {})
            maneuver_type = maneuver.get("type", "continue")
            modifier = maneuver.get("modifier", "")
            name = step.get("name", "")
            distance = round(step.get("distance", 0))

            instruction = f"{maneuver_type.replace('_', ' ').title()}"

            if modifier:
                instruction += f" {modifier}"

            if name:
                instruction += f" onto {name}"

            instruction += f" for {distance} metres."

            steps_output.append(instruction)

    if steps_output:
        steps_output.append("You have arrived at your destination.")

    return steps_output


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
            "overview": "false",
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

        steps = build_osrm_steps(route_data)

        return {
            "success": True,
            "destination_name": destination["name"],
            "distance_m": distance_m,
            "duration_min": round(duration_s / 60),
            "steps": steps
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