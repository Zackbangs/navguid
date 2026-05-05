import requests


"""
NavGuid Outdoor Navigation Service
----------------------------------

This file handles the outdoor navigation backend logic.

It is responsible for:
1. Searching a destination using Nominatim / OpenStreetMap
2. Requesting a walking route from OSRM
3. Converting OSRM route steps into simple spoken instructions
4. Returning route steps, step coordinates, destination coordinates, and route geometry
"""


def search_destination(destination_query):
    """
    Search for a destination using Nominatim.

    Example:
        "Marion Shopping Centre"

    Returns:
        A dictionary containing the destination name, latitude, and longitude.
        Returns None if no result is found.
    """

    # Nominatim search endpoint used to convert a destination name into coordinates.
    url = "https://nominatim.openstreetmap.org/search"

    # Search parameters sent to Nominatim.
    params = {
        "q": destination_query,
        "format": "jsonv2",
        "limit": 1
    }

    # User-Agent is required by Nominatim usage policy.
    headers = {
        "User-Agent": "NavGuid/1.0"
    }

    # Sends the destination search request and raises an error if the request fails.
    response = requests.get(url, params=params, headers=headers, timeout=15)
    response.raise_for_status()

    results = response.json()

    # Returns None when no matching destination is found.
    if not results:
        return None

    # Returns the first matching destination in a simple backend-friendly format.
    return {
        "name": results[0].get("display_name", destination_query),
        "lat": float(results[0]["lat"]),
        "lon": float(results[0]["lon"])
    }


def _clean_street_name(street_name):
    """
    Clean street names before using them in spoken instructions.
    """

    # Removes extra whitespace and safely handles empty street names.
    return (street_name or "").strip()


def _round_distance(step):
    """
    Round OSRM step distance to a clean whole number.

    This makes spoken guidance easier to understand.
    """

    # Ensures distance is never negative and is easy to speak aloud.
    return max(0, round(step.get("distance", 0)))


def _build_spoken_instruction(step):
    """
    Convert a single OSRM step into a natural spoken instruction.

    OSRM returns technical route data such as:
    - maneuver type
    - turn modifier
    - street name
    - distance

    This function converts that into user-friendly speech such as:
    - "Keep going straight"
    - "Turn left in about 20 metres"
    - "You have arrived at your destination"
    """

    # Extracts the main OSRM navigation details for this route step.
    maneuver = step.get("maneuver", {})
    maneuver_type = maneuver.get("type", "continue")
    modifier = (maneuver.get("modifier", "") or "").replace("_", " ").strip()
    street_name = _clean_street_name(step.get("name", ""))
    distance = _round_distance(step)

    # Starting instruction for the route.
    if maneuver_type == "depart":
        if street_name:
            return f"Start walking on {street_name}."
        return "Start walking now."

    # Instruction for continuing straight or along the same street.
    if maneuver_type == "continue":
        if street_name:
            return f"Keep going on {street_name} for about {distance} metres."
        return f"Keep going straight for about {distance} metres."

    # Instruction for left/right turns.
    if maneuver_type == "turn":
        if modifier and street_name:
            return f"In about {distance} metres, turn {modifier} onto {street_name}."
        if modifier:
            return f"In about {distance} metres, turn {modifier}."
        return f"Turn ahead in about {distance} metres."

    # Instruction for forked paths.
    if maneuver_type == "fork":
        if modifier:
            return f"In about {distance} metres, keep {modifier}."
        return f"Keep ahead in about {distance} metres."

    # Instruction for merging into another path or road.
    if maneuver_type == "merge":
        if modifier:
            return f"In about {distance} metres, merge {modifier}."
        return f"Merge ahead in about {distance} metres."

    # Instruction for roundabouts, including exit number when available.
    if maneuver_type == "roundabout":
        exit_number = maneuver.get("exit")
        if exit_number:
            return f"In about {distance} metres, approach the roundabout and take exit {exit_number}."
        return f"In about {distance} metres, approach the roundabout."

    # Final arrival instruction.
    if maneuver_type == "arrive":
        return "You have arrived at your destination."

    # Fallback for uncommon OSRM maneuver types.
    if street_name:
        return f"{maneuver_type.replace('_', ' ').title()} on {street_name} for about {distance} metres."

    return f"{maneuver_type.replace('_', ' ').title()} for about {distance} metres."


def build_osrm_steps_and_data(route_data):
    """
    Extract spoken route steps and coordinate-based step data from OSRM response.

    The frontend uses:
    - steps_output for readable route instructions
    - step_data for live GPS step tracking

    step_data includes each maneuver location so the frontend can detect:
    - when the user is near a step
    - when to say "turn now"
    - when to move to the next instruction
    """

    # Stores readable voice instructions for the user.
    steps_output = []

    # Stores detailed step information for live GPS tracking.
    step_data = []

    # Gets route list from OSRM response.
    routes = route_data.get("routes", [])
    if not routes:
        return steps_output, step_data

    # Gets walking route legs from the first route.
    legs = routes[0].get("legs", [])
    if not legs:
        return steps_output, step_data

    # Loops through each OSRM step and converts it into frontend-ready data.
    for leg in legs:
        for step in leg.get("steps", []):
            maneuver = step.get("maneuver", {})
            location = maneuver.get("location", [])

            spoken_instruction = _build_spoken_instruction(step)

            # OSRM location format is [longitude, latitude].
            lat = float(location[1]) if len(location) == 2 else None
            lon = float(location[0]) if len(location) == 2 else None

            # Adds the readable instruction to the step list.
            steps_output.append(spoken_instruction)

            # Adds detailed step data used by the frontend for live tracking.
            step_data.append({
                "instruction": spoken_instruction,
                "lat": lat,
                "lon": lon,
                "maneuver_type": maneuver.get("type", "continue"),
                "modifier": maneuver.get("modifier", ""),
                "street_name": _clean_street_name(step.get("name", "")),
                "distance_m": _round_distance(step),
                "duration_s": round(step.get("duration", 0)),

                # Bearings are kept for future direction/orientation upgrades.
                "bearing_before": maneuver.get("bearing_before"),
                "bearing_after": maneuver.get("bearing_after")
            })

    # Add a final arrival step so the frontend has a clear endpoint instruction.
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
    """
    Convert OSRM route geometry into a frontend-friendly list of coordinates.

    The geometry can be used later for:
    - drawing the route line
    - checking if the user has moved away from the route
    - future rerouting improvements
    """

    # Gets the GeoJSON route geometry from OSRM.
    geometry = route.get("geometry", {})
    coordinates = geometry.get("coordinates", [])

    output = []

    # Converts every OSRM coordinate from [longitude, latitude] into {lat, lon}.
    for coord in coordinates:
        if len(coord) == 2:
            # OSRM coordinate format is [longitude, latitude].
            output.append({
                "lat": float(coord[1]),
                "lon": float(coord[0])
            })

    return output


def get_outdoor_route(start_lat, start_lon, destination_query):
    """
    Main outdoor navigation function.

    This function:
    1. Finds the destination coordinates
    2. Requests a walking route from OSRM
    3. Builds spoken step instructions
    4. Returns route data to the Flask route /outdoor-navigate

    The returned structure is kept stable so the frontend navigation page
    does not need to change.
    """

    try:
        # Step 1: Search for destination coordinates.
        destination = search_destination(destination_query)

        if not destination:
            return {
                "success": False,
                "message": "Destination not found."
            }

        dest_lat = destination["lat"]
        dest_lon = destination["lon"]

        # Step 2: Build OSRM walking route request.
        # OSRM requires coordinates in longitude,latitude order.
        osrm_url = (
            f"https://router.project-osrm.org/route/v1/foot/"
            f"{start_lon},{start_lat};{dest_lon},{dest_lat}"
        )

        # Requests full route geometry and turn-by-turn steps.
        params = {
            "overview": "full",
            "steps": "true",
            "geometries": "geojson"
        }

        # Sends the walking route request to OSRM.
        response = requests.get(osrm_url, params=params, timeout=15)
        response.raise_for_status()

        route_data = response.json()

        # Handles cases where OSRM cannot generate a route.
        if route_data.get("code") != "Ok" or not route_data.get("routes"):
            return {
                "success": False,
                "message": "Could not generate walking route."
            }

        # Step 3: Extract the first route.
        route = route_data["routes"][0]
        distance_m = round(route.get("distance", 0))
        duration_s = round(route.get("duration", 0))

        # Step 4: Build frontend-ready navigation data.
        steps, step_data = build_osrm_steps_and_data(route_data)
        route_geometry = build_route_geometry(route)

        # Returns complete outdoor navigation information to the frontend.
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

    # Handles failed requests to Nominatim or OSRM.
    except requests.RequestException:
        return {
            "success": False,
            "message": "Network error while contacting map services."
        }

    # Handles unexpected backend errors safely.
    except Exception as e:
        return {
            "success": False,
            "message": f"Unexpected error: {str(e)}"
        }