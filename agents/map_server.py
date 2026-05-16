"""
Map Server — serves the SLAM map for the AutoPassing system.
=============================================================
In the real system, the patrolling program builds the map and
uploads it here via POST /map. The Navigate node pulls it via
GET /map.

For now, the server starts with a dummy map pre-loaded.

Usage:
    python map_server.py

The server runs at http://127.0.0.1:8100
    GET  /map          → returns the current map as JSON
    POST /map          → replaces the map (JSON body)
    GET  /map/status   → returns map metadata (version, last updated)
"""

from flask import Flask, jsonify, request
from datetime import datetime

app = Flask(__name__)

# ============================================================
# DUMMY MAP DATA
# ============================================================
# A small city grid with streets, intersections, and landmarks.
# The patrolling program would build and upload a real one.

city_map = {
    "version": 1,
    "last_updated": datetime.now().isoformat(),
    "city": "SimCity",

    "streets": [
        {"name": "Main Street",      "direction": "east-west",  "lanes": 2, "speed_limit_ms": 15.0, "length_m": 800},
        {"name": "University Avenue", "direction": "east-west",  "lanes": 2, "speed_limit_ms": 13.0, "length_m": 600},
        {"name": "Highway 5",         "direction": "north-south","lanes": 4, "speed_limit_ms": 30.0, "length_m": 2000},
        {"name": "Oak Boulevard",     "direction": "north-south","lanes": 2, "speed_limit_ms": 11.0, "length_m": 500},
        {"name": "Park Road",         "direction": "east-west",  "lanes": 1, "speed_limit_ms": 8.0,  "length_m": 300},
        {"name": "Harbor Drive",      "direction": "east-west",  "lanes": 3, "speed_limit_ms": 20.0, "length_m": 1200},
    ],

    "intersections": [
        {"id": 1, "streets": ["Main Street", "Oak Boulevard"],      "has_traffic_light": True,  "position": {"x": 200, "y": 400}},
        {"id": 2, "streets": ["Main Street", "Highway 5"],          "has_traffic_light": True,  "position": {"x": 500, "y": 400}},
        {"id": 3, "streets": ["University Avenue", "Oak Boulevard"],"has_traffic_light": True,  "position": {"x": 200, "y": 200}},
        {"id": 4, "streets": ["University Avenue", "Highway 5"],    "has_traffic_light": True,  "position": {"x": 500, "y": 200}},
        {"id": 5, "streets": ["Park Road", "Oak Boulevard"],        "has_traffic_light": False, "position": {"x": 200, "y": 100}},
        {"id": 6, "streets": ["Harbor Drive", "Highway 5"],         "has_traffic_light": True,  "position": {"x": 500, "y": 600}},
    ],

    "landmarks": [
        {"name": "Airport",            "nearest_street": "Harbor Drive",      "nearest_intersection": "int_6", "position": {"x": 700, "y": 600}},
        {"name": "University Campus",  "nearest_street": "University Avenue", "nearest_intersection": "int_3", "position": {"x": 150, "y": 200}},
        {"name": "Downtown Mall",      "nearest_street": "Main Street",       "nearest_intersection": "int_1", "position": {"x": 200, "y": 420}},
        {"name": "Central Hospital",   "nearest_street": "Oak Boulevard",     "nearest_intersection": "int_5", "position": {"x": 180, "y": 100}},
        {"name": "Train Station",      "nearest_street": "Highway 5",         "nearest_intersection": "int_2", "position": {"x": 520, "y": 400}},
        {"name": "City Park",          "nearest_street": "Park Road",         "nearest_intersection": "int_5", "position": {"x": 250, "y": 80}},
        {"name": "Harbor Terminal",     "nearest_street": "Harbor Drive",      "nearest_intersection": "int_6", "position": {"x": 400, "y": 620}},
        {"name": "Office District",    "nearest_street": "Main Street",       "nearest_intersection": "int_2", "position": {"x": 480, "y": 380}},
    ],
}


# ============================================================
# ROUTES
# ============================================================

@app.route("/map", methods=["GET"])
def get_map():
    """Return the current city map."""
    return jsonify(city_map)


@app.route("/map", methods=["POST"])
def update_map():
    """
    Replace the city map with new data.
    The patrolling program calls this after building/updating the SLAM map.
    """
    global city_map
    new_map = request.get_json()
    if not new_map:
        return jsonify({"error": "No JSON body provided"}), 400

    new_map["last_updated"] = datetime.now().isoformat()
    new_map["version"] = city_map.get("version", 0) + 1
    city_map = new_map

    return jsonify({"status": "ok", "version": city_map["version"]})


@app.route("/map/status", methods=["GET"])
def map_status():
    """Return metadata about the current map."""
    return jsonify({
        "version": city_map.get("version"),
        "last_updated": city_map.get("last_updated"),
        "city": city_map.get("city"),
        "num_streets": len(city_map.get("streets", [])),
        "num_intersections": len(city_map.get("intersections", [])),
        "num_landmarks": len(city_map.get("landmarks", [])),
    })


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("=" * 50)
    print("AutoPassing Map Server")
    print("=" * 50)
    print(f"Map: {city_map['city']} (v{city_map['version']})")
    print(f"  {len(city_map['streets'])} streets")
    print(f"  {len(city_map['intersections'])} intersections")
    print(f"  {len(city_map['landmarks'])} landmarks")
    print()
    print("Endpoints:")
    print("  GET  http://127.0.0.1:8100/map        → full map")
    print("  POST http://127.0.0.1:8100/map        → update map")
    print("  GET  http://127.0.0.1:8100/map/status  → map info")
    print("=" * 50)

    app.run(host="127.0.0.1", port=8100, debug=True)
