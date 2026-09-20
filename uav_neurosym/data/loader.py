import json
import requests
import os
import glob
from typing import List, Optional, Dict, Any
from uav_neurosym.schema import (
    FlightScenario, UAVConfig, Weather, PolygonGeofence, NoFlyZone,
    Point3D, MissionType, RiskLevel, FaultInjection, UAVType
)


def load_benchmark_scenarios() -> List[FlightScenario]:
    """
    Loads benchmark scenarios directly from JSON dataset files in data/scenarios/.
    If local JSON files exist, parses and returns them; otherwise falls back to memory generator.
    """
    scenarios_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "scenarios")
    json_files = glob.glob(os.path.join(scenarios_dir, "*.json"))

    if json_files:
        scenarios: List[FlightScenario] = []
        for file_path in sorted(json_files):
            try:
                scenarios.append(load_scenario_from_json(file_path))
            except Exception as e:
                print(f"Warning: Failed to parse scenario JSON {file_path}: {e}")
        if scenarios:
            return scenarios

    # Fallback memory generator
    return _generate_default_scenarios()


def load_scenario_from_json(json_path: str) -> FlightScenario:
    """Parses a local UAVBench JSON scenario file into a FlightScenario object."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return FlightScenario(**data)


def fetch_official_uavbench_json(raw_url: str) -> Optional[FlightScenario]:
    """Fetches a raw JSON flight scenario directly from the official UAVBench GitHub repo."""
    try:
        res = requests.get(raw_url, timeout=10)
        if res.status_code == 200:
            return FlightScenario(**res.json())
    except Exception:
        pass
    return None


def _generate_default_scenarios() -> List[FlightScenario]:
    """Fallback generator in case local JSON files are missing."""
    scenarios: List[FlightScenario] = []

    scenarios.append(FlightScenario(
        scenario_id="UAVBENCH-001",
        title="Nominal Urban Corridor Inspection",
        description="Rotorcraft performing routine grid survey over urban industrial corridor.",
        mission_type=MissionType.INSPECTION,
        uav=UAVConfig(
            uav_id="UAV-Quad-01",
            uav_type=UAVType.ROTORCRAFT,
            mass_kg=2.8,
            battery_capacity_wh=150.0,
            max_velocity_ms=15.0,
            hover_power_w=140.0
        ),
        weather=Weather(wind_speed_ms=3.0, visibility_km=12.0),
        airspace=PolygonGeofence(
            name="Sector A Airspace",
            vertices=[(-200, -200), (300, -200), (300, 300), (-200, 300)],
            min_alt_m=10.0,
            max_alt_m=100.0
        ),
        no_fly_zones=[],
        spawn_point=Point3D(x=0.0, y=0.0, z=20.0),
        waypoints=[
            Point3D(x=50.0, y=50.0, z=30.0),
            Point3D(x=150.0, y=50.0, z=40.0),
            Point3D(x=150.0, y=150.0, z=40.0),
            Point3D(x=0.0, y=0.0, z=20.0)
        ],
        time_budget_s=400.0,
        risk_level=RiskLevel.LOW,
        safety_tag="Nominal"
    ))

    return scenarios
