import pytest
from uav_neurosym.schema import (
    UAVConfig, Weather, PolygonGeofence, NoFlyZone, Point3D,
    ActionProposal, UAVType, FlightScenario, MissionType, RiskLevel
)
from uav_neurosym.guardrails.physics_engine import check_physics_feasibility
from uav_neurosym.guardrails.geometry_engine import check_geometry_feasibility
from uav_neurosym.guardrails.validator import SymbolicGuardrailValidator
from uav_neurosym.agents.llm_client import LLMClient
from uav_neurosym.agents.neuro_symbolic_agent import NeuroSymbolicAgent


def test_physics_energy_exhaustion():
    uav = UAVConfig(battery_capacity_wh=50.0, reserve_fraction=0.20)  # Usable = 40Wh
    weather = Weather(wind_speed_ms=2.0)
    path = [Point3D(x=0, y=0, z=20), Point3D(x=10000, y=0, z=20)]  # Long path

    is_valid, cat, err, metrics = check_physics_feasibility(
        uav, weather, path, target_airspeed_ms=15.0, estimated_duration_s=666.0
    )
    assert is_valid is False
    assert cat == "ENERGY_EXHAUSTION"
    assert err > 0.0


def test_geometry_nfz_penetration():
    airspace = PolygonGeofence(vertices=[(0, 0), (500, 0), (500, 500), (0, 500)])
    nfz = NoFlyZone(
        nfz_id="NFZ-1", name="Test NFZ",
        polygon=[(50, 50), (150, 50), (150, 150), (50, 150)],
        min_alt_m=0.0, max_alt_m=100.0, is_active=True
    )
    # Path passing straight through NFZ
    path = [Point3D(x=0, y=100, z=30), Point3D(x=200, y=100, z=30)]

    is_valid, cat, err, metrics = check_geometry_feasibility(path, airspace, [nfz])
    assert is_valid is False
    assert cat == "NO_FLY_ZONE_PENETRATION"


def test_neuro_symbolic_reflection_loop():
    sc = FlightScenario(
        scenario_id="TEST-001",
        title="Test Scenario",
        description="Test",
        mission_type=MissionType.INSPECTION,
        uav=UAVConfig(battery_capacity_wh=150.0),
        weather=Weather(),
        airspace=PolygonGeofence(vertices=[(0, 0), (500, 0), (500, 500), (0, 500)]),
        no_fly_zones=[
            NoFlyZone(
                nfz_id="NFZ-T", name="Blocked",
                polygon=[(40, 20), (160, 20), (160, 120), (40, 120)],
                min_alt_m=0.0, max_alt_m=150.0, is_active=True
            )
        ],
        spawn_point=Point3D(x=0, y=0, z=20),
        waypoints=[Point3D(x=100, y=50, z=35)],
        risk_level=RiskLevel.HIGH
    )

    client = LLMClient(provider="mock")
    agent = NeuroSymbolicAgent(client, max_retries=3)
    proposal, val, history = agent.run(sc)

    assert len(history) > 1  # Verify reflection loop triggered
    assert val.is_valid is True  # Verify self-corrected to safe plan
