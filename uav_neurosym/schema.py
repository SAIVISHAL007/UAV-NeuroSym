from enum import Enum
from typing import List, Optional, Tuple, Dict, Any
from pydantic import BaseModel, Field


class UAVType(str, Enum):
    ROTORCRAFT = "rotorcraft"
    FIXED_WING = "fixed_wing"
    HYBRID_VTOL = "hybrid_vtol"


class MissionType(str, Enum):
    INSPECTION = "inspection"
    DELIVERY = "delivery"
    SEARCH_AND_RESCUE = "search_and_rescue"
    RECONNAISSANCE = "reconnaissance"
    SWARM_COORDINATION = "swarm_coordination"
    HAZMAT_RESPONSE = "hazmat_response"


class RiskLevel(int, Enum):
    LOW = 0
    MODERATE = 1
    HIGH = 2
    CRITICAL = 3


class Point3D(BaseModel):
    x: float = Field(..., description="Ground coordinate X (meters)")
    y: float = Field(..., description="Ground coordinate Y (meters)")
    z: float = Field(..., description="Altitude AGL Z (meters)")

    def to_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def to_2d_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)


class UAVConfig(BaseModel):
    uav_id: str = "UAV-01"
    uav_type: UAVType = UAVType.ROTORCRAFT
    mass_kg: float = Field(3.5, description="UAV total mass including payload (kg)")
    battery_capacity_wh: float = Field(120.0, description="Total battery capacity Eb (Wh)")
    reserve_fraction: float = Field(0.20, description="Mandatory energy reserve fraction r (0.20 = 20%)")
    max_velocity_ms: float = Field(18.0, description="Maximum velocity vmax (m/s)")
    max_tilt_deg: float = Field(35.0, description="Maximum tilt angle phi_max (deg)")
    hover_power_w: float = Field(180.0, description="Hover power Ph (W)")
    drag_coeff_kd: float = Field(0.045, description="Aerodynamic drag coefficient kd")
    maneuver_coeff_km: float = Field(0.010, description="Maneuver power coefficient km")
    rotor_disk_area_m2: float = Field(0.35, description="Total rotor disk area Ad (m2)")
    wing_area_m2: Optional[float] = Field(None, description="Wing area S for fixed wing (m2)")
    stall_velocity_ms: Optional[float] = Field(None, description="Stall velocity for fixed wing (m/s)")


class Weather(BaseModel):
    wind_speed_ms: float = Field(4.5, description="Wind speed (m/s)")
    wind_direction_deg: float = Field(180.0, description="Wind direction (degrees)")
    gust_amplitude_ms: float = Field(1.5, description="Gust amplitude (m/s)")
    visibility_km: float = Field(10.0, description="Visibility distance (km)")
    icing_risk: bool = Field(False, description="Presence of atmospheric icing conditions")
    gnss_jamming_power_dbm: float = Field(-120.0, description="GNSS jamming power (dBm, > -90 is severe)")


class PolygonGeofence(BaseModel):
    name: str = "Operational Airspace"
    vertices: List[Tuple[float, float]] = Field(..., description="2D ground polygonal boundary points [(x, y)]")
    min_alt_m: float = Field(10.0, description="Minimum allowed altitude AGL (m)")
    max_alt_m: float = Field(120.0, description="Maximum allowed altitude AGL (m)")


class NoFlyZone(BaseModel):
    nfz_id: str
    name: str
    polygon: List[Tuple[float, float]] = Field(..., description="2D ground polygonal boundary [(x, y)]")
    min_alt_m: float = Field(0.0, description="NFZ floor altitude (m)")
    max_alt_m: float = Field(150.0, description="NFZ ceiling altitude (m)")
    is_active: bool = Field(True, description="Whether NFZ is actively enforced")


class FaultInjection(BaseModel):
    fault_type: str = Field(..., description="e.g. 'motor_failure', 'gnss_denial', 'battery_drain_surge'")
    start_time_s: float = Field(0.0, description="Time step when fault starts (s)")
    duration_s: float = Field(30.0, description="Duration of fault (s)")
    severity: float = Field(0.5, description="Fault severity from 0.0 to 1.0")


class FlightScenario(BaseModel):
    scenario_id: str
    title: str
    description: str
    mission_type: MissionType
    uav: UAVConfig
    weather: Weather
    airspace: PolygonGeofence
    no_fly_zones: List[NoFlyZone] = Field(default_factory=list)
    spawn_point: Point3D
    waypoints: List[Point3D]
    time_budget_s: float = Field(300.0, description="Total mission time budget T (s)")
    min_separation_m: float = Field(10.0, description="Minimum swarm/obstacle separation distance (m)")
    min_ttc_s: float = Field(3.0, description="Minimum Time-To-Collision TTCmin (s)")
    faults: List[FaultInjection] = Field(default_factory=list)
    risk_level: RiskLevel = RiskLevel.LOW
    safety_tag: str = "Nominal"


class ActionProposal(BaseModel):
    proposed_path: List[Point3D] = Field(..., description="Ordered list of waypoints chosen for mission execution")
    target_airspeed_ms: float = Field(..., description="Target flight speed (m/s)")
    estimated_duration_s: float = Field(..., description="Estimated mission duration (s)")
    emergency_action: Optional[str] = Field(None, description="e.g. 'REROUTE', 'RETURN_TO_BASE', 'EMERGENCY_LANDING'")
    rationale: str = Field(..., description="Reasoning narrative explaining why this path/action was selected")


class GuardrailValidationResult(BaseModel):
    is_valid: bool = Field(..., description="True if proposed action satisfies ALL symbolic guardrails")
    violation_category: Optional[str] = Field(None, description="e.g. 'ENERGY_EXHAUSTION', 'NFZ_PENETRATION', 'ALTITUDE_BREACH', 'STALL_SPEED'")
    numerical_error: float = Field(0.0, description="Quantified deviation amount (e.g., Wh over limit, meters breached)")
    diagnostic_message: str = Field(..., description="Detailed diagnostic string to feed back into the agent reflection loop")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Calculated physical values (energy used Wh, reserve remaining Wh, distance m, etc.)")
