import math
from typing import List, Tuple, Dict, Any
from uav_neurosym.schema import UAVConfig, Point3D, UAVType, Weather


def calculate_hover_power(uav: UAVConfig) -> float:
    """
    Calculate hover power Ph based on Eq. (11) from UAVBench paper:
    Ph ≈ c_eta * (m^(3/2) / sqrt(Ad))
    """
    if uav.hover_power_w > 0:
        return uav.hover_power_w
    c_eta = 30.0  # Propulsive efficiency factor
    disk_area = max(uav.rotor_disk_area_m2, 0.05)
    return c_eta * (math.pow(uav.mass_kg, 1.5) / math.sqrt(disk_area))


def calculate_flight_power(uav: UAVConfig, airspeed_ms: float, maneuver_rate: float = 0.0) -> float:
    """
    Calculate dynamic flight power P(v, u_dot) based on Eq. (9) from UAVBench paper:
    P(v, u_dot) = Ph + kd * v^3 + km * ||u_dot||_2
    """
    Ph = calculate_hover_power(uav)
    drag_power = uav.drag_coeff_kd * math.pow(airspeed_ms, 3)
    maneuver_power = uav.maneuver_coeff_km * maneuver_rate
    return Ph + drag_power + maneuver_power


def calculate_path_length_m(path: List[Point3D]) -> float:
    """Calculates total 3D Euclidean Euclidean path distance in meters."""
    if len(path) < 2:
        return 0.0
    total_dist = 0.0
    for i in range(len(path) - 1):
        p1 = path[i]
        p2 = path[i + 1]
        dx = p2.x - p1.x
        dy = p2.y - p1.y
        dz = p2.z - p1.z
        total_dist += math.sqrt(dx * dx + dy * dy + dz * dz)
    return total_dist


def check_physics_feasibility(
    uav: UAVConfig,
    weather: Weather,
    proposed_path: List[Point3D],
    target_airspeed_ms: float,
    estimated_duration_s: float
) -> Tuple[bool, str, float, Dict[str, Any]]:
    """
    Evaluates physical & aerodynamic constraints based on UAVBench equations (9)-(12).
    Returns (is_valid, error_category, error_magnitude, details_dict).
    """
    metrics: Dict[str, Any] = {}

    # 1. Max Velocity Check
    max_allowed_speed = uav.max_velocity_ms
    effective_wind = weather.wind_speed_ms + weather.gust_amplitude_ms
    ground_speed = target_airspeed_ms

    if target_airspeed_ms > max_allowed_speed + 0.1:
        error_mag = target_airspeed_ms - max_allowed_speed
        return False, "MAX_VELOCITY_EXCEEDED", error_mag, {
            "requested_speed": target_airspeed_ms,
            "max_velocity": max_allowed_speed
        }

    # 2. Fixed-Wing Stall Speed Check
    if uav.uav_type == UAVType.FIXED_WING:
        # Default stall speed calculation if not given
        rho = 1.225  # Air density kg/m3
        wing_area = uav.wing_area_m2 or 0.5
        CL_max = 1.4
        computed_stall = math.sqrt((2.0 * uav.mass_kg * 9.81) / (rho * wing_area * CL_max))
        stall_limit = uav.stall_velocity_ms or computed_stall
        if target_airspeed_ms < stall_limit:
            error_mag = stall_limit - target_airspeed_ms
            return False, "STALL_VELOCITY_BREACH", error_mag, {
                "target_speed": target_airspeed_ms,
                "stall_speed": stall_limit
            }

    # 3. Path & Duration Calculation
    path_dist = calculate_path_length_m(proposed_path)
    if target_airspeed_ms > 0:
        flight_time_s = path_dist / target_airspeed_ms
    else:
        flight_time_s = estimated_duration_s

    # 4. Energy Consumption Calculation (Eq. 10)
    # P_avg * t <= (1 - r) * Eb * 3600
    power_w = calculate_flight_power(uav, target_airspeed_ms, maneuver_rate=2.0)
    
    # Atmospheric Icing penalty increases power required by 25%
    if weather.icing_risk:
        power_w *= 1.25

    total_energy_joules = power_w * flight_time_s
    total_energy_wh = total_energy_joules / 3600.0

    usable_battery_wh = (1.0 - uav.reserve_fraction) * uav.battery_capacity_wh
    reserve_wh = uav.reserve_fraction * uav.battery_capacity_wh

    metrics["power_watts"] = round(power_w, 2)
    metrics["flight_time_s"] = round(flight_time_s, 2)
    metrics["energy_consumed_wh"] = round(total_energy_wh, 2)
    metrics["usable_battery_wh"] = round(usable_battery_wh, 2)
    metrics["reserve_wh"] = round(reserve_wh, 2)

    if total_energy_wh > usable_battery_wh:
        excess_wh = total_energy_wh - usable_battery_wh
        return False, "ENERGY_EXHAUSTION", excess_wh, metrics

    return True, "", 0.0, metrics
