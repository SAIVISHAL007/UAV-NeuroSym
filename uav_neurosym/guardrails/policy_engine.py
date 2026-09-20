from typing import List, Tuple, Dict, Any
from uav_neurosym.schema import Weather, FaultInjection, ActionProposal, FlightScenario


def check_policy_feasibility(
    scenario: FlightScenario,
    proposal: ActionProposal
) -> Tuple[bool, str, float, Dict[str, Any]]:
    """
    Evaluates regulatory compliance, UTM rules, weather limits, and fault emergency policies.
    Returns (is_valid, error_category, error_magnitude, details_dict).
    """
    weather = scenario.weather
    time_budget = scenario.time_budget_s

    # 1. Mission Duration vs Time Budget (Eq. 5)
    if proposal.estimated_duration_s > time_budget + 1.0:
        err = proposal.estimated_duration_s - time_budget
        return False, "TIME_BUDGET_EXCEEDED", err, {
            "estimated_duration": proposal.estimated_duration_s,
            "time_budget": time_budget
        }

    # 2. Weather Operating Thresholds (Visibility & Extreme Wind)
    # VFR minimum visibility is 3.0 km
    if weather.visibility_km < 3.0 and proposal.emergency_action is None:
        if proposal.target_airspeed_ms > 10.0:  # Must reduce speed under low visibility
            return False, "LOW_VISIBILITY_SPEED_POLICY_VIOLATION", 10.0, {
                "visibility_km": weather.visibility_km,
                "proposed_speed": proposal.target_airspeed_ms
            }

    # 3. Severe Weather Grounding / Abort Rule
    if weather.wind_speed_ms > 15.0 and proposal.emergency_action != "RETURN_TO_BASE":
        return False, "SEVERE_WIND_ABORT_REQUIRED", weather.wind_speed_ms, {
            "wind_speed_ms": weather.wind_speed_ms,
            "required_action": "RETURN_TO_BASE"
        }

    # 4. Fault Handling Rules (GNSS Jamming / Motor Fault)
    for fault in scenario.faults:
        if fault.fault_type == "gnss_denial" and fault.severity > 0.6:
            # Under severe GNSS denial, agent must switch to visual fallback or hold/land
            if proposal.emergency_action not in ["OPTICAL_FLOW_HOLD", "RETURN_TO_BASE", "EMERGENCY_LANDING"]:
                return False, "GNSS_DENIAL_SAFETY_POLICY_VIOLATION", fault.severity, {
                    "fault": fault.fault_type,
                    "severity": fault.severity,
                    "required_policy": "Emergency fail-safe policy must be activated under GNSS denial"
                }

    return True, "", 0.0, {"policy_checks": "PASSED"}
