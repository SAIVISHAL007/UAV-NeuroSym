from typing import Dict, Any
from uav_neurosym.schema import FlightScenario, ActionProposal, GuardrailValidationResult
from uav_neurosym.guardrails.physics_engine import check_physics_feasibility
from uav_neurosym.guardrails.geometry_engine import check_geometry_feasibility
from uav_neurosym.guardrails.policy_engine import check_policy_feasibility


class SymbolicGuardrailValidator:
    """
    Unified Deterministic Symbolic Guardrail Engine for UAV Flight Decisions.
    Combines First-Order Physics, Aerodynamics, Energy Integrals, 3D Geofence Containment,
    No-Fly Zone Intersections, and UTM Policy Compliance.
    """

    def validate(self, scenario: FlightScenario, proposal: ActionProposal) -> GuardrailValidationResult:
        all_metrics: Dict[str, Any] = {}

        # 1. Physics & Energy Validation
        p_valid, p_cat, p_err, p_metrics = check_physics_feasibility(
            uav=scenario.uav,
            weather=scenario.weather,
            proposed_path=proposal.proposed_path,
            target_airspeed_ms=proposal.target_airspeed_ms,
            estimated_duration_s=proposal.estimated_duration_s
        )
        all_metrics.update(p_metrics)

        if not p_valid:
            msg = self._build_diagnostic_message(p_cat, p_err, p_metrics, scenario)
            return GuardrailValidationResult(
                is_valid=False,
                violation_category=p_cat,
                numerical_error=p_err,
                diagnostic_message=msg,
                metrics=all_metrics
            )

        # 2. Geometry & Spatial Validation
        g_valid, g_cat, g_err, g_metrics = check_geometry_feasibility(
            proposed_path=proposal.proposed_path,
            airspace=scenario.airspace,
            no_fly_zones=scenario.no_fly_zones,
            min_separation_m=scenario.min_separation_m
        )
        all_metrics.update(g_metrics)

        if not g_valid:
            msg = self._build_diagnostic_message(g_cat, g_err, g_metrics, scenario)
            return GuardrailValidationResult(
                is_valid=False,
                violation_category=g_cat,
                numerical_error=g_err,
                diagnostic_message=msg,
                metrics=all_metrics
            )

        # 3. Policy & Emergency Rules Validation
        pol_valid, pol_cat, pol_err, pol_metrics = check_policy_feasibility(scenario, proposal)
        all_metrics.update(pol_metrics)

        if not pol_valid:
            msg = self._build_diagnostic_message(pol_cat, pol_err, pol_metrics, scenario)
            return GuardrailValidationResult(
                is_valid=False,
                violation_category=pol_cat,
                numerical_error=pol_err,
                diagnostic_message=msg,
                metrics=all_metrics
            )

        # Passed all guardrails!
        return GuardrailValidationResult(
            is_valid=True,
            violation_category=None,
            numerical_error=0.0,
            diagnostic_message="CERTIFIED SAFE: Proposed flight plan complies with all physical, aerodynamic, spatial, and policy guardrails.",
            metrics=all_metrics
        )

    def _build_diagnostic_message(
        self,
        category: str,
        error_mag: float,
        metrics: Dict[str, Any],
        scenario: FlightScenario
    ) -> str:
        """Constructs precise, numerical diagnostic feedback for the reflection loop."""
        if category == "ENERGY_EXHAUSTION":
            return (
                f"SYMBOLIC REJECTION [ENERGY_EXHAUSTION]: Proposed path consumes {metrics.get('energy_consumed_wh', 0)} Wh, "
                f"which exceeds the maximum usable battery budget of {metrics.get('usable_battery_wh', 0)} Wh "
                f"(Reserve: {metrics.get('reserve_wh', 0)} Wh). Excess required energy: {round(error_mag, 2)} Wh. "
                f"ACTION REQUIRED: Reduce target airspeed or select a shorter/direct waypoint sequence."
            )
        elif category == "NO_FLY_ZONE_PENETRATION":
            return (
                f"SYMBOLIC REJECTION [NO_FLY_ZONE_PENETRATION]: Proposed flight trajectory intersects active No-Fly Zone "
                f"'{metrics.get('nfz_name', 'NFZ')}' with a penetration distance of {metrics.get('penetration_length_m', 0)} meters. "
                f"ACTION REQUIRED: Reroute around the NFZ polygon boundary."
            )
        elif category == "GEOFENCE_LATERAL_BREACH":
            return (
                f"SYMBOLIC REJECTION [GEOFENCE_LATERAL_BREACH]: Waypoint index {metrics.get('waypoint_index', 0)} breaches "
                f"authorized geofence boundary by {round(error_mag, 2)} meters. "
                f"ACTION REQUIRED: Adjust waypoint coordinates to remain strictly within operational geofence."
            )
        elif category == "ALTITUDE_EXCEEDED_CEILING":
            return (
                f"SYMBOLIC REJECTION [ALTITUDE_EXCEEDED_CEILING]: Waypoint altitude {metrics.get('waypoint_alt')}m exceeds "
                f"maximum authorized ceiling limit of {metrics.get('max_alt')}m. "
                f"ACTION REQUIRED: Descend flight path to maintain altitude below ceiling limit."
            )
        elif category == "STALL_VELOCITY_BREACH":
            return (
                f"SYMBOLIC REJECTION [STALL_VELOCITY_BREACH]: Target speed {metrics.get('target_speed')} m/s is below "
                f"fixed-wing aerodynamic stall velocity limit of {metrics.get('stall_speed')} m/s. "
                f"ACTION REQUIRED: Increase target airspeed above stall threshold."
            )
        elif category == "GNSS_DENIAL_SAFETY_POLICY_VIOLATION":
            return (
                f"SYMBOLIC REJECTION [GNSS_DENIAL_POLICY]: Severe GNSS denial fault detected (severity: {error_mag}). "
                f"Operating nominal path under unaugmented GPS is unsafe. "
                f"ACTION REQUIRED: Activate emergency action protocol (e.g. 'RETURN_TO_BASE' or 'OPTICAL_FLOW_HOLD')."
            )
        else:
            return f"SYMBOLIC REJECTION [{category}]: Violation magnitude = {round(error_mag, 2)}. Re-plan required."
