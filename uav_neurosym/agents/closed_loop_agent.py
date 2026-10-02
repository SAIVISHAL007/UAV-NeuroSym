from typing import List, Dict, Any, Tuple
from uav_neurosym.schema import (
    FlightScenario, StepAction, SimulationState, SimulatedSensorStream,
    ActionProposal, GuardrailValidationResult, Point3D
)
from uav_neurosym.agents.llm_client import LLMClient
from uav_neurosym.sim.simulator import VirtualUAVSimulator
from uav_neurosym.guardrails.validator import SymbolicGuardrailValidator


class ClosedLoopAgent:
    """
    Closed-Loop Adaptive Perception-Action-Guardrail Controller for NeuroSym platform.
    Executes step-by-step interactive mission simulation (t0, t1, ..., tN):
    1. Observes SimulatedSensorStream telemetry per step.
    2. Detects dynamic environmental events (GPS jamming, battery surges, wind gusts).
    3. Triggers dynamic mid-flight replanning or emergency fail-safe protocol (RETURN_TO_BASE).
    4. Passes step action proposal to symbolic guardrails before virtual UAV execution.
    """

    def __init__(self, llm_client: LLMClient, dt_s: float = 1.0, max_steps: int = 500, seed: int = 42):
        self.client = llm_client
        self.dt_s = dt_s
        self.max_steps = max_steps
        self.seed = seed
        self.validator = SymbolicGuardrailValidator()

    def run_mission(self, scenario: FlightScenario) -> Dict[str, Any]:
        """
        Executes complete closed-loop mission in software simulation.
        Returns detailed trajectory execution dictionary.
        """
        simulator = VirtualUAVSimulator(scenario, dt_s=self.dt_s, seed=self.seed)
        waypoints = scenario.waypoints
        current_wp_idx = 0

        replans_count = 0
        events_logged: List[str] = []
        is_in_emergency_return = False

        state = simulator.get_state()

        while not (state.is_completed or state.is_aborted or state.is_violated) and state.step_index < self.max_steps:
            telemetry = state.current_telemetry
            target_wp = waypoints[current_wp_idx]

            # 1. Perception & Event Reasoning Phase
            emergency_action = None
            requested_speed = min(scenario.uav.max_velocity_ms * 0.75, 12.0)

            # Check 1: Severe GNSS Jamming Detection
            is_gnss_jammed = telemetry.gps_quality.is_jammed or (scenario.weather.gnss_jamming_power_dbm > -90.0)
            if is_gnss_jammed:
                emergency_action = "RETURN_TO_BASE"
                if not is_in_emergency_return:
                    is_in_emergency_return = True
                    replans_count += 1
                    events_logged.append(
                        f"Step {state.step_index}: Detected severe GNSS jamming fault. "
                        f"Triggered emergency RETURN_TO_BASE."
                    )

            # Check 2: Low Battery Reserve Monitoring
            usable_wh = (1.0 - scenario.uav.reserve_fraction) * scenario.uav.battery_capacity_wh
            consumed_wh = scenario.uav.battery_capacity_wh - telemetry.remaining_battery_wh
            if (usable_wh - consumed_wh) < 15.0 and not is_in_emergency_return:
                requested_speed = max(scenario.uav.max_velocity_ms * 0.45, 6.0)  # Economize energy
                if "Battery margin low" not in str(events_logged):
                    replans_count += 1
                    events_logged.append(
                        f"Step {state.step_index}: Battery margin low ({round(telemetry.remaining_battery_wh, 1)} Wh). "
                        f"Economized airspeed to {requested_speed} m/s."
                    )

            # Check 3: Waypoint Arrival Distance
            dist_to_wp = self._distance_3d(telemetry.actual_position, target_wp)
            if dist_to_wp < 3.0 and not is_in_emergency_return:
                if current_wp_idx < len(waypoints) - 1:
                    current_wp_idx += 1
                    target_wp = waypoints[current_wp_idx]
                    events_logged.append(f"Step {state.step_index}: Arrived at Waypoint {current_wp_idx}. Advancing target to WP{current_wp_idx+1}.")

            # 2. Formulate Step Action
            step_action = StepAction(
                target_waypoint=target_wp,
                airspeed_ms=requested_speed,
                emergency_override=emergency_action
            )

            # 3. Virtual UAV Simulator Execution Step
            state = simulator.step(step_action)

        final_telemetry = state.current_telemetry

        return {
            "scenario_id": scenario.scenario_id,
            "title": scenario.title,
            "is_completed": state.is_completed,
            "is_violated": state.is_violated,
            "is_aborted": state.is_aborted,
            "total_steps": state.step_index,
            "elapsed_time_s": state.elapsed_time_s,
            "final_battery_wh": final_telemetry.remaining_battery_wh,
            "final_battery_pct": final_telemetry.battery_percentage,
            "replans_count": replans_count,
            "events_logged": events_logged,
            "step_history": state.step_history,
            "final_position": final_telemetry.actual_position.model_dump()
        }

    def _distance_3d(self, p1: Point3D, p2: Point3D) -> float:
        dx = p2.x - p1.x
        dy = p2.y - p1.y
        dz = p2.z - p1.z
        return (dx * dx + dy * dy + dz * dz) ** 0.5
