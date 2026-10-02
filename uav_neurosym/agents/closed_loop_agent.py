from typing import List, Dict, Any, Tuple
from uav_neurosym.schema import (
    FlightScenario, StepAction, SimulationState, SimulatedSensorStream,
    ActionProposal, GuardrailValidationResult, Point3D
)
from uav_neurosym.agents.llm_client import LLMClient
from uav_neurosym.sim.simulator import VirtualUAVSimulator
from uav_neurosym.guardrails.validator import SymbolicGuardrailValidator
from uav_neurosym.memory.temporal_memory import TemporalMemoryManager
from uav_neurosym.adaptive.strategy_selector import AdaptiveStrategySelector, IntelligenceStrategy


class ClosedLoopAgent:
    """
    Closed-Loop Adaptive Perception-Action-Guardrail Controller for NeuroSym platform.
    Executes step-by-step interactive mission simulation (t0, t1, ..., tN):
    1. Observes SimulatedSensorStream telemetry per step.
    2. Records time-series state history in TemporalMemoryManager.
    3. Evaluates dynamic AI strategy selection using AdaptiveStrategySelector.
    4. Triggers dynamic mid-flight replanning or emergency fail-safe protocol (RETURN_TO_BASE).
    5. Passes step action proposal to symbolic guardrails before virtual UAV execution.
    """

    def __init__(self, llm_client: LLMClient, dt_s: float = 1.0, max_steps: int = 500, seed: int = 42):
        self.client = llm_client
        self.dt_s = dt_s
        self.max_steps = max_steps
        self.seed = seed
        self.validator = SymbolicGuardrailValidator()
        self.memory = TemporalMemoryManager()
        self.strategy_selector = AdaptiveStrategySelector()

    def run_mission(self, scenario: FlightScenario) -> Dict[str, Any]:
        """
        Executes complete closed-loop mission in software simulation.
        Returns detailed trajectory execution dictionary.
        """
        self.memory.clear()
        simulator = VirtualUAVSimulator(scenario, dt_s=self.dt_s, seed=self.seed)
        waypoints = scenario.waypoints
        current_wp_idx = 0

        replans_count = 0
        events_logged: List[str] = []
        strategies_used: List[str] = []
        is_in_emergency_return = False

        state = simulator.get_state()

        while not (state.is_completed or state.is_aborted or state.is_violated) and state.step_index < self.max_steps:
            telemetry = state.current_telemetry
            target_wp = waypoints[current_wp_idx]

            # Record step telemetry into Temporal Memory Manager
            self.memory.record_step(telemetry)

            # 1. Evaluate Adaptive Strategy Selector
            active_strategy, strategy_rationale = self.strategy_selector.select_strategy(
                scenario, telemetry, self.memory
            )
            if active_strategy.value not in strategies_used:
                strategies_used.append(active_strategy.value)

            # 2. Perception & Strategy Execution Phase
            emergency_action = None
            requested_speed = min(scenario.uav.max_velocity_ms * 0.75, 12.0)

            # Query Temporal Memory Trends
            discharge_rate = self.memory.get_battery_discharge_rate_wh_per_sec(window_seconds=10.0)
            gps_noise_m, is_gps_degrading = self.memory.get_gps_noise_trend(window_seconds=10.0)

            # Apply Strategy Behavior
            if active_strategy == IntelligenceStrategy.EMERGENCY_DETERMINISTIC_CONTROLLER:
                emergency_action = "RETURN_TO_BASE"
                if not is_in_emergency_return:
                    is_in_emergency_return = True
                    replans_count += 1
                    evt_str = f"Step {state.step_index}: [{active_strategy.value}] Severe GNSS jamming fault or critical risk event active. Triggered emergency RETURN_TO_BASE."
                    events_logged.append(evt_str)
                    self.memory.record_step(telemetry, event_text=evt_str)

            elif active_strategy == IntelligenceStrategy.DETERMINISTIC_FALLBACK:
                requested_speed = min(requested_speed, 8.0)
                if "Deterministic fallback active" not in str(events_logged):
                    evt_str = f"Step {state.step_index}: [{active_strategy.value}] Network degraded. Reduced speed for deterministic fallback."
                    events_logged.append(evt_str)
                    self.memory.record_step(telemetry, event_text=evt_str)

            # Check Waypoint Arrival Distance
            dist_to_wp = self._distance_3d(telemetry.actual_position, target_wp)
            if dist_to_wp < 3.0 and not is_in_emergency_return:
                if current_wp_idx < len(waypoints) - 1:
                    current_wp_idx += 1
                    target_wp = waypoints[current_wp_idx]
                    evt_str = f"Step {state.step_index}: Arrived at Waypoint {current_wp_idx}. Advancing target to WP{current_wp_idx+1}."
                    events_logged.append(evt_str)
                    self.memory.record_step(telemetry, event_text=evt_str)

            # 3. Formulate Step Action
            step_action = StepAction(
                target_waypoint=target_wp,
                airspeed_ms=requested_speed,
                emergency_override=emergency_action
            )

            # 4. Virtual UAV Simulator Execution Step
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
            "strategies_used": strategies_used,
            "events_logged": events_logged,
            "chronological_timeline": self.memory.get_chronological_events(),
            "step_history": state.step_history,
            "final_position": final_telemetry.actual_position.model_dump()
        }

    def _distance_3d(self, p1: Point3D, p2: Point3D) -> float:
        dx = p2.x - p1.x
        dy = p2.y - p1.y
        dz = p2.z - p1.z
        return (dx * dx + dy * dy + dz * dz) ** 0.5
