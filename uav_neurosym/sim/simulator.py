import math
import random
from typing import List, Tuple, Dict, Any, Optional
from uav_neurosym.schema import (
    FlightScenario, StepAction, SimulationState, SimulatedSensorStream,
    Point3D, GuardrailValidationResult, ActionProposal, GPSQuality, NetworkState
)
from uav_neurosym.sim.environment import VirtualEnvironment
from uav_neurosym.guardrails.physics_engine import calculate_flight_power
from uav_neurosym.guardrails.validator import SymbolicGuardrailValidator


class VirtualUAVSimulator:
    """
    Discrete-time Virtual UAV Simulation Engine for NeuroSym platform.
    Executes closed-loop step progression (t0, t1, ..., tN) in software.
    - Integrates 3D kinematics and actual battery power consumption per step (dt).
    - Simulates sensor telemetry streams (GPS noise, IMU, battery Wh, network state).
    - Evaluates step-by-step symbolic guardrail compliance.
    """

    def __init__(self, scenario: FlightScenario, dt_s: float = 1.0, seed: int = 42):
        self.scenario = scenario
        self.dt_s = dt_s
        self.seed = seed
        random.seed(seed)

        self.environment = VirtualEnvironment(scenario, seed=seed)
        self.validator = SymbolicGuardrailValidator()

        # State Variables
        self.elapsed_time_s: float = 0.0
        self.step_index: int = 0
        self.actual_position = Point3D(
            x=scenario.spawn_point.x,
            y=scenario.spawn_point.y,
            z=scenario.spawn_point.z
        )
        self.remaining_battery_wh: float = scenario.uav.battery_capacity_wh
        self.is_completed: bool = False
        self.is_aborted: bool = False
        self.is_violated: bool = False

        self.history: List[Dict[str, Any]] = []

        # Initial Telemetry Snapshot
        self.current_telemetry = self._generate_telemetry_stream(
            target_speed_ms=0.0,
            power_w=0.0
        )

    def get_state(self) -> SimulationState:
        """Returns current snapshot of the simulation state."""
        return SimulationState(
            step_index=self.step_index,
            elapsed_time_s=self.elapsed_time_s,
            dt_s=self.dt_s,
            is_completed=self.is_completed,
            is_aborted=self.is_aborted,
            is_violated=self.is_violated,
            current_telemetry=self.current_telemetry,
            active_faults=self.environment.active_faults,
            step_history=self.history
        )

    def step(self, action: StepAction) -> SimulationState:
        """
        Advances the virtual UAV simulation by dt_s seconds using requested StepAction.
        Updates 3D kinematics, power/battery state, sensor streams, and safety validation.
        """
        if self.is_completed or self.is_aborted:
            return self.get_state()

        self.step_index += 1
        self.elapsed_time_s += self.dt_s

        # 1. Update Environment State
        weather, gps_quality, network_state, obstacles = self.environment.update_step(
            self.elapsed_time_s, self.dt_s
        )

        # 2. Check Emergency Override Actions
        if action.emergency_override == "RETURN_TO_BASE":
            action.target_waypoint = self.scenario.spawn_point
            action.airspeed_ms = min(action.airspeed_ms, 8.0)
        elif action.emergency_override in ["LAND", "HOLD"]:
            action.airspeed_ms = 0.0

        # 3. Kinematic Position Update (Move actual_position toward target_waypoint)
        current_p = self.actual_position
        target_p = action.target_waypoint

        dx = target_p.x - current_p.x
        dy = target_p.y - current_p.y
        dz = target_p.z - current_p.z
        distance_to_target = math.sqrt(dx * dx + dy * dy + dz * dz)

        if distance_to_target < 1.0:
            # Reached target waypoint
            actual_speed = 0.0
            heading_deg = 0.0
        else:
            # Move step distance v * dt
            step_dist = min(action.airspeed_ms * self.dt_s, distance_to_target)
            unit_x = dx / distance_to_target
            unit_y = dy / distance_to_target
            unit_z = dz / distance_to_target

            # Apply wind disturbance vector to movement
            wind_dx = (weather.wind_speed_ms * 0.05) * math.cos(math.radians(weather.wind_direction_deg)) * self.dt_s
            wind_dy = (weather.wind_speed_ms * 0.05) * math.sin(math.radians(weather.wind_direction_deg)) * self.dt_s

            current_p.x += unit_x * step_dist + wind_dx
            current_p.y += unit_y * step_dist + wind_dy
            current_p.z += unit_z * step_dist

            actual_speed = step_dist / self.dt_s
            heading_deg = math.degrees(math.atan2(dy, dx)) % 360.0

        # 4. Integrate Battery Power Draw (Eq. 9 & 10)
        power_w = calculate_flight_power(self.scenario.uav, actual_speed, maneuver_rate=1.0)
        if weather.icing_risk:
            power_w *= 1.25

        energy_used_wh = (power_w * self.dt_s) / 3600.0
        self.remaining_battery_wh = max(0.0, self.remaining_battery_wh - energy_used_wh)

        # 5. Generate Sensor Telemetry Stream (with simulated GPS noise)
        self.current_telemetry = self._generate_telemetry_stream(
            target_speed_ms=actual_speed,
            power_w=power_w,
            heading_deg=heading_deg
        )

        # 6. Step Safety Guardrail Validation
        # Construct temporary single-step action proposal for validator
        proposal = ActionProposal(
            proposed_path=[self.actual_position],
            target_airspeed_ms=actual_speed,
            estimated_duration_s=self.elapsed_time_s,
            emergency_action=action.emergency_override,
            rationale=f"Step {self.step_index} execution"
        )

        validation = self.validator.validate(self.scenario, proposal)
        if not validation.is_valid:
            self.is_violated = True

        # Check battery exhaustion limit
        usable_battery_wh = (1.0 - self.scenario.uav.reserve_fraction) * self.scenario.uav.battery_capacity_wh
        total_energy_consumed = self.scenario.uav.battery_capacity_wh - self.remaining_battery_wh
        if total_energy_consumed > usable_battery_wh:
            self.is_violated = True

        # Check mission completion
        dist_to_final = math.sqrt(
            (current_p.x - self.scenario.waypoints[-1].x) ** 2 +
            (current_p.y - self.scenario.waypoints[-1].y) ** 2 +
            (current_p.z - self.scenario.waypoints[-1].z) ** 2
        )
        if dist_to_final < 3.0 and not self.is_violated:
            self.is_completed = True

        # Log Step Snapshot
        self.history.append({
            "step": self.step_index,
            "time_s": self.elapsed_time_s,
            "actual_position": self.actual_position.model_dump(),
            "estimated_position": self.current_telemetry.estimated_position.model_dump(),
            "remaining_battery_wh": round(self.remaining_battery_wh, 2),
            "power_draw_w": round(power_w, 2),
            "ground_speed_ms": round(actual_speed, 2),
            "heading_deg": round(heading_deg, 2),
            "altitude_m": round(self.actual_position.z, 2),
            "battery_percentage": round(self.remaining_battery_wh / max(self.scenario.uav.battery_capacity_wh, 0.01) * 100.0, 2),
            "is_valid": validation.is_valid,
            "violation": validation.violation_category
        })

        return self.get_state()

    def _generate_telemetry_stream(
        self,
        target_speed_ms: float,
        power_w: float,
        heading_deg: float = 0.0
    ) -> SimulatedSensorStream:
        """Generates a SimulatedSensorStream output containing software-generated sensor noise."""
        weather = self.environment.current_weather
        gps_q = GPSQuality(is_jammed=False)
        net_s = NetworkState()

        # Generate simulated estimated position with GPS noise
        noise_std = 0.5
        if hasattr(self.environment, "last_gps_q"):
            noise_std = self.environment.last_gps_q.position_noise_m

        noisy_x = self.actual_position.x + random.gauss(0, noise_std)
        noisy_y = self.actual_position.y + random.gauss(0, noise_std)
        noisy_z = self.actual_position.z + random.gauss(0, noise_std * 0.5)

        estimated_pos = Point3D(
            x=round(noisy_x, 2),
            y=round(noisy_y, 2),
            z=round(noisy_z, 2)
        )

        batt_pct = (self.remaining_battery_wh / self.scenario.uav.battery_capacity_wh) * 100.0

        return SimulatedSensorStream(
            timestamp_s=self.elapsed_time_s,
            estimated_position=estimated_pos,
            actual_position=self.actual_position.model_copy(),
            ground_speed_ms=round(target_speed_ms, 2),
            heading_deg=round(heading_deg, 1),
            remaining_battery_wh=round(self.remaining_battery_wh, 2),
            battery_percentage=round(batt_pct, 1),
            power_draw_w=round(power_w, 2),
            gps_quality=gps_q,
            network_state=net_s,
            weather_state=weather.model_copy(),
            observed_obstacles=self.environment.obstacles
        )
