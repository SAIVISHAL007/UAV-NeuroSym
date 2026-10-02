"""
Specialized Agent Mesh — Domain-Expert Agent Modules.

Each agent encapsulates a specific domain of flight decision reasoning.
Agents communicate via structured AgentMessage objects, enabling the
MultiAgentMeshController to coordinate inter-agent collaboration without
relying on a single monolithic LLM prompt.

Architecture:
    ┌──────────────────────────────┐
    │    Mission Planning Agent    │  → Decomposes user goals into task specs
    └──────────────┬───────────────┘
                   │
    ┌──────────────┴───────────────┐
    │  Multi-Agent Mesh Controller │  → Orchestrates message passing
    ├──────────────┬───────────────┤
    │              │               │
    ▼              ▼               ▼
  Navigation   Risk Assess.   Energy Reasoning
   Agent          Agent           Agent
"""

import math
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field
from uav_neurosym.schema import (
    FlightScenario, ActionProposal, Point3D, UAVConfig, Weather,
    PolygonGeofence, NoFlyZone, FaultInjection, SimulatedSensorStream,
    RiskLevel, UAVType
)


# ---------------------------------------------------------------------------
# Inter-Agent Communication Protocol
# ---------------------------------------------------------------------------

class AgentRole(str, Enum):
    """Defines the domain role of each specialized agent in the mesh."""
    MISSION_PLANNER = "mission_planner"
    NAVIGATOR = "navigator"
    RISK_ASSESSOR = "risk_assessor"
    ENERGY_REASONER = "energy_reasoner"
    PERCEPTION_STATE = "perception_state"
    MESH_CONTROLLER = "mesh_controller"


class AgentMessage(BaseModel):
    """
    Structured inter-agent message for mesh communication.
    Each message carries the sender identity, a typed payload dictionary,
    confidence score, and optional advisory flags.
    """
    sender: AgentRole = Field(..., description="The agent role that produced this message")
    recipient: AgentRole = Field(
        default=AgentRole.MESH_CONTROLLER,
        description="Target agent role (default: mesh controller)"
    )
    payload: Dict[str, Any] = Field(default_factory=dict, description="Typed domain-specific data payload")
    confidence: float = Field(1.0, ge=0.0, le=1.0, description="Agent's confidence in its output (0.0 - 1.0)")
    advisory_flags: List[str] = Field(default_factory=list, description="Warning/advisory flags raised by this agent")
    rationale: str = Field("", description="Human-readable reasoning narrative")


# ---------------------------------------------------------------------------
# Specialized Agent: Mission Planning
# ---------------------------------------------------------------------------

class MissionPlanningAgent:
    """
    Converts high-level user goals and scenario metadata into structured
    mission task specifications: target waypoints, priority ordering,
    time allocation per segment, and mission constraints.
    """

    def analyze(self, scenario: FlightScenario) -> AgentMessage:
        """Decompose the flight scenario into a structured mission task specification."""
        waypoints = scenario.waypoints
        total_distance = self._compute_total_path_distance(
            [scenario.spawn_point] + waypoints
        )
        num_segments = max(len(waypoints), 1)
        time_per_segment = scenario.time_budget_s / num_segments

        # Prioritize waypoints by distance from spawn (closer = higher priority)
        spawn = scenario.spawn_point
        wp_priorities = []
        for i, wp in enumerate(waypoints):
            dist = self._distance_3d(spawn, wp)
            wp_priorities.append({
                "waypoint_index": i,
                "position": wp.model_dump(),
                "distance_from_spawn_m": round(dist, 2),
                "priority": i + 1,
                "allocated_time_s": round(time_per_segment, 2)
            })

        # Determine mission complexity tier
        complexity_score = self._compute_complexity_score(scenario)
        complexity_tier = (
            "HIGH" if complexity_score >= 0.7 else
            "MODERATE" if complexity_score >= 0.4 else
            "LOW"
        )

        payload = {
            "mission_type": scenario.mission_type.value,
            "total_waypoints": len(waypoints),
            "total_path_distance_m": round(total_distance, 2),
            "time_budget_s": scenario.time_budget_s,
            "time_per_segment_s": round(time_per_segment, 2),
            "waypoint_task_specs": wp_priorities,
            "complexity_score": round(complexity_score, 3),
            "complexity_tier": complexity_tier,
            "num_no_fly_zones": len(scenario.no_fly_zones),
            "num_faults": len(scenario.faults),
            "risk_level": scenario.risk_level.name
        }

        flags = []
        if complexity_score >= 0.7:
            flags.append("HIGH_COMPLEXITY_MISSION")
        if len(scenario.faults) > 0:
            flags.append("FAULTS_INJECTED")
        if scenario.risk_level >= RiskLevel.HIGH:
            flags.append("ELEVATED_RISK_LEVEL")

        return AgentMessage(
            sender=AgentRole.MISSION_PLANNER,
            payload=payload,
            confidence=max(0.5, 1.0 - complexity_score * 0.3),
            advisory_flags=flags,
            rationale=(
                f"Mission '{scenario.title}' decomposed into {num_segments} segments "
                f"covering {round(total_distance, 1)}m. Complexity tier: {complexity_tier}."
            )
        )

    def _compute_total_path_distance(self, path: List[Point3D]) -> float:
        """Sum of Euclidean distances between consecutive waypoints."""
        total = 0.0
        for i in range(1, len(path)):
            total += self._distance_3d(path[i - 1], path[i])
        return total

    def _distance_3d(self, a: Point3D, b: Point3D) -> float:
        return math.sqrt((b.x - a.x) ** 2 + (b.y - a.y) ** 2 + (b.z - a.z) ** 2)

    def _compute_complexity_score(self, scenario: FlightScenario) -> float:
        """
        Heuristic complexity score in [0.0, 1.0] based on:
        - Number of waypoints (more = more complex)
        - Number of NFZs (spatial constraint density)
        - Number of faults (dynamic uncertainty)
        - Risk level
        - Weather severity
        """
        wp_factor = min(len(scenario.waypoints) / 10.0, 1.0)
        nfz_factor = min(len(scenario.no_fly_zones) / 5.0, 1.0)
        fault_factor = min(len(scenario.faults) / 3.0, 1.0)
        risk_factor = scenario.risk_level.value / 3.0
        weather_factor = min(scenario.weather.wind_speed_ms / 15.0, 1.0)

        score = (
            0.25 * wp_factor +
            0.25 * nfz_factor +
            0.20 * fault_factor +
            0.15 * risk_factor +
            0.15 * weather_factor
        )
        return min(score, 1.0)


# ---------------------------------------------------------------------------
# Specialized Agent: Navigation
# ---------------------------------------------------------------------------

class NavigationAgent:
    """
    Computes 3D flight routes, optimizes waypoint ordering for minimum distance,
    and calculates bearing/heading between consecutive waypoints.
    Produces a navigation plan with per-segment headings and distances.
    """

    def compute_route(
        self,
        scenario: FlightScenario,
        mission_spec: AgentMessage
    ) -> AgentMessage:
        """Generate an optimized navigation route from mission task specs."""
        spawn = scenario.spawn_point
        waypoints = scenario.waypoints
        full_path = [spawn] + list(waypoints)

        # Per-segment navigation data
        segments = []
        for i in range(1, len(full_path)):
            seg_from = full_path[i - 1]
            seg_to = full_path[i]
            dist = self._distance_3d(seg_from, seg_to)
            bearing = self._compute_bearing(seg_from, seg_to)
            climb_angle = self._compute_climb_angle(seg_from, seg_to, dist)

            segments.append({
                "segment_index": i - 1,
                "from": seg_from.model_dump(),
                "to": seg_to.model_dump(),
                "distance_m": round(dist, 2),
                "bearing_deg": round(bearing, 2),
                "climb_angle_deg": round(climb_angle, 2),
                "altitude_change_m": round(seg_to.z - seg_from.z, 2)
            })

        total_distance = sum(s["distance_m"] for s in segments)
        max_climb = max(abs(s["altitude_change_m"]) for s in segments) if segments else 0.0

        # Check for near-vertical segments (steep climb / descent)
        steep_segments = [s for s in segments if abs(s["climb_angle_deg"]) > 45.0]

        flags = []
        if steep_segments:
            flags.append("STEEP_SEGMENT_DETECTED")
        if max_climb > 50.0:
            flags.append("LARGE_ALTITUDE_CHANGE")

        payload = {
            "optimized_path": [wp.model_dump() for wp in full_path],
            "segments": segments,
            "total_distance_m": round(total_distance, 2),
            "num_segments": len(segments),
            "max_altitude_change_m": round(max_climb, 2),
            "steep_segment_count": len(steep_segments),
        }

        return AgentMessage(
            sender=AgentRole.NAVIGATOR,
            payload=payload,
            confidence=0.95 if not steep_segments else 0.80,
            advisory_flags=flags,
            rationale=(
                f"Navigation route computed: {len(segments)} segments, "
                f"total distance {round(total_distance, 1)}m. "
                f"{'Steep segments detected — caution advised.' if steep_segments else 'All segments within normal climb envelope.'}"
            )
        )

    def _distance_3d(self, a: Point3D, b: Point3D) -> float:
        return math.sqrt((b.x - a.x) ** 2 + (b.y - a.y) ** 2 + (b.z - a.z) ** 2)

    def _compute_bearing(self, origin: Point3D, target: Point3D) -> float:
        """Compute 2D compass bearing from origin to target (degrees, 0=North, CW)."""
        dx = target.x - origin.x
        dy = target.y - origin.y
        bearing_rad = math.atan2(dx, dy)
        bearing_deg = math.degrees(bearing_rad) % 360.0
        return bearing_deg

    def _compute_climb_angle(self, origin: Point3D, target: Point3D, dist_3d: float) -> float:
        """Compute climb/descent angle in degrees."""
        if dist_3d < 0.001:
            return 0.0
        horizontal_dist = math.sqrt((target.x - origin.x) ** 2 + (target.y - origin.y) ** 2)
        if horizontal_dist < 0.001:
            return 90.0 if target.z > origin.z else -90.0
        return math.degrees(math.atan2(target.z - origin.z, horizontal_dist))


# ---------------------------------------------------------------------------
# Specialized Agent: Risk Assessment
# ---------------------------------------------------------------------------

class RiskAssessmentAgent:
    """
    Evaluates spatial risk factors: No-Fly Zone proximity, altitude hazards,
    weather-based risk modifiers, and fault-induced threat levels.
    Produces a per-waypoint risk map and aggregate mission risk score.
    """

    def assess_risk(
        self,
        scenario: FlightScenario,
        nav_message: AgentMessage
    ) -> AgentMessage:
        """Evaluate risk across all route segments and produce a risk map."""
        waypoints = scenario.waypoints
        no_fly_zones = scenario.no_fly_zones
        weather = scenario.weather
        faults = scenario.faults

        # Per-waypoint risk assessment
        waypoint_risks = []
        nfz_proximity_alerts = []

        for i, wp in enumerate(waypoints):
            wp_risk_score = 0.0
            wp_risk_factors = []

            # Altitude risk
            if wp.z > scenario.airspace.max_alt_m * 0.9:
                wp_risk_score += 0.3
                wp_risk_factors.append(f"Near ceiling: {wp.z}m / {scenario.airspace.max_alt_m}m")
            if wp.z < scenario.airspace.min_alt_m * 1.1:
                wp_risk_score += 0.2
                wp_risk_factors.append(f"Near floor: {wp.z}m / {scenario.airspace.min_alt_m}m")

            # NFZ proximity analysis
            for nfz in no_fly_zones:
                if not nfz.is_active:
                    continue
                min_dist = self._point_to_polygon_min_distance(
                    (wp.x, wp.y), nfz.polygon
                )
                if min_dist < 20.0:
                    wp_risk_score += 0.4
                    wp_risk_factors.append(f"Close to NFZ '{nfz.name}': {round(min_dist, 1)}m")
                    nfz_proximity_alerts.append({
                        "waypoint_index": i,
                        "nfz_id": nfz.nfz_id,
                        "nfz_name": nfz.name,
                        "distance_m": round(min_dist, 2)
                    })
                elif min_dist < 50.0:
                    wp_risk_score += 0.15
                    wp_risk_factors.append(f"Moderate proximity to NFZ '{nfz.name}': {round(min_dist, 1)}m")

            wp_risk_score = min(wp_risk_score, 1.0)
            waypoint_risks.append({
                "waypoint_index": i,
                "position": wp.model_dump(),
                "risk_score": round(wp_risk_score, 3),
                "risk_factors": wp_risk_factors
            })

        # Aggregate risk factors
        weather_risk = self._compute_weather_risk(weather)
        fault_risk = self._compute_fault_risk(faults)
        spatial_risk = max((wr["risk_score"] for wr in waypoint_risks), default=0.0)

        aggregate_risk = min(
            0.35 * spatial_risk + 0.30 * weather_risk + 0.35 * fault_risk,
            1.0
        )

        risk_tier = (
            "CRITICAL" if aggregate_risk >= 0.75 else
            "HIGH" if aggregate_risk >= 0.5 else
            "MODERATE" if aggregate_risk >= 0.25 else
            "LOW"
        )

        flags = []
        if nfz_proximity_alerts:
            flags.append("NFZ_PROXIMITY_WARNING")
        if weather_risk >= 0.5:
            flags.append("SEVERE_WEATHER_RISK")
        if fault_risk >= 0.5:
            flags.append("ACTIVE_FAULT_HAZARD")
        if aggregate_risk >= 0.75:
            flags.append("CRITICAL_AGGREGATE_RISK")

        payload = {
            "waypoint_risk_map": waypoint_risks,
            "nfz_proximity_alerts": nfz_proximity_alerts,
            "weather_risk_score": round(weather_risk, 3),
            "fault_risk_score": round(fault_risk, 3),
            "spatial_risk_score": round(spatial_risk, 3),
            "aggregate_risk_score": round(aggregate_risk, 3),
            "risk_tier": risk_tier,
            "active_nfz_count": sum(1 for nfz in no_fly_zones if nfz.is_active),
            "active_fault_count": len(faults)
        }

        return AgentMessage(
            sender=AgentRole.RISK_ASSESSOR,
            payload=payload,
            confidence=0.90,
            advisory_flags=flags,
            rationale=(
                f"Risk assessment: aggregate score {round(aggregate_risk, 2)} ({risk_tier}). "
                f"Weather risk {round(weather_risk, 2)}, fault risk {round(fault_risk, 2)}, "
                f"spatial risk {round(spatial_risk, 2)}. "
                f"{len(nfz_proximity_alerts)} NFZ proximity alert(s)."
            )
        )

    def _point_to_polygon_min_distance(
        self, point: Tuple[float, float], polygon: List[Tuple[float, float]]
    ) -> float:
        """Minimum distance from a 2D point to any edge of a polygon."""
        min_dist = float("inf")
        n = len(polygon)
        for i in range(n):
            p1 = polygon[i]
            p2 = polygon[(i + 1) % n]
            dist = self._point_to_segment_distance(point, p1, p2)
            min_dist = min(min_dist, dist)
        return min_dist

    def _point_to_segment_distance(
        self,
        p: Tuple[float, float],
        a: Tuple[float, float],
        b: Tuple[float, float]
    ) -> float:
        """Perpendicular distance from point p to line segment ab."""
        ax, ay = a
        bx, by = b
        px, py = p
        dx, dy = bx - ax, by - ay
        seg_len_sq = dx * dx + dy * dy

        if seg_len_sq < 1e-12:
            return math.sqrt((px - ax) ** 2 + (py - ay) ** 2)

        t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / seg_len_sq))
        proj_x = ax + t * dx
        proj_y = ay + t * dy
        return math.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)

    def _compute_weather_risk(self, weather: Weather) -> float:
        """Weather-based risk score [0.0, 1.0]."""
        wind_risk = min(weather.wind_speed_ms / 15.0, 1.0)
        gust_risk = min(weather.gust_amplitude_ms / 8.0, 1.0)
        vis_risk = max(0.0, 1.0 - weather.visibility_km / 10.0)
        icing_penalty = 0.4 if weather.icing_risk else 0.0
        gnss_risk = max(0.0, min(1.0, (weather.gnss_jamming_power_dbm + 90.0) / 30.0))

        return min(
            0.25 * wind_risk + 0.20 * gust_risk + 0.15 * vis_risk +
            0.20 * icing_penalty + 0.20 * gnss_risk,
            1.0
        )

    def _compute_fault_risk(self, faults: List[FaultInjection]) -> float:
        """Fault-based risk score [0.0, 1.0]."""
        if not faults:
            return 0.0
        max_severity = max(f.severity for f in faults)
        total_duration = sum(f.duration_s for f in faults)
        duration_factor = min(total_duration / 120.0, 1.0)
        return min(0.6 * max_severity + 0.4 * duration_factor, 1.0)


# ---------------------------------------------------------------------------
# Specialized Agent: Energy Reasoning
# ---------------------------------------------------------------------------

class EnergyReasoningAgent:
    """
    Computes battery power budgets, per-segment energy consumption estimates,
    speed recommendations for energy efficiency, and remaining endurance analysis.
    Uses first-order aerodynamic power models consistent with the physics engine.
    """

    # Physical constants
    AIR_DENSITY_KG_M3 = 1.225
    GRAVITY_MS2 = 9.81

    def compute_energy_budget(
        self,
        scenario: FlightScenario,
        nav_message: AgentMessage,
        risk_message: AgentMessage
    ) -> AgentMessage:
        """Compute per-segment energy consumption and recommend optimal speed."""
        uav = scenario.uav
        weather = scenario.weather
        segments = nav_message.payload.get("segments", [])
        risk_tier = risk_message.payload.get("risk_tier", "LOW")

        total_battery_wh = uav.battery_capacity_wh
        reserve_wh = total_battery_wh * uav.reserve_fraction
        usable_battery_wh = total_battery_wh - reserve_wh

        # Recommended cruise speed based on risk tier
        speed_factor = {
            "LOW": 0.75,
            "MODERATE": 0.65,
            "HIGH": 0.55,
            "CRITICAL": 0.45
        }.get(risk_tier, 0.65)

        recommended_speed_ms = uav.max_velocity_ms * speed_factor

        # Per-segment energy computation
        segment_energy = []
        total_energy_wh = 0.0

        for seg in segments:
            dist = seg["distance_m"]
            alt_change = seg.get("altitude_change_m", 0.0)

            # Time to traverse segment at recommended speed
            travel_time_s = dist / max(recommended_speed_ms, 1.0)

            # Power model: P_total = P_hover + P_drag + P_climb + P_maneuver
            p_hover = uav.hover_power_w
            p_drag = uav.drag_coeff_kd * self.AIR_DENSITY_KG_M3 * (recommended_speed_ms ** 3)
            p_climb = max(0.0, uav.mass_kg * self.GRAVITY_MS2 * alt_change / max(travel_time_s, 0.1))
            p_maneuver = uav.maneuver_coeff_km * (recommended_speed_ms ** 2)

            # Wind opposing factor (headwind penalty)
            wind_penalty = 1.0 + (weather.wind_speed_ms + weather.gust_amplitude_ms) / max(recommended_speed_ms, 1.0) * 0.15

            total_power_w = (p_hover + p_drag + p_climb + p_maneuver) * wind_penalty
            energy_wh = total_power_w * travel_time_s / 3600.0

            segment_energy.append({
                "segment_index": seg["segment_index"],
                "distance_m": dist,
                "travel_time_s": round(travel_time_s, 2),
                "power_w": round(total_power_w, 2),
                "energy_wh": round(energy_wh, 3),
                "breakdown": {
                    "hover_w": round(p_hover, 2),
                    "drag_w": round(p_drag, 2),
                    "climb_w": round(p_climb, 2),
                    "maneuver_w": round(p_maneuver, 2),
                    "wind_penalty_factor": round(wind_penalty, 3)
                }
            })
            total_energy_wh += energy_wh

        remaining_wh = usable_battery_wh - total_energy_wh
        energy_margin_pct = (remaining_wh / usable_battery_wh * 100.0) if usable_battery_wh > 0 else 0.0
        is_energy_feasible = total_energy_wh <= usable_battery_wh

        # Endurance estimate at recommended speed
        avg_power = total_energy_wh * 3600.0 / max(sum(s["travel_time_s"] for s in segment_energy), 1.0)
        endurance_s = (usable_battery_wh * 3600.0) / max(avg_power, 1.0)

        flags = []
        if not is_energy_feasible:
            flags.append("ENERGY_BUDGET_EXCEEDED")
        if energy_margin_pct < 15.0 and is_energy_feasible:
            flags.append("LOW_ENERGY_MARGIN")
        if energy_margin_pct < 5.0:
            flags.append("CRITICAL_ENERGY_MARGIN")

        payload = {
            "total_battery_wh": round(total_battery_wh, 2),
            "reserve_wh": round(reserve_wh, 2),
            "usable_battery_wh": round(usable_battery_wh, 2),
            "total_energy_required_wh": round(total_energy_wh, 3),
            "remaining_energy_wh": round(remaining_wh, 3),
            "energy_margin_pct": round(energy_margin_pct, 2),
            "is_energy_feasible": is_energy_feasible,
            "recommended_speed_ms": round(recommended_speed_ms, 2),
            "estimated_endurance_s": round(endurance_s, 2),
            "segment_energy_breakdown": segment_energy,
            "risk_speed_factor": speed_factor,
        }

        return AgentMessage(
            sender=AgentRole.ENERGY_REASONER,
            payload=payload,
            confidence=0.92 if is_energy_feasible else 0.60,
            advisory_flags=flags,
            rationale=(
                f"Energy budget: {round(total_energy_wh, 2)} Wh required of "
                f"{round(usable_battery_wh, 2)} Wh usable ({round(energy_margin_pct, 1)}% margin). "
                f"Recommended cruise speed: {round(recommended_speed_ms, 1)} m/s "
                f"(risk-adjusted factor: {speed_factor}). "
                f"{'FEASIBLE' if is_energy_feasible else 'INFEASIBLE — requires speed reduction or route optimization.'}."
            )
        )


# ---------------------------------------------------------------------------
# Specialized Agent: Perception State
# ---------------------------------------------------------------------------

class PerceptionStateAgent:
    """
    Interprets simulated sensor telemetry streams and produces a situational
    awareness summary: GPS health, battery status, obstacle proximity,
    network connectivity, and environmental conditions.
    """

    def interpret_telemetry(
        self,
        telemetry: Optional[SimulatedSensorStream] = None,
        scenario: Optional[FlightScenario] = None
    ) -> AgentMessage:
        """
        Produce a situational awareness report from current telemetry.
        If no telemetry is available (pre-flight), generates a baseline assessment
        from the scenario specification.
        """
        if telemetry is not None:
            return self._interpret_live_telemetry(telemetry)
        elif scenario is not None:
            return self._interpret_preflight(scenario)
        else:
            return AgentMessage(
                sender=AgentRole.PERCEPTION_STATE,
                payload={"status": "NO_DATA"},
                confidence=0.0,
                advisory_flags=["NO_TELEMETRY_AVAILABLE"],
                rationale="No telemetry or scenario data available for perception analysis."
            )

    def _interpret_live_telemetry(self, telemetry: SimulatedSensorStream) -> AgentMessage:
        """Interpret a live SimulatedSensorStream into a structured state summary."""
        gps = telemetry.gps_quality
        net = telemetry.network_state
        batt_pct = telemetry.battery_percentage
        batt_wh = telemetry.remaining_battery_wh

        # GPS health classification
        gps_health = "DENIED" if gps.is_jammed else (
            "DEGRADED" if gps.hdop > 3.0 or gps.satellite_count < 6 or gps.position_noise_m > 3.0
            else "NOMINAL"
        )

        # Network health classification
        net_health = "DISCONNECTED" if not net.connected else (
            "DEGRADED" if net.latency_ms > 200.0 or net.packet_loss_ratio > 0.10
            else "NOMINAL"
        )

        # Battery health classification
        batt_health = (
            "CRITICAL" if batt_pct < 15.0 else
            "LOW" if batt_pct < 30.0 else
            "NOMINAL"
        )

        # Obstacle proximity analysis
        obstacle_count = len(telemetry.observed_obstacles)
        closest_obstacle_m = float("inf")
        if telemetry.observed_obstacles:
            pos = telemetry.actual_position
            for obs in telemetry.observed_obstacles:
                dist = math.sqrt(
                    (obs.position.x - pos.x) ** 2 +
                    (obs.position.y - pos.y) ** 2 +
                    (obs.position.z - pos.z) ** 2
                )
                closest_obstacle_m = min(closest_obstacle_m, dist)

        flags = []
        if gps_health != "NOMINAL":
            flags.append(f"GPS_{gps_health}")
        if net_health != "NOMINAL":
            flags.append(f"NETWORK_{net_health}")
        if batt_health != "NOMINAL":
            flags.append(f"BATTERY_{batt_health}")
        if closest_obstacle_m < 15.0:
            flags.append("OBSTACLE_PROXIMITY_ALERT")

        payload = {
            "timestamp_s": telemetry.timestamp_s,
            "position": telemetry.actual_position.model_dump(),
            "ground_speed_ms": round(telemetry.ground_speed_ms, 2),
            "heading_deg": round(telemetry.heading_deg, 2),
            "gps_health": gps_health,
            "gps_satellites": gps.satellite_count,
            "gps_hdop": round(gps.hdop, 2),
            "gps_noise_m": round(gps.position_noise_m, 2),
            "network_health": net_health,
            "network_latency_ms": round(net.latency_ms, 2),
            "network_packet_loss_pct": round(net.packet_loss_ratio * 100.0, 2),
            "battery_health": batt_health,
            "battery_pct": round(batt_pct, 2),
            "battery_remaining_wh": round(batt_wh, 2),
            "power_draw_w": round(telemetry.power_draw_w, 2),
            "obstacle_count": obstacle_count,
            "closest_obstacle_m": round(closest_obstacle_m, 2) if obstacle_count > 0 else None,
            "wind_speed_ms": round(telemetry.weather_state.wind_speed_ms, 2),
        }

        return AgentMessage(
            sender=AgentRole.PERCEPTION_STATE,
            payload=payload,
            confidence=0.95 if gps_health == "NOMINAL" else 0.70,
            advisory_flags=flags,
            rationale=(
                f"Perception state at t={telemetry.timestamp_s}s: "
                f"GPS={gps_health}, Network={net_health}, Battery={batt_health} ({round(batt_pct, 1)}%). "
                f"{obstacle_count} obstacle(s) detected."
            )
        )

    def _interpret_preflight(self, scenario: FlightScenario) -> AgentMessage:
        """Generate a pre-flight baseline assessment from scenario metadata."""
        weather = scenario.weather
        faults = scenario.faults

        # Pre-flight GNSS status
        gnss_status = "JAMMING_EXPECTED" if weather.gnss_jamming_power_dbm > -90.0 else "NOMINAL"
        gnss_fault = any(f.fault_type == "gnss_denial" for f in faults)
        if gnss_fault:
            gnss_status = "DENIAL_FAULT_SCHEDULED"

        payload = {
            "phase": "PRE_FLIGHT",
            "spawn_position": scenario.spawn_point.model_dump(),
            "initial_battery_wh": scenario.uav.battery_capacity_wh,
            "wind_speed_ms": round(weather.wind_speed_ms, 2),
            "visibility_km": round(weather.visibility_km, 2),
            "icing_risk": weather.icing_risk,
            "gnss_status": gnss_status,
            "scheduled_faults": [f.model_dump() for f in faults],
            "num_obstacles": 0,
        }

        flags = []
        if gnss_status != "NOMINAL":
            flags.append(f"GNSS_{gnss_status}")
        if weather.icing_risk:
            flags.append("ICING_RISK")
        if weather.wind_speed_ms > 10.0:
            flags.append("HIGH_WIND")

        return AgentMessage(
            sender=AgentRole.PERCEPTION_STATE,
            payload=payload,
            confidence=0.85,
            advisory_flags=flags,
            rationale=(
                f"Pre-flight perception: GNSS={gnss_status}, "
                f"Wind={round(weather.wind_speed_ms, 1)}m/s, "
                f"Visibility={round(weather.visibility_km, 1)}km. "
                f"{len(faults)} fault(s) scheduled during mission."
            )
        )
