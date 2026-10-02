"""
Test Suite — Phase 6: Specialized Agent Mesh Architecture.

Validates:
- Each specialized agent produces correctly structured AgentMessage outputs.
- Message passing between agents in the mesh pipeline works correctly.
- MeshController synthesizes valid ActionProposal from multi-agent consensus.
- Advisory flags and confidence scores propagate correctly.
- Reflection feedback (guardrail rejection) adjusts the synthesized proposal.
"""

import pytest
from uav_neurosym.schema import (
    FlightScenario, UAVConfig, Weather, PolygonGeofence, Point3D,
    NoFlyZone, FaultInjection, ActionProposal, RiskLevel, MissionType,
    SimulatedSensorStream, GPSQuality, NetworkState
)
from uav_neurosym.mesh.specialized_agents import (
    MissionPlanningAgent,
    NavigationAgent,
    RiskAssessmentAgent,
    EnergyReasoningAgent,
    PerceptionStateAgent,
    AgentMessage,
    AgentRole,
)
from uav_neurosym.mesh.mesh_controller import MultiAgentMeshController


# ---------------------------------------------------------------------------
# Test Fixtures
# ---------------------------------------------------------------------------

def _make_nominal_scenario() -> FlightScenario:
    """Create a nominal low-risk test scenario."""
    return FlightScenario(
        scenario_id="MESH-TEST-001",
        title="Nominal Delivery Mission",
        description="Low-risk package delivery across open airspace.",
        mission_type=MissionType.DELIVERY,
        uav=UAVConfig(
            uav_id="UAV-MESH-01",
            mass_kg=3.5,
            battery_capacity_wh=120.0,
            reserve_fraction=0.20,
            max_velocity_ms=18.0,
            hover_power_w=180.0
        ),
        weather=Weather(
            wind_speed_ms=4.0,
            gust_amplitude_ms=1.0,
            visibility_km=10.0,
            icing_risk=False,
            gnss_jamming_power_dbm=-120.0
        ),
        airspace=PolygonGeofence(
            name="Test Airspace",
            vertices=[(-200, -200), (600, -200), (600, 600), (-200, 600)],
            min_alt_m=10.0,
            max_alt_m=120.0
        ),
        spawn_point=Point3D(x=0.0, y=0.0, z=30.0),
        waypoints=[
            Point3D(x=100.0, y=100.0, z=50.0),
            Point3D(x=200.0, y=150.0, z=55.0),
            Point3D(x=300.0, y=200.0, z=50.0),
        ],
        time_budget_s=300.0,
        risk_level=RiskLevel.LOW,
    )


def _make_complex_scenario() -> FlightScenario:
    """Create a high-risk scenario with NFZs and faults for thorough testing."""
    return FlightScenario(
        scenario_id="MESH-TEST-002",
        title="High-Risk Reconnaissance Under Faults",
        description="Reconnaissance mission through NFZ-dense airspace with GNSS jamming.",
        mission_type=MissionType.RECONNAISSANCE,
        uav=UAVConfig(
            uav_id="UAV-MESH-02",
            mass_kg=4.0,
            battery_capacity_wh=90.0,
            reserve_fraction=0.20,
            max_velocity_ms=15.0,
            hover_power_w=200.0
        ),
        weather=Weather(
            wind_speed_ms=11.0,
            gust_amplitude_ms=4.0,
            visibility_km=3.0,
            icing_risk=True,
            gnss_jamming_power_dbm=-85.0
        ),
        airspace=PolygonGeofence(
            name="Urban Airspace",
            vertices=[(-100, -100), (500, -100), (500, 500), (-100, 500)],
            min_alt_m=15.0,
            max_alt_m=100.0
        ),
        spawn_point=Point3D(x=0.0, y=0.0, z=25.0),
        waypoints=[
            Point3D(x=80.0, y=80.0, z=45.0),
            Point3D(x=160.0, y=200.0, z=60.0),
            Point3D(x=300.0, y=350.0, z=50.0),
        ],
        no_fly_zones=[
            NoFlyZone(
                nfz_id="NFZ-A",
                name="Hospital Helipad",
                polygon=[(90, 90), (130, 90), (130, 130), (90, 130)],
                min_alt_m=0.0,
                max_alt_m=150.0,
                is_active=True
            ),
        ],
        faults=[
            FaultInjection(
                fault_type="gnss_denial",
                start_time_s=30.0,
                duration_s=60.0,
                severity=0.8
            ),
        ],
        time_budget_s=250.0,
        risk_level=RiskLevel.HIGH,
    )


def _make_live_telemetry() -> SimulatedSensorStream:
    """Create a sample live telemetry stream."""
    return SimulatedSensorStream(
        timestamp_s=45.0,
        estimated_position=Point3D(x=100.0, y=100.0, z=50.0),
        actual_position=Point3D(x=101.0, y=99.5, z=50.2),
        ground_speed_ms=10.5,
        heading_deg=45.0,
        remaining_battery_wh=85.0,
        battery_percentage=70.8,
        power_draw_w=200.0,
        gps_quality=GPSQuality(
            is_jammed=False,
            satellite_count=10,
            hdop=1.5,
            position_noise_m=1.0
        ),
        network_state=NetworkState(
            connected=True,
            latency_ms=30.0,
            bandwidth_mbps=8.0,
            packet_loss_ratio=0.02
        ),
    )


# ---------------------------------------------------------------------------
# Test: Mission Planning Agent
# ---------------------------------------------------------------------------

class TestMissionPlanningAgent:
    def test_nominal_mission_analysis(self):
        agent = MissionPlanningAgent()
        scenario = _make_nominal_scenario()
        msg = agent.analyze(scenario)

        assert isinstance(msg, AgentMessage)
        assert msg.sender == AgentRole.MISSION_PLANNER
        assert msg.confidence > 0.0
        assert msg.payload["total_waypoints"] == 3
        assert msg.payload["time_budget_s"] == 300.0
        assert msg.payload["total_path_distance_m"] > 0.0
        assert len(msg.payload["waypoint_task_specs"]) == 3

    def test_complex_mission_flags(self):
        agent = MissionPlanningAgent()
        scenario = _make_complex_scenario()
        msg = agent.analyze(scenario)

        assert msg.payload["num_no_fly_zones"] == 1
        assert msg.payload["num_faults"] == 1
        assert msg.payload["risk_level"] == "HIGH"
        assert "ELEVATED_RISK_LEVEL" in msg.advisory_flags or "FAULTS_INJECTED" in msg.advisory_flags

    def test_complexity_score_range(self):
        agent = MissionPlanningAgent()
        scenario = _make_nominal_scenario()
        msg = agent.analyze(scenario)

        score = msg.payload["complexity_score"]
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Test: Navigation Agent
# ---------------------------------------------------------------------------

class TestNavigationAgent:
    def test_route_computation(self):
        mission_agent = MissionPlanningAgent()
        nav_agent = NavigationAgent()
        scenario = _make_nominal_scenario()

        mission_msg = mission_agent.analyze(scenario)
        nav_msg = nav_agent.compute_route(scenario, mission_msg)

        assert isinstance(nav_msg, AgentMessage)
        assert nav_msg.sender == AgentRole.NAVIGATOR
        assert nav_msg.payload["num_segments"] == 3
        assert nav_msg.payload["total_distance_m"] > 0.0
        assert len(nav_msg.payload["segments"]) == 3

    def test_segment_data_structure(self):
        mission_agent = MissionPlanningAgent()
        nav_agent = NavigationAgent()
        scenario = _make_nominal_scenario()

        mission_msg = mission_agent.analyze(scenario)
        nav_msg = nav_agent.compute_route(scenario, mission_msg)

        for seg in nav_msg.payload["segments"]:
            assert "segment_index" in seg
            assert "distance_m" in seg
            assert "bearing_deg" in seg
            assert "climb_angle_deg" in seg
            assert seg["distance_m"] > 0.0

    def test_bearing_range(self):
        mission_agent = MissionPlanningAgent()
        nav_agent = NavigationAgent()
        scenario = _make_nominal_scenario()

        mission_msg = mission_agent.analyze(scenario)
        nav_msg = nav_agent.compute_route(scenario, mission_msg)

        for seg in nav_msg.payload["segments"]:
            assert 0.0 <= seg["bearing_deg"] < 360.0


# ---------------------------------------------------------------------------
# Test: Risk Assessment Agent
# ---------------------------------------------------------------------------

class TestRiskAssessmentAgent:
    def test_low_risk_scenario(self):
        mission_agent = MissionPlanningAgent()
        nav_agent = NavigationAgent()
        risk_agent = RiskAssessmentAgent()
        scenario = _make_nominal_scenario()

        mission_msg = mission_agent.analyze(scenario)
        nav_msg = nav_agent.compute_route(scenario, mission_msg)
        risk_msg = risk_agent.assess_risk(scenario, nav_msg)

        assert isinstance(risk_msg, AgentMessage)
        assert risk_msg.sender == AgentRole.RISK_ASSESSOR
        assert risk_msg.payload["risk_tier"] in ["LOW", "MODERATE"]
        assert risk_msg.payload["aggregate_risk_score"] < 0.5

    def test_high_risk_with_nfz(self):
        mission_agent = MissionPlanningAgent()
        nav_agent = NavigationAgent()
        risk_agent = RiskAssessmentAgent()
        scenario = _make_complex_scenario()

        mission_msg = mission_agent.analyze(scenario)
        nav_msg = nav_agent.compute_route(scenario, mission_msg)
        risk_msg = risk_agent.assess_risk(scenario, nav_msg)

        assert risk_msg.payload["active_nfz_count"] == 1
        assert risk_msg.payload["active_fault_count"] == 1
        assert len(risk_msg.advisory_flags) > 0

    def test_risk_score_range(self):
        risk_agent = RiskAssessmentAgent()
        mission_agent = MissionPlanningAgent()
        nav_agent = NavigationAgent()
        scenario = _make_complex_scenario()

        mission_msg = mission_agent.analyze(scenario)
        nav_msg = nav_agent.compute_route(scenario, mission_msg)
        risk_msg = risk_agent.assess_risk(scenario, nav_msg)

        assert 0.0 <= risk_msg.payload["aggregate_risk_score"] <= 1.0
        assert 0.0 <= risk_msg.payload["weather_risk_score"] <= 1.0
        assert 0.0 <= risk_msg.payload["fault_risk_score"] <= 1.0


# ---------------------------------------------------------------------------
# Test: Energy Reasoning Agent
# ---------------------------------------------------------------------------

class TestEnergyReasoningAgent:
    def test_energy_budget_computation(self):
        mission_agent = MissionPlanningAgent()
        nav_agent = NavigationAgent()
        risk_agent = RiskAssessmentAgent()
        energy_agent = EnergyReasoningAgent()
        scenario = _make_nominal_scenario()

        mission_msg = mission_agent.analyze(scenario)
        nav_msg = nav_agent.compute_route(scenario, mission_msg)
        risk_msg = risk_agent.assess_risk(scenario, nav_msg)
        energy_msg = energy_agent.compute_energy_budget(scenario, nav_msg, risk_msg)

        assert isinstance(energy_msg, AgentMessage)
        assert energy_msg.sender == AgentRole.ENERGY_REASONER
        assert energy_msg.payload["total_battery_wh"] == 120.0
        assert energy_msg.payload["reserve_wh"] == 24.0
        assert energy_msg.payload["usable_battery_wh"] == 96.0
        assert energy_msg.payload["total_energy_required_wh"] > 0.0

    def test_energy_feasibility_flag(self):
        energy_agent = EnergyReasoningAgent()
        mission_agent = MissionPlanningAgent()
        nav_agent = NavigationAgent()
        risk_agent = RiskAssessmentAgent()
        scenario = _make_nominal_scenario()

        mission_msg = mission_agent.analyze(scenario)
        nav_msg = nav_agent.compute_route(scenario, mission_msg)
        risk_msg = risk_agent.assess_risk(scenario, nav_msg)
        energy_msg = energy_agent.compute_energy_budget(scenario, nav_msg, risk_msg)

        assert isinstance(energy_msg.payload["is_energy_feasible"], bool)

    def test_segment_energy_breakdown(self):
        energy_agent = EnergyReasoningAgent()
        mission_agent = MissionPlanningAgent()
        nav_agent = NavigationAgent()
        risk_agent = RiskAssessmentAgent()
        scenario = _make_nominal_scenario()

        mission_msg = mission_agent.analyze(scenario)
        nav_msg = nav_agent.compute_route(scenario, mission_msg)
        risk_msg = risk_agent.assess_risk(scenario, nav_msg)
        energy_msg = energy_agent.compute_energy_budget(scenario, nav_msg, risk_msg)

        breakdowns = energy_msg.payload["segment_energy_breakdown"]
        assert len(breakdowns) == 3
        for seg in breakdowns:
            assert "energy_wh" in seg
            assert "power_w" in seg
            assert "breakdown" in seg
            assert seg["energy_wh"] >= 0.0


# ---------------------------------------------------------------------------
# Test: Perception State Agent
# ---------------------------------------------------------------------------

class TestPerceptionStateAgent:
    def test_preflight_perception(self):
        agent = PerceptionStateAgent()
        scenario = _make_nominal_scenario()
        msg = agent.interpret_telemetry(scenario=scenario)

        assert isinstance(msg, AgentMessage)
        assert msg.sender == AgentRole.PERCEPTION_STATE
        assert msg.payload["phase"] == "PRE_FLIGHT"
        assert msg.payload["initial_battery_wh"] == 120.0

    def test_live_telemetry_perception(self):
        agent = PerceptionStateAgent()
        telemetry = _make_live_telemetry()
        msg = agent.interpret_telemetry(telemetry=telemetry)

        assert msg.sender == AgentRole.PERCEPTION_STATE
        assert msg.payload["gps_health"] == "NOMINAL"
        assert msg.payload["network_health"] == "NOMINAL"
        assert msg.payload["battery_health"] == "NOMINAL"
        assert msg.payload["timestamp_s"] == 45.0

    def test_no_data_perception(self):
        agent = PerceptionStateAgent()
        msg = agent.interpret_telemetry()

        assert msg.confidence == 0.0
        assert "NO_TELEMETRY_AVAILABLE" in msg.advisory_flags

    def test_degraded_gps_detection(self):
        agent = PerceptionStateAgent()
        telemetry = _make_live_telemetry()
        telemetry.gps_quality.is_jammed = True
        msg = agent.interpret_telemetry(telemetry=telemetry)

        assert msg.payload["gps_health"] == "DENIED"
        assert "GPS_DENIED" in msg.advisory_flags
        assert msg.confidence < 0.90


# ---------------------------------------------------------------------------
# Test: Multi-Agent Mesh Controller
# ---------------------------------------------------------------------------

class TestMultiAgentMeshController:
    def test_full_mesh_execution_nominal(self):
        controller = MultiAgentMeshController()
        scenario = _make_nominal_scenario()
        proposal, report = controller.execute_mesh(scenario)

        assert isinstance(proposal, ActionProposal)
        assert len(proposal.proposed_path) > 0
        assert proposal.target_airspeed_ms > 0.0
        assert proposal.estimated_duration_s > 0.0
        assert proposal.emergency_action is None

        assert report["agents_executed"] == 5
        assert report["aggregate_confidence"] > 0.0

    def test_full_mesh_execution_complex(self):
        controller = MultiAgentMeshController()
        scenario = _make_complex_scenario()
        proposal, report = controller.execute_mesh(scenario)

        assert isinstance(proposal, ActionProposal)
        assert len(proposal.proposed_path) > 0
        assert report["agents_executed"] == 5
        assert len(report["all_advisory_flags"]) > 0

    def test_mesh_with_live_telemetry(self):
        controller = MultiAgentMeshController()
        scenario = _make_nominal_scenario()
        telemetry = _make_live_telemetry()
        proposal, report = controller.execute_mesh(scenario, telemetry=telemetry)

        assert isinstance(proposal, ActionProposal)
        # Perception should have used live telemetry
        perception_summary = report["agent_summaries"]["perception_state"]
        assert perception_summary["confidence"] > 0.0

    def test_mesh_with_reflection_feedback(self):
        controller = MultiAgentMeshController()
        scenario = _make_nominal_scenario()
        feedback = "SYMBOLIC REJECTION [ENERGY_EXHAUSTION]: Excess required energy: 15.2 Wh."

        proposal, report = controller.execute_mesh(
            scenario, reflection_feedback=feedback
        )

        assert isinstance(proposal, ActionProposal)
        # Speed should be reduced due to energy feedback
        assert proposal.target_airspeed_ms <= scenario.uav.max_velocity_ms * 0.8

    def test_mesh_execution_log(self):
        controller = MultiAgentMeshController()
        scenario = _make_nominal_scenario()
        controller.execute_mesh(scenario)

        log = controller.get_execution_log()
        assert len(log) == 5
        agent_names = [entry["agent"] for entry in log]
        assert "perception_state" in agent_names
        assert "mission_planner" in agent_names
        assert "navigator" in agent_names
        assert "risk_assessor" in agent_names
        assert "energy_reasoner" in agent_names

    def test_emergency_action_on_gps_denial(self):
        controller = MultiAgentMeshController()
        scenario = _make_nominal_scenario()
        # Simulate live telemetry with GPS jammed
        telemetry = _make_live_telemetry()
        telemetry.gps_quality.is_jammed = True

        proposal, report = controller.execute_mesh(scenario, telemetry=telemetry)

        assert proposal.emergency_action == "RETURN_TO_BASE"
        assert report["critical_flag_count"] > 0

    def test_mesh_report_structure(self):
        controller = MultiAgentMeshController()
        scenario = _make_nominal_scenario()
        _, report = controller.execute_mesh(scenario)

        assert "mesh_pipeline" in report
        assert "agents_executed" in report
        assert "agent_summaries" in report
        assert "aggregate_confidence" in report
        assert "all_advisory_flags" in report
        assert "proposal_summary" in report
        assert "execution_log" in report
