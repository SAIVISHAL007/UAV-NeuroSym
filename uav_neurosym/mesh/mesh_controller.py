"""
Multi-Agent Mesh Controller — Inter-Agent Coordination and Consensus Synthesis.

Orchestrates the specialized agent mesh by:
1. Dispatching scenario data to each specialized agent in dependency order.
2. Collecting structured AgentMessage outputs from all agents.
3. Aggregating advisory flags and confidence scores.
4. Synthesizing a unified ActionProposal from domain-expert recommendations.
5. Producing a detailed mesh execution report for the guardrail loop.

Execution Pipeline:
    MissionPlanningAgent ──► NavigationAgent ──►
                              │                    ──► MeshController (Synthesis)
    PerceptionStateAgent ──►  │
                              ▼
                          RiskAssessmentAgent ──► EnergyReasoningAgent ──►
"""

from typing import List, Dict, Any, Optional, Tuple
from uav_neurosym.schema import (
    FlightScenario, ActionProposal, Point3D, SimulatedSensorStream
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


class MultiAgentMeshController:
    """
    Coordinates inter-agent message passing and synthesizes specialized agent
    outputs into a unified ActionProposal.

    The controller executes the mesh pipeline in strict dependency order:
        1. PerceptionStateAgent (independent — pre-flight assessment)
        2. MissionPlanningAgent (independent — task decomposition)
        3. NavigationAgent (depends on MissionPlanningAgent output)
        4. RiskAssessmentAgent (depends on NavigationAgent output)
        5. EnergyReasoningAgent (depends on NavigationAgent + RiskAssessmentAgent)
        6. Synthesis (aggregates all agent outputs into ActionProposal)
    """

    def __init__(self):
        self.mission_planner = MissionPlanningAgent()
        self.navigator = NavigationAgent()
        self.risk_assessor = RiskAssessmentAgent()
        self.energy_reasoner = EnergyReasoningAgent()
        self.perception = PerceptionStateAgent()

        # Mesh execution history for inspection and debugging
        self.execution_log: List[Dict[str, Any]] = []

    def execute_mesh(
        self,
        scenario: FlightScenario,
        telemetry: Optional[SimulatedSensorStream] = None,
        reflection_feedback: Optional[str] = None
    ) -> Tuple[ActionProposal, Dict[str, Any]]:
        """
        Execute the full specialized agent mesh pipeline and produce a unified
        ActionProposal along with a detailed mesh execution report.

        Args:
            scenario: The flight scenario specification.
            telemetry: Optional live telemetry (None for pre-flight planning).
            reflection_feedback: Optional guardrail rejection feedback for re-planning.

        Returns:
            Tuple of (ActionProposal, mesh_report_dict)
        """
        self.execution_log = []
        agent_messages: Dict[AgentRole, AgentMessage] = {}
        all_flags: List[str] = []

        # ─── Stage 1: Perception State Assessment ───
        perception_msg = self.perception.interpret_telemetry(
            telemetry=telemetry,
            scenario=scenario
        )
        agent_messages[AgentRole.PERCEPTION_STATE] = perception_msg
        all_flags.extend(perception_msg.advisory_flags)
        self._log_agent_step("perception_state", perception_msg)

        # ─── Stage 2: Mission Planning Decomposition ───
        mission_msg = self.mission_planner.analyze(scenario)
        agent_messages[AgentRole.MISSION_PLANNER] = mission_msg
        all_flags.extend(mission_msg.advisory_flags)
        self._log_agent_step("mission_planner", mission_msg)

        # ─── Stage 3: Navigation Route Computation ───
        nav_msg = self.navigator.compute_route(scenario, mission_msg)
        agent_messages[AgentRole.NAVIGATOR] = nav_msg
        all_flags.extend(nav_msg.advisory_flags)
        self._log_agent_step("navigator", nav_msg)

        # ─── Stage 4: Risk Assessment ───
        risk_msg = self.risk_assessor.assess_risk(scenario, nav_msg)
        agent_messages[AgentRole.RISK_ASSESSOR] = risk_msg
        all_flags.extend(risk_msg.advisory_flags)
        self._log_agent_step("risk_assessor", risk_msg)

        # ─── Stage 5: Energy Reasoning ───
        energy_msg = self.energy_reasoner.compute_energy_budget(
            scenario, nav_msg, risk_msg
        )
        agent_messages[AgentRole.ENERGY_REASONER] = energy_msg
        all_flags.extend(energy_msg.advisory_flags)
        self._log_agent_step("energy_reasoner", energy_msg)

        # ─── Stage 6: Consensus Synthesis ───
        proposal = self._synthesize_proposal(
            scenario, agent_messages, all_flags, reflection_feedback
        )

        # Build mesh execution report
        mesh_report = self._build_mesh_report(agent_messages, all_flags, proposal)

        return proposal, mesh_report

    def _synthesize_proposal(
        self,
        scenario: FlightScenario,
        messages: Dict[AgentRole, AgentMessage],
        flags: List[str],
        reflection_feedback: Optional[str]
    ) -> ActionProposal:
        """
        Synthesize all agent outputs into a unified ActionProposal.

        Decision Logic:
        - Path: Use the navigation agent's optimized path (excluding spawn point).
        - Speed: Use the energy agent's recommended speed, adjusted for risk.
        - Duration: Computed from path distance and recommended speed.
        - Emergency Action: Triggered by critical flags from any agent.
        - Rationale: Aggregated narrative from all agent reasonings.
        """
        nav_msg = messages.get(AgentRole.NAVIGATOR)
        energy_msg = messages.get(AgentRole.ENERGY_REASONER)
        risk_msg = messages.get(AgentRole.RISK_ASSESSOR)
        mission_msg = messages.get(AgentRole.MISSION_PLANNER)

        # Extract optimized path from navigator (exclude spawn point at index 0)
        optimized_path_raw = nav_msg.payload.get("optimized_path", [])
        if len(optimized_path_raw) > 1:
            proposed_path = [Point3D(**wp) for wp in optimized_path_raw[1:]]
        else:
            proposed_path = list(scenario.waypoints)

        # Speed: from energy reasoner's risk-adjusted recommendation
        recommended_speed = energy_msg.payload.get(
            "recommended_speed_ms", scenario.uav.max_velocity_ms * 0.65
        )

        # Duration estimate from total distance and speed
        total_distance = nav_msg.payload.get("total_distance_m", 0.0)
        estimated_duration = total_distance / max(recommended_speed, 1.0)

        # Emergency action logic based on aggregated flags
        emergency_action = None
        critical_flags = [f for f in flags if any(
            kw in f for kw in ["CRITICAL", "DENIED", "EMERGENCY", "JAMMING", "DISCONNECTED"]
        )]
        if critical_flags:
            # Check severity: GPS denial or critical battery triggers RTB
            if any("GPS_DENIED" in f or "GNSS_DENIAL" in f for f in critical_flags):
                emergency_action = "RETURN_TO_BASE"
            elif any("BATTERY_CRITICAL" in f or "CRITICAL_ENERGY" in f for f in critical_flags):
                emergency_action = "EMERGENCY_LANDING"
            elif any("CRITICAL_AGGREGATE_RISK" in f for f in critical_flags):
                emergency_action = "RETURN_TO_BASE"

        # Apply reflection feedback adjustments
        if reflection_feedback:
            if "ENERGY_EXHAUSTION" in reflection_feedback:
                recommended_speed = max(recommended_speed * 0.7, 5.0)
                if len(proposed_path) > 2:
                    proposed_path = proposed_path[:2]
            elif "NO_FLY_ZONE" in reflection_feedback:
                proposed_path = [
                    Point3D(x=wp.x + 80.0, y=wp.y + 80.0, z=wp.z)
                    for wp in proposed_path
                ]
            elif "GNSS_DENIAL" in reflection_feedback:
                emergency_action = "RETURN_TO_BASE"

            # Recalculate duration after adjustments
            estimated_duration = self._compute_path_distance(proposed_path) / max(recommended_speed, 1.0)

        # Aggregate rationale narrative
        rationale_parts = [
            f"[MeshController] Synthesized from {len(messages)} specialized agents.",
            f"Mission: {mission_msg.rationale}",
            f"Navigation: {nav_msg.rationale}",
            f"Risk: {risk_msg.rationale}",
            f"Energy: {energy_msg.rationale}",
        ]
        if critical_flags:
            rationale_parts.append(f"Critical flags raised: {', '.join(critical_flags)}.")
        if reflection_feedback:
            rationale_parts.append(f"Reflection feedback applied: {reflection_feedback[:100]}...")

        # Compute aggregate confidence (weighted average)
        confidence_weights = {
            AgentRole.NAVIGATOR: 0.30,
            AgentRole.ENERGY_REASONER: 0.30,
            AgentRole.RISK_ASSESSOR: 0.25,
            AgentRole.MISSION_PLANNER: 0.10,
            AgentRole.PERCEPTION_STATE: 0.05,
        }
        aggregate_confidence = sum(
            messages[role].confidence * weight
            for role, weight in confidence_weights.items()
            if role in messages
        )

        return ActionProposal(
            proposed_path=proposed_path,
            target_airspeed_ms=round(recommended_speed, 2),
            estimated_duration_s=round(estimated_duration, 2),
            emergency_action=emergency_action,
            rationale=" | ".join(rationale_parts)
        )

    def _build_mesh_report(
        self,
        messages: Dict[AgentRole, AgentMessage],
        flags: List[str],
        proposal: ActionProposal
    ) -> Dict[str, Any]:
        """Build a comprehensive mesh execution report for diagnostics."""
        # Per-agent summary
        agent_summaries = {}
        for role, msg in messages.items():
            agent_summaries[role.value] = {
                "confidence": msg.confidence,
                "flags": msg.advisory_flags,
                "rationale": msg.rationale,
                "payload_keys": list(msg.payload.keys()),
            }

        # Aggregate confidence
        confidence_weights = {
            AgentRole.NAVIGATOR: 0.30,
            AgentRole.ENERGY_REASONER: 0.30,
            AgentRole.RISK_ASSESSOR: 0.25,
            AgentRole.MISSION_PLANNER: 0.10,
            AgentRole.PERCEPTION_STATE: 0.05,
        }
        aggregate_confidence = sum(
            messages[role].confidence * weight
            for role, weight in confidence_weights.items()
            if role in messages
        )

        # Unique deduplicated flags
        unique_flags = list(dict.fromkeys(flags))

        return {
            "mesh_pipeline": "specialized_agent_mesh_v1",
            "agents_executed": len(messages),
            "agent_summaries": agent_summaries,
            "aggregate_confidence": round(aggregate_confidence, 3),
            "all_advisory_flags": unique_flags,
            "critical_flag_count": len([f for f in unique_flags if "CRITICAL" in f or "DENIED" in f]),
            "proposal_summary": {
                "num_waypoints": len(proposal.proposed_path),
                "target_speed_ms": proposal.target_airspeed_ms,
                "estimated_duration_s": proposal.estimated_duration_s,
                "emergency_action": proposal.emergency_action,
            },
            "execution_log": self.execution_log,
        }

    def _log_agent_step(self, agent_name: str, message: AgentMessage):
        """Record an agent execution step in the internal log."""
        self.execution_log.append({
            "agent": agent_name,
            "confidence": message.confidence,
            "flags": message.advisory_flags,
            "rationale": message.rationale,
        })

    def _compute_path_distance(self, path: List[Point3D]) -> float:
        """Sum of 3D Euclidean distances along a path."""
        import math
        total = 0.0
        for i in range(1, len(path)):
            a, b = path[i - 1], path[i]
            total += math.sqrt(
                (b.x - a.x) ** 2 + (b.y - a.y) ** 2 + (b.z - a.z) ** 2
            )
        return total

    def get_execution_log(self) -> List[Dict[str, Any]]:
        """Return the full execution log from the last mesh pipeline run."""
        return self.execution_log
