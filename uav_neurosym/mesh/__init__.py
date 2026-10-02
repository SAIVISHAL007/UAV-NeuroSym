"""
NeuroSym Phase 6: Specialized Agent Mesh Architecture.

Decomposes monolithic LLM prompts into a collaborative network of
domain-specialized agents that communicate via structured message passing.

Agents:
    - MissionPlanningAgent: High-level mission decomposition and task specification.
    - NavigationAgent: 3D waypoint route computation and path optimization.
    - RiskAssessmentAgent: No-Fly Zone / altitude hazard evaluation and risk scoring.
    - EnergyReasoningAgent: Battery power budget and speed recommendation engine.
    - PerceptionStateAgent: Simulated sensor telemetry interpretation and state assessment.

Controller:
    - MultiAgentMeshController: Coordinates inter-agent message passing, consensus
      synthesis, and produces a unified ActionProposal.
"""

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

__all__ = [
    "MissionPlanningAgent",
    "NavigationAgent",
    "RiskAssessmentAgent",
    "EnergyReasoningAgent",
    "PerceptionStateAgent",
    "AgentMessage",
    "AgentRole",
    "MultiAgentMeshController",
]
