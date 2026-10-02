from enum import Enum
from typing import Tuple, Dict, Any, Optional
from uav_neurosym.schema import FlightScenario, SimulatedSensorStream, RiskLevel
from uav_neurosym.memory.temporal_memory import TemporalMemoryManager


class IntelligenceStrategy(str, Enum):
    LIGHTWEIGHT_LOCAL_SLM = "lightweight_local_slm"
    STRONG_CLOUD_LLM = "strong_cloud_llm"
    DETERMINISTIC_FALLBACK = "deterministic_fallback"
    EMERGENCY_DETERMINISTIC_CONTROLLER = "emergency_deterministic_controller"


class AdaptiveStrategySelector:
    """
    Adaptive AI & Control Strategy Selector for NeuroSym platform.
    Dynamically evaluates state parameters and selects the optimal intelligence strategy:
    - Low Risk + Nominal Battery + Stable Network -> LIGHTWEIGHT_LOCAL_SLM (Low cost, fast)
    - High Complexity / Re-planning Needed -> STRONG_CLOUD_LLM (High reasoning)
    - Network Outage / High Packet Loss -> DETERMINISTIC_FALLBACK (No cloud reliance)
    - Severe Fault / Critical Risk Event -> EMERGENCY_DETERMINISTIC_CONTROLLER (Hard safety)
    """

    def select_strategy(
        self,
        scenario: FlightScenario,
        telemetry: SimulatedSensorStream,
        memory: Optional[TemporalMemoryManager] = None
    ) -> Tuple[IntelligenceStrategy, str]:
        """
        Selects optimal IntelligenceStrategy and provides decision rationale.
        """
        net = telemetry.network_state
        gps = telemetry.gps_quality
        batt_pct = telemetry.battery_percentage

        # 1. Emergency Safety Override Trigger
        if gps.is_jammed or scenario.risk_level == RiskLevel.CRITICAL:
            return (
                IntelligenceStrategy.EMERGENCY_DETERMINISTIC_CONTROLLER,
                "Emergency strategy selected: Severe GNSS denial fault or critical risk event active."
            )

        if batt_pct < 20.0:
            return (
                IntelligenceStrategy.EMERGENCY_DETERMINISTIC_CONTROLLER,
                f"Emergency strategy selected: Low battery percentage ({batt_pct}%)."
            )

        # 2. Network Outage / Loss Trigger -> Deterministic Fallback
        if not net.connected or net.packet_loss_ratio > 0.15 or net.latency_ms > 250.0:
            return (
                IntelligenceStrategy.DETERMINISTIC_FALLBACK,
                f"Deterministic fallback selected: Network degraded (latency={net.latency_ms}ms, loss={int(net.packet_loss_ratio*100)}%)."
            )

        # 3. High Risk / Complex Scenario -> Strong Cloud LLM
        if scenario.risk_level >= RiskLevel.HIGH or len(scenario.no_fly_zones) > 0:
            return (
                IntelligenceStrategy.STRONG_CLOUD_LLM,
                "Strong cloud LLM selected: High risk level or complex No-Fly Zone geometry."
            )

        # 4. Default Nominal Case -> Lightweight Local SLM
        return (
            IntelligenceStrategy.LIGHTWEIGHT_LOCAL_SLM,
            "Lightweight local SLM selected: Nominal operating conditions, low risk, stable network."
        )
