import pytest
from uav_neurosym.adaptive.strategy_selector import AdaptiveStrategySelector, IntelligenceStrategy
from uav_neurosym.data.loader import load_benchmark_scenarios
from uav_neurosym.schema import SimulatedSensorStream, Point3D, GPSQuality, NetworkState, Weather, RiskLevel


def test_strategy_selector_nominal():
    scenarios = load_benchmark_scenarios()
    scenario = scenarios[0]  # Nominal scenario (Risk LOW)
    selector = AdaptiveStrategySelector()

    telemetry = SimulatedSensorStream(
        timestamp_s=1.0,
        estimated_position=Point3D(x=0, y=0, z=20),
        actual_position=Point3D(x=0, y=0, z=20),
        remaining_battery_wh=140.0,
        battery_percentage=93.3
    )

    strategy, rationale = selector.select_strategy(scenario, telemetry)
    assert strategy == IntelligenceStrategy.LIGHTWEIGHT_LOCAL_SLM


def test_strategy_selector_high_risk():
    scenarios = load_benchmark_scenarios()
    scenario = scenarios[2]  # VIP NFZ Scenario (Risk HIGH)
    selector = AdaptiveStrategySelector()

    telemetry = SimulatedSensorStream(
        timestamp_s=1.0,
        estimated_position=Point3D(x=0, y=0, z=20),
        actual_position=Point3D(x=0, y=0, z=20),
        remaining_battery_wh=170.0,
        battery_percentage=94.4
    )

    strategy, rationale = selector.select_strategy(scenario, telemetry)
    assert strategy == IntelligenceStrategy.STRONG_CLOUD_LLM


def test_strategy_selector_network_degradation():
    scenarios = load_benchmark_scenarios()
    scenario = scenarios[0]
    selector = AdaptiveStrategySelector()

    telemetry = SimulatedSensorStream(
        timestamp_s=1.0,
        estimated_position=Point3D(x=0, y=0, z=20),
        actual_position=Point3D(x=0, y=0, z=20),
        remaining_battery_wh=140.0,
        network_state=NetworkState(connected=True, latency_ms=300.0, packet_loss_ratio=0.20)
    )

    strategy, rationale = selector.select_strategy(scenario, telemetry)
    assert strategy == IntelligenceStrategy.DETERMINISTIC_FALLBACK


def test_strategy_selector_emergency():
    scenarios = load_benchmark_scenarios()
    scenario = scenarios[3]  # GNSS denial (Risk CRITICAL)
    selector = AdaptiveStrategySelector()

    telemetry = SimulatedSensorStream(
        timestamp_s=1.0,
        estimated_position=Point3D(x=0, y=0, z=20),
        actual_position=Point3D(x=0, y=0, z=20),
        remaining_battery_wh=180.0,
        gps_quality=GPSQuality(is_jammed=True)
    )

    strategy, rationale = selector.select_strategy(scenario, telemetry)
    assert strategy == IntelligenceStrategy.EMERGENCY_DETERMINISTIC_CONTROLLER
