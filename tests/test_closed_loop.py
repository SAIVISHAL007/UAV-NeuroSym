import pytest
from uav_neurosym.data.loader import load_benchmark_scenarios
from uav_neurosym.agents.llm_client import LLMClient
from uav_neurosym.agents.closed_loop_agent import ClosedLoopAgent


def test_closed_loop_nominal_mission():
    scenarios = load_benchmark_scenarios()
    scenario = scenarios[0]  # Nominal Urban Corridor Inspection

    client = LLMClient(provider="mock")
    agent = ClosedLoopAgent(client, dt_s=1.0, max_steps=300)
    result = agent.run_mission(scenario)

    assert result["total_steps"] > 0
    assert result["elapsed_time_s"] > 0.0
    assert result["final_battery_wh"] > 0.0
    assert result["is_violated"] is False


def test_closed_loop_gnss_fault_recovery():
    scenarios = load_benchmark_scenarios()
    scenario = scenarios[3]  # GNSS denial fault scenario

    client = LLMClient(provider="mock")
    agent = ClosedLoopAgent(client, dt_s=1.0, max_steps=300)
    result = agent.run_mission(scenario)

    # Verify event logged and emergency protocol activated
    assert result["replans_count"] > 0
    assert any("GNSS jamming" in event for event in result["events_logged"])
