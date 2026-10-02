import pytest
from uav_neurosym.data.loader import load_benchmark_scenarios
from uav_neurosym.schema import StepAction, Point3D
from uav_neurosym.sim.simulator import VirtualUAVSimulator
from uav_neurosym.sim.environment import VirtualEnvironment


def test_simulation_step_progression():
    scenarios = load_benchmark_scenarios()
    scenario = scenarios[0]  # Nominal scenario

    sim = VirtualUAVSimulator(scenario, dt_s=1.0)
    initial_state = sim.get_state()

    assert initial_state.step_index == 0
    assert initial_state.elapsed_time_s == 0.0
    assert initial_state.current_telemetry.remaining_battery_wh == scenario.uav.battery_capacity_wh

    # Execute 5 steps towards target waypoint
    target_wp = scenario.waypoints[0]
    for _ in range(5):
        sim.step(StepAction(target_waypoint=target_wp, airspeed_ms=10.0))

    state_after_5s = sim.get_state()
    assert state_after_5s.step_index == 5
    assert state_after_5s.elapsed_time_s == 5.0
    assert state_after_5s.current_telemetry.remaining_battery_wh < scenario.uav.battery_capacity_wh
    assert state_after_5s.current_telemetry.power_draw_w > 0.0


def test_simulation_environment_faults():
    scenarios = load_benchmark_scenarios()
    scenario = scenarios[3]  # GNSS denial scenario (UAVBENCH-004)

    env = VirtualEnvironment(scenario, seed=42)
    weather, gps_q, net_s, obs = env.update_step(elapsed_time_s=15.0, dt_s=1.0)

    # Active fault in scenario 4 starts at t=10.0s for 120s
    assert gps_q.is_jammed is True
    assert gps_q.position_noise_m > 5.0


def test_simulator_emergency_override():
    scenarios = load_benchmark_scenarios()
    scenario = scenarios[0]

    sim = VirtualUAVSimulator(scenario, dt_s=1.0)
    # Move drone out
    sim.step(StepAction(target_waypoint=scenario.waypoints[0], airspeed_ms=12.0))

    # Issue emergency override RETURN_TO_BASE
    state = sim.step(StepAction(
        target_waypoint=scenario.waypoints[1],
        airspeed_ms=12.0,
        emergency_override="RETURN_TO_BASE"
    ))

    # Actual position should be heading back towards spawn_point (0, 0, 20)
    pos = state.current_telemetry.actual_position
    assert pos.z >= 10.0
