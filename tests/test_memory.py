import pytest
from uav_neurosym.memory.temporal_memory import TemporalMemoryManager
from uav_neurosym.schema import SimulatedSensorStream, Point3D, GPSQuality, NetworkState, Weather


def test_temporal_memory_discharge_rate():
    memory = TemporalMemoryManager()

    # Record 5 seconds of telemetry snapshots with decreasing battery
    for i in range(6):
        telemetry = SimulatedSensorStream(
            timestamp_s=float(i),
            estimated_position=Point3D(x=0, y=0, z=20),
            actual_position=Point3D(x=0, y=0, z=20),
            remaining_battery_wh=100.0 - (i * 2.0),  # 2 Wh/sec drop rate
            power_draw_w=180.0
        )
        memory.record_step(telemetry)

    rate = memory.get_battery_discharge_rate_wh_per_sec(window_seconds=5.0)
    assert rate == pytest.approx(2.0, rel=1e-2)


def test_temporal_memory_gps_trend():
    memory = TemporalMemoryManager()

    # Record GPS noise degrading over time
    for i in range(5):
        telemetry = SimulatedSensorStream(
            timestamp_s=float(i * 2),
            estimated_position=Point3D(x=0, y=0, z=20),
            actual_position=Point3D(x=0, y=0, z=20),
            remaining_battery_wh=90.0,
            gps_quality=GPSQuality(position_noise_m=0.5 + (i * 2.0))
        )
        memory.record_step(telemetry)

    noise_m, is_degrading = memory.get_gps_noise_trend(window_seconds=10.0)
    assert noise_m > 5.0
    assert is_degrading is True


def test_chronological_events_log():
    memory = TemporalMemoryManager()
    t = SimulatedSensorStream(
        timestamp_s=10.0,
        estimated_position=Point3D(x=0, y=0, z=20),
        actual_position=Point3D(x=0, y=0, z=20),
        remaining_battery_wh=80.0
    )

    memory.record_step(t, event_text="Test Event Triggered")
    events = memory.get_chronological_events()

    assert len(events) == 1
    assert events[0]["event"] == "Test Event Triggered"
    assert events[0]["timestamp_s"] == 10.0
