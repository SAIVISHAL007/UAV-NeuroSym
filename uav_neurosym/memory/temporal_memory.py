from typing import List, Dict, Any, Optional, Tuple
from uav_neurosym.schema import SimulatedSensorStream, Point3D


class TemporalMemoryManager:
    """
    Temporal Memory & Trajectory State History Manager for NeuroSym platform.
    Maintains a sliding window time-series log of past simulation states (t0 ... tN):
    - Calculates battery discharge rate slope (Wh/sec).
    - Calculates sensor noise & degradation trends over time.
    - Logs chronological timeline of critical events (fault triggers, guardrails, replans).
    """

    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.history: List[SimulatedSensorStream] = []
        self.events: List[Dict[str, Any]] = []

    def record_step(self, telemetry: SimulatedSensorStream, event_text: Optional[str] = None):
        """Records a new discrete simulation step telemetry snapshot."""
        self.history.append(telemetry)
        if len(self.history) > self.capacity:
            self.history.pop(0)

        if event_text:
            self.events.append({
                "timestamp_s": telemetry.timestamp_s,
                "event": event_text,
                "battery_wh": telemetry.remaining_battery_wh,
                "position": telemetry.actual_position.model_dump()
            })

    def get_battery_discharge_rate_wh_per_sec(self, window_seconds: float = 10.0) -> float:
        """
        Calculates the average battery discharge rate (Wh/sec) over the last window_seconds.
        """
        if len(self.history) < 2:
            return 0.0

        current_t = self.history[-1].timestamp_s
        cutoff_t = current_t - window_seconds

        # Find historical snapshot at cutoff
        past_snapshot = self.history[0]
        for snap in reversed(self.history):
            if snap.timestamp_s <= cutoff_t:
                past_snapshot = snap
                break

        dt = self.history[-1].timestamp_s - past_snapshot.timestamp_s
        if dt <= 0.0:
            return 0.0

        dE = past_snapshot.remaining_battery_wh - self.history[-1].remaining_battery_wh
        return max(0.0, dE / dt)

    def get_gps_noise_trend(self, window_seconds: float = 10.0) -> Tuple[float, bool]:
        """
        Returns (current_average_noise_m, is_degrading_trend).
        is_degrading_trend is True if position noise is increasing over time.
        """
        if len(self.history) < 2:
            return (0.5, False)

        recent = [s for s in self.history if s.timestamp_s >= (self.history[-1].timestamp_s - window_seconds)]
        if len(recent) < 2:
            return (self.history[-1].gps_quality.position_noise_m, False)

        current_noise = recent[-1].gps_quality.position_noise_m
        past_noise = recent[0].gps_quality.position_noise_m
        is_degrading = (current_noise - past_noise) > 1.0

        return (current_noise, is_degrading)

    def get_chronological_events(self) -> List[Dict[str, Any]]:
        return self.events

    def clear(self):
        self.history.clear()
        self.events.clear()
