import random
import math
from typing import List, Tuple, Dict, Any, Optional
from uav_neurosym.schema import (
    Weather, FaultInjection, GPSQuality, NetworkState, ObstacleObservation, Point3D, FlightScenario
)


class VirtualEnvironment:
    """
    Manages dynamic software simulation of environmental factors:
    - Weather fluctuations & wind gust vectors
    - Simulated sensor noise & GPS degradation / GNSS jamming
    - Simulated network telemetry (latency, packet loss, bandwidth)
    - Dynamic procedurally generated obstacles
    - Time-varying fault injections
    """

    def __init__(self, scenario: FlightScenario, seed: int = 42):
        self.scenario = scenario
        self.seed = seed
        random.seed(seed)

        self.current_weather = scenario.weather.model_copy()
        self.active_faults: List[FaultInjection] = []
        self.obstacles: List[ObstacleObservation] = []

        # Initialize procedurally generated obstacles if scenario has NFZs or Risk
        self._init_simulated_obstacles()

    def _init_simulated_obstacles(self):
        """Generates procedural static and dynamic obstacles within the airspace."""
        for idx, nfz in enumerate(self.scenario.no_fly_zones):
            # Centroid of NFZ polygon acts as a known static obstacle center
            px = sum(pt[0] for pt in nfz.polygon) / len(nfz.polygon)
            py = sum(pt[1] for pt in nfz.polygon) / len(nfz.polygon)
            pz = (nfz.min_alt_m + nfz.max_alt_m) / 2.0

            self.obstacles.append(ObstacleObservation(
                obstacle_id=f"OBS-NFZ-{idx+1}",
                position=Point3D(x=px, y=py, z=pz),
                radius_m=max(abs(nfz.polygon[0][0] - px), 15.0),
                velocity_ms=(0.0, 0.0, 0.0),
                is_dynamic=False
            ))

    def update_step(self, elapsed_time_s: float, dt_s: float) -> Tuple[Weather, GPSQuality, NetworkState, List[ObstacleObservation]]:
        """
        Advances the virtual environment state by dt_s seconds.
        Returns (updated_weather, gps_quality, network_state, visible_obstacles).
        """
        # 1. Update Weather (simulate mild wind gusts and icing)
        base_wind = self.scenario.weather.wind_speed_ms
        gust_amplitude = self.scenario.weather.gust_amplitude_ms
        # Sinusoidal wind gust oscillation + noise
        current_wind = max(0.0, base_wind + gust_amplitude * math.sin(elapsed_time_s / 5.0) + random.uniform(-0.5, 0.5))
        self.current_weather.wind_speed_ms = round(current_wind, 2)

        # 2. Check Active Fault Injections
        self.active_faults = []
        gnss_jamming_active = False
        jamming_severity = 0.0

        for fault in self.scenario.faults:
            if fault.start_time_s <= elapsed_time_s <= (fault.start_time_s + fault.duration_s):
                self.active_faults.append(fault)
                if fault.fault_type == "gnss_denial":
                    gnss_jamming_active = True
                    jamming_severity = fault.severity

        # 3. Simulate GPS Sensor Quality
        if gnss_jamming_active or self.scenario.weather.gnss_jamming_power_dbm > -90.0:
            gps_quality = GPSQuality(
                is_jammed=True,
                satellite_count=max(2, int(12 * (1.0 - jamming_severity))),
                hdop=round(1.1 + jamming_severity * 8.5, 2),
                position_noise_m=round(0.5 + jamming_severity * 25.0, 2)
            )
        else:
            gps_quality = GPSQuality(
                is_jammed=False,
                satellite_count=12,
                hdop=1.1,
                position_noise_m=0.5
            )

        # 4. Simulate Network Connection Telemetry
        if self.current_weather.wind_speed_ms > 12.0:
            # Network degradation under severe wind
            network_state = NetworkState(
                connected=True,
                latency_ms=round(85.0 + random.uniform(10.0, 40.0), 1),
                bandwidth_mbps=2.5,
                packet_loss_ratio=0.08
            )
        else:
            network_state = NetworkState(
                connected=True,
                latency_ms=round(25.0 + random.uniform(-3.0, 5.0), 1),
                bandwidth_mbps=10.0,
                packet_loss_ratio=0.01
            )

        # 5. Step Dynamic Obstacles
        for obs in self.obstacles:
            if obs.is_dynamic:
                vx, vy, vz = obs.velocity_ms
                obs.position.x += vx * dt_s
                obs.position.y += vy * dt_s
                obs.position.z += vz * dt_s

        return self.current_weather, gps_quality, network_state, self.obstacles
