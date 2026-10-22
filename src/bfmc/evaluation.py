from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from .pipeline import FrameResult, LaneKeepingPipeline
from .simulator import TrackSimulator

__all__ = ["LapReport", "SimulatedSpeedSensor", "drive"]


class SimulatedSpeedSensor:
    """Without it the controller sees a standstill and schedules its lookahead for one."""

    def __init__(self, simulator: TrackSimulator) -> None:
        self._simulator = simulator

    @property
    def speed_ms(self) -> float:
        return self._simulator.car.speed_ms

    @property
    def rpm(self) -> float:
        circumference = 0.20
        return self.speed_ms / circumference * 60.0

    def start(self) -> SimulatedSpeedSensor:
        return self

    def stop(self, timeout_s: float = 0.0) -> None:
        return None


@dataclass
class LapReport:
    frames: int
    duration_s: float
    distance_m: float
    completed: bool
    #: Metres, signed, one per step.
    errors: np.ndarray = field(repr=False)
    steering_deg: np.ndarray = field(repr=False)
    speeds_ms: np.ndarray = field(repr=False)
    latencies_ms: np.ndarray = field(repr=False)
    state_counts: Counter = field(default_factory=Counter, repr=False)

    @property
    def mean_abs_error_m(self) -> float:
        return float(np.abs(self.errors).mean()) if len(self.errors) else 0.0

    @property
    def rms_error_m(self) -> float:
        return float(np.sqrt((self.errors ** 2).mean())) if len(self.errors) else 0.0

    @property
    def max_abs_error_m(self) -> float:
        return float(np.abs(self.errors).max()) if len(self.errors) else 0.0

    @property
    def mean_speed_ms(self) -> float:
        return float(self.speeds_ms.mean()) if len(self.speeds_ms) else 0.0

    @property
    def steering_effort(self) -> float:
        """Degrees per frame: sawing at the wheel is worse even at the same mean error."""
        if len(self.steering_deg) < 2:
            return 0.0
        return float(np.abs(np.diff(self.steering_deg)).mean())

    @property
    def mean_latency_ms(self) -> float:
        return float(self.latencies_ms.mean()) if len(self.latencies_ms) else 0.0

    @property
    def fps(self) -> float:
        latency = self.mean_latency_ms
        return 1000.0 / latency if latency > 0 else 0.0

    def summary(self) -> str:
        verdict = "completed" if self.completed else "LEFT THE LANE"
        states = ", ".join(f"{name} {count}" for name, count in self.state_counts.most_common())
        return (
            f"{verdict}: {self.frames} frames, {self.distance_m:.2f} m in "
            f"{self.duration_s:.1f} s\n"
            f"  cross-track error  mean {self.mean_abs_error_m * 100:5.2f} cm   "
            f"rms {self.rms_error_m * 100:5.2f} cm   "
            f"max {self.max_abs_error_m * 100:5.2f} cm\n"
            f"  steering effort    {self.steering_effort:5.2f} deg/frame   "
            f"mean speed {self.mean_speed_ms:.2f} m/s\n"
            f"  perception         {self.mean_latency_ms:5.1f} ms/frame "
            f"({self.fps:.0f} fps)\n"
            f"  states             {states}"
        )


def drive(
    simulator: TrackSimulator,
    pipeline: LaneKeepingPipeline,
    steps: int = 600,
    dt: float = 0.05,
    stop_when_off_track: bool = True,
    on_result=None,
) -> LapReport:
    """Closed loop: each command moves the car, which determines the next frame."""
    pipeline.speed_sensor = SimulatedSpeedSensor(simulator)
    pipeline.reset()

    errors, steering, speeds, latencies = [], [], [], []
    states: Counter = Counter()
    completed = True

    for _ in range(steps):
        frame = simulator.render()
        result: FrameResult = pipeline.process(frame)

        if on_result is not None:
            on_result(result)

        simulator.step(result.command.steering_deg, result.command.target_speed_ms, dt)

        errors.append(simulator.cross_track_error_m)
        steering.append(result.command.steering_deg)
        speeds.append(simulator.car.speed_ms)
        latencies.append(result.latency_s * 1000.0)
        states[str(result.snapshot.state)] += 1

        if not simulator.on_track:
            completed = False
            if stop_when_off_track:
                break

        if simulator.at_end_of_track:
            break

    return LapReport(
        frames=len(errors),
        duration_s=len(errors) * dt,
        distance_m=simulator.car.odometer_m,
        completed=completed,
        errors=np.asarray(errors),
        steering_deg=np.asarray(steering),
        speeds_ms=np.asarray(speeds),
        latencies_ms=np.asarray(latencies),
        state_counts=states,
    )
