from __future__ import annotations

import logging
import threading

__all__ = ["SpeedSensor", "parse_rpm"]

log = logging.getLogger(__name__)

#: Seconds.
RECONNECT_DELAY_S = 2.0


def parse_rpm(line: str) -> float | None:
    """Accepts ``"RPM:134.5"``, ``"rpm: 134.5"`` or ``"134.5"``; ``None`` otherwise."""
    cleaned = line.strip().replace("RPM:", "").replace("rpm:", "").strip()
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


class SpeedSensor:
    """If the port will not open the sensor reads zero and keeps retrying, rather than crashing the
    loop.
    """

    def __init__(
        self,
        port: str = "/dev/ttyUSB0",
        baud_rate: int = 115200,
        wheel_circumference_m: float = 0.20,
        timeout_s: float = 1.0,
    ) -> None:
        self.port = port
        self.baud_rate = baud_rate
        self.wheel_circumference_m = wheel_circumference_m
        self.timeout_s = timeout_s

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._rpm = 0.0
        self._speed_ms = 0.0

    def start(self) -> SpeedSensor:
        if self._thread is not None:
            return self
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="speed-sensor", daemon=True)
        self._thread.start()
        return self

    def stop(self, timeout_s: float = 2.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout_s)
            self._thread = None

    def __enter__(self) -> SpeedSensor:
        return self.start()

    def __exit__(self, *exc_info: object) -> None:
        self.stop()

    @property
    def speed_ms(self) -> float:
        with self._lock:
            return self._speed_ms

    @property
    def rpm(self) -> float:
        with self._lock:
            return self._rpm

    def _publish(self, rpm: float) -> None:
        with self._lock:
            self._rpm = rpm
            self._speed_ms = (rpm / 60.0) * self.wheel_circumference_m

    def _run(self) -> None:
        import serial  # lazily, so the package works without pyserial

        while not self._stop.is_set():
            try:
                connection = serial.Serial(self.port, self.baud_rate, timeout=self.timeout_s)
            except serial.SerialException as exc:
                log.warning("Could not open %s (%s); speed stays 0", self.port, exc)
                # Otherwise a missing port spins the CPU.
                self._stop.wait(RECONNECT_DELAY_S)
                continue

            log.info("Serial open on %s", self.port)
            try:
                self._read_loop(connection)
            except serial.SerialException as exc:
                log.warning("Serial error on %s (%s); reconnecting", self.port, exc)
                self._publish(0.0)
                self._stop.wait(RECONNECT_DELAY_S)
            finally:
                connection.close()

    def _read_loop(self, connection: object) -> None:
        while not self._stop.is_set():
            line = connection.readline().decode("utf-8", errors="ignore")
            rpm = parse_rpm(line)
            if rpm is not None:
                self._publish(rpm)


class NullSpeedSensor:
    speed_ms = 0.0
    rpm = 0.0

    def start(self) -> NullSpeedSensor:
        return self

    def stop(self, timeout_s: float = 0.0) -> None:
        return None

    def __enter__(self) -> NullSpeedSensor:
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None
