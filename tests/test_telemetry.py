from __future__ import annotations

import pytest

from bfmc.telemetry import NullSpeedSensor, SpeedSensor, parse_rpm


@pytest.mark.parametrize(
    "line, expected",
    [
        ("134.5\n", 134.5),
        ("RPM:134.5\n", 134.5),
        ("rpm: 134.5\r\n", 134.5),
        ("  0  ", 0.0),
        ("-12.5", -12.5),
    ],
)
def test_parse_rpm_accepts_the_arduino_formats(line, expected):
    assert parse_rpm(line) == pytest.approx(expected)


@pytest.mark.parametrize("line", ["", "\n", "garbage", "RPM:", "1.2.3"])
def test_parse_rpm_rejects_noise(line):
    assert parse_rpm(line) is None


def test_rpm_converts_to_metres_per_second():
    sensor = SpeedSensor(wheel_circumference_m=0.20)
    sensor._publish(600.0)  # 10 rev/s * 0.20 m = 2.0 m/s
    assert sensor.speed_ms == pytest.approx(2.0)
    assert sensor.rpm == pytest.approx(600.0)


def test_sensor_reads_zero_before_any_data():
    assert SpeedSensor().speed_ms == 0.0


def test_stopping_a_sensor_that_never_started_is_safe():
    SpeedSensor().stop()


def test_a_missing_port_leaves_the_speed_at_zero_without_spinning():
    """An unplugged Arduino must not take the control loop down with it."""
    sensor = SpeedSensor(port="/dev/does-not-exist", timeout_s=0.01)
    with sensor:
        assert sensor.speed_ms == 0.0
    assert sensor.speed_ms == 0.0


def test_null_sensor_satisfies_the_interface():
    with NullSpeedSensor() as sensor:
        assert sensor.speed_ms == 0.0
        assert sensor.rpm == 0.0
