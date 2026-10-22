# Contributing

## Setup

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

`picamera2` is not installed by the dev extra — it only exists on a Raspberry
Pi. Everything else runs on a laptop, and `camera.create_source` falls back to
the OpenCV backend when it is missing.

## Before opening a pull request

```bash
ruff check src tests scripts
pytest
```

Both run in CI on Python 3.9, 3.11 and 3.12.

## Guidelines

- **Keep the maths hardware-free.** `geometry`, `fitting`, `polyfit`,
  `tracking` and `control` must not import `cv2`, `picamera2` or `serial`.
  That separation is what makes them testable without a car.
- **Measure changes to perception or control.** `scripts/simulate.py --compare`
  gives a cross-track error, a steering effort and a mean speed on a known
  track. Put the before and after numbers in the pull request; "it looked
  smoother" is not a result.
- **Add a test with behaviour changes.** New tunables need a case showing they
  do something; bug fixes need one that fails before the fix. If the change
  affects how the car drives, the case belongs in `tests/test_driving.py`.
- **Every tunable belongs in `config.py`** with a default and a comment, not as
  a constant buried in a module.
- **Explain non-obvious guards.** Several of the checks in `polyfit.py` and
  `tracking.py` exist because of a specific failure that is invisible from the
  code alone. If you add one, say what it prevents.

## Things worth working on

- **Actuation**: turning a `DriveCommand` into servo and motor output.
- **A better lane model for tight corners.** `x = f(y)` degenerates past about
  60° of turn; an arc-length parameterisation or a clothoid would not.
- **Kalman filtering of the lane state** across frames, rather than only
  smoothing the steering output.
- **Wider simulator coverage**: intersections, missing boundaries, shadows
  across the road, a camera mounted off-centre.
- **Profiling on the Pi.** The loop runs at ~75 fps on a laptop; the Pi 4 is a
  different story and `--log-csv` records per-frame latency to prove it.
