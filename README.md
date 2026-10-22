# BFMC Lane Keeping

Vision-based lane keeping for a Bosch Future Mobility Challenge vehicle: a
Raspberry Pi camera, an Arduino wheel encoder, and a pure-pursuit steering
controller — with a closed-loop simulator so the whole thing can be run,
measured and tuned without touching the car.

[![CI](https://github.com/stefanoanea/bfmc-lanekeeping/actions/workflows/ci.yml/badge.svg)](https://github.com/stefanoanea/bfmc-lanekeeping/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Try it without hardware

```bash
pip install -e ".[dev]"
python scripts/simulate.py
```

That drives a simulated car round a track using the real pipeline — the
simulator renders what the camera would see, the pipeline steers, and the
steering moves the car. Because the track is known, the cross-track error is
known at every step:

```
track: oval
completed: 300 frames, 7.25 m in 15.0 s
  cross-track error  mean  0.49 cm   rms  0.68 cm   max  2.02 cm
  steering effort     0.10 deg/frame   mean speed 0.48 m/s
  perception          13.4 ms/frame (74 fps)
  states             LANE KEEP 300
```

Half a centimetre of error in a 35 cm lane. Other tracks: `--track tight`
(0.8 m corners), `--track hairpin` (0.6 m), `--track figure8` (turns both
ways). Add `--record lap.mp4` to write out the debug view above.

## How it works

```
camera ─▶ bird's-eye warp ─▶ adaptive lane mask ─▶ sliding-window search
                                                            │
                                          polynomial fit ◀──┘
                                                │
                              lane memory + recovery + width check
                                                │
                       pure pursuit ─▶ smoothing ─▶ steering angle
                                   └─▶ curvature + horizon ─▶ target speed
```

1. **Bird's-eye warp** turns converging lane markings into roughly parallel
   vertical lines, which is what makes a lane width in pixels and a metric
   scale mean anything.
2. **Adaptive lane mask** combines a locally-adaptive brightness threshold, a
   normalised gradient, and a saturation filter. Nothing here has an absolute
   brightness threshold, so the same settings survive a change of lighting.
3. **Sliding-window search** walks a window up from each boundary's base,
   re-centring as it goes, and fits `x = a·y² + b·y + c` through the pixels it
   collects. Once a fit exists the next frame searches a band around it
   instead, which is faster and steadier.
4. **Lane memory and recovery** replay a lost boundary for a few frames, and
   reconstruct a missing one from the opposite boundary plus the calibrated
   lane width.
5. **Pure pursuit** aims at the road centre a lookahead distance ahead, where
   the lookahead grows with speed so the car looks a roughly constant *time*
   ahead. Stanley is available as an alternative.
6. **Speed** comes from the measured curve radius (`v ≤ √(a_max · R)`) and from
   how far the lane fit actually reaches — the car will not outdrive what it
   can see.

The vehicle reports one of five states: `LANE KEEP`, `COASTING` (running on
lane memory), `LEFT ONLY` / `RIGHT ONLY` (one boundary reconstructed), or
`STOP` (nothing to follow).

## What the simulator is for

Tuning a lane keeper by watching videos is guesswork. With ground truth
available every frame, design choices become measurements:

```bash
python scripts/simulate.py --compare
```

```
variant                         done     mean     rms     max  effort   speed
-----------------------------------------------------------------------------
pure pursuit (default)           yes     0.49    0.68    2.02    0.10    0.48
stanley                          yes     0.52    0.58    1.09    0.06    0.40
pure pursuit + feedforward       yes     6.16    8.14   17.70    0.27    0.66
hough (straight-line fit)        yes     3.17    4.51   11.43    1.81    0.43
no horizon speed limit           yes     0.74    0.93    2.36    0.21    0.99
```

Three of the defaults in this repository came out of that table:

- **Pure pursuit gets no curvature feedforward.** It already follows curvature
  through its goal point; adding a feedforward commands the same turn twice and
  costs an order of magnitude in accuracy. Stanley is the opposite — its terms
  only regulate error, so it *needs* one, and without it leaves the lane
  entirely. `control.feedforward: auto` pairs them correctly.
- **A polynomial fit beats a straight-line fit**, and the clearest sign is not
  accuracy but *steering effort*: the Hough baseline saws at the wheel 18×
  harder because a straight line cannot represent a bend, so it re-aims every
  frame.
- **Capping speed by the sensing horizon is what gets round tight corners.**
  Without it the car is faster on the easy oval and leaves the lane on the
  0.8 m one.

## Install on the vehicle

```bash
git clone https://github.com/stefanoanea/bfmc-lanekeeping.git
cd bfmc-lanekeeping
sudo apt install -y python3-picamera2   # ships with Raspberry Pi OS, not PyPI
pip install -e .
bfmc --config configs/vehicle.yaml
```

Offline, against a recorded clip:

```bash
bfmc --config configs/replay.yaml --video path/to/lap.mp4
```

Press `q` in the debug window to quit, or Ctrl-C in the terminal.

### Useful flags

| Flag | Effect |
| --- | --- |
| `--config FILE` | Load a YAML config; anything omitted keeps its default |
| `--video FILE` | Replay a clip instead of a live camera |
| `--method {sliding_window,hough}` | Pick the lane detector |
| `--law {pure_pursuit,stanley}` | Pick the steering law |
| `--no-display` | Headless; log one line per frame |
| `--show-mask` | Add the binary mask to the debug view |
| `--record FILE` | Write the debug view to a video |
| `--log-csv FILE` | Write per-frame telemetry for offline analysis |
| `--set KEY=VALUE` | Override any config value, repeatable |

`--set` reaches every tunable by dotted path, which makes an experiment a
one-liner:

```bash
bfmc --video lap.mp4 --set control.lookahead_time_s=0.6 --set tracking.lane_width_px=280
```

## Configuration

Defaults live in [`src/bfmc/config.py`](src/bfmc/config.py); a YAML file only
needs the values that differ for your car. Unknown keys are rejected rather
than ignored, so a typo fails loudly instead of leaving a tunable at its
default.

Three starting points are provided: [`configs/vehicle.yaml`](configs/vehicle.yaml)
for the car, [`configs/replay.yaml`](configs/replay.yaml) for recorded clips,
and [`configs/sim.yaml`](configs/sim.yaml) for simulation.

The values that matter most are the ones specific to your hardware: the
bird's-eye trapezoid (`warp.*`), the lane width in warped pixels
(`tracking.lane_width_px`), and the two real-world measurements that set the
metric scale (`metric.lane_width_m`, `metric.view_depth_m`). See
[docs/TUNING.md](docs/TUNING.md).

## Hardware

- Raspberry Pi 4 with a Pi Camera, 640×480
- Arduino with a hall-effect sensor on the drive wheel, over USB

The sketch is in [`arduino/encoder/`](arduino/). It reports RPM at 115200 baud,
one value per line; the Pi converts that to m/s using the wheel circumference.
If the port cannot be opened the reader retries in the background and the
controller falls back to an assumed speed rather than taking the control loop
down with it.

## Development

```bash
pytest                            # 238 tests
ruff check src tests scripts      # lint
```

The perception and control logic has no camera or serial dependencies, so it is
testable on a laptop. [`tests/test_driving.py`](tests/test_driving.py) is the
one that matters: it drives simulated laps end to end and asserts on the
resulting cross-track error, so a regression in any layer shows up as a car
that wanders.

## Repository layout

```
src/bfmc/
  config.py       typed configuration with YAML overlays
  camera.py       frame sources: Pi camera, USB camera, video file
  thresholding.py adaptive binary lane masks
  polyfit.py      sliding-window search and robust polynomial fitting
  perception.py   perspective warp and the two detectors
  geometry.py     the lane model: curvature, heading, metric scale
  fitting.py      length-weighted straight-line fit (Hough baseline)
  tracking.py     lane memory, boundary recovery, driving states
  control.py      pure pursuit, Stanley, speed policy, smoothing
  telemetry.py    background serial reader for wheel speed
  simulator.py    track model, bicycle dynamics, camera rendering
  evaluation.py   closed-loop scoring
  overlay.py      debug visualisation
  pipeline.py     the loop that wires it together
  cli.py          command-line entry point
arduino/encoder/  Arduino sketch for the wheel encoder
configs/          ready-made configurations
scripts/          simulate.py, calibrate_warp.py
docs/             architecture and tuning notes
```

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — how the modules fit together,
  coordinate conventions, and the failure modes each guard exists for
- [docs/TUNING.md](docs/TUNING.md) — calibrating the warp, the scale, and the gains
- [CONTRIBUTING.md](CONTRIBUTING.md) — development workflow

## Known limits

- A second-order polynomial in `x = f(y)` cannot represent a boundary that has
  turned much past 60°. On corners tighter than about 0.8 m radius the inner
  boundary leaves the frame, the fit is truncated, and the car falls back to
  reconstructing it from the outer one and slows right down. It stays in the
  lane, but a wider-angle camera would serve it better than more tuning.
- Curvature from a parabola fitted to an arc carries a systematic error of
  roughly 10–15% at BFMC corner radii. The steering does not depend on it —
  pure pursuit needs a *position*, not a second derivative — but the speed
  policy does, which is why it also uses the sensing horizon.
- There is no actuation here. The pipeline produces a steering angle and a
  target speed; driving a servo and a motor controller is left to the vehicle's
  own stack.

## License

MIT — see [LICENSE](LICENSE).
