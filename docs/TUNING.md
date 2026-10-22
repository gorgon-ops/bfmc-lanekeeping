# Tuning

Tune in this order. Each step assumes the previous one is correct — a wrong
warp makes every gain below it meaningless.

Tune against the simulator first where you can. It is repeatable, a bad setting
costs nothing, and it gives you a number instead of an impression:

```bash
python scripts/simulate.py --set control.lookahead_time_s=0.6
python scripts/simulate.py --compare
```

## 1. The perspective warp

`warp.*` defines a trapezoid in the raw camera image that gets stretched to
fill the bird's-eye view.

Park on a straight section with both lane markings visible, then:

```bash
python scripts/calibrate_warp.py --config configs/vehicle.yaml
```

Drag the sliders until **both lane markings come out vertical and parallel** in
the warped pane. Press `s` to print a YAML block to paste into your config.

- `horizon_y_frac` too low pulls in sky and distant clutter.
- `horizon_y_frac` too high throws away the lookahead you need to steer
  smoothly, and shortens the sensing horizon that caps your speed.
- If the markings bow outwards at the top, widen `top_left_x_frac` /
  `top_right_x_frac`; if they pinch inwards, narrow them.

## 2. The metric scale

Two real measurements turn pixels into metres, and everything physical
downstream depends on them — curve radius, lateral offset, lookahead distance,
target speed:

| Key | What to measure |
| --- | --- |
| `metric.lane_width_m` | The real distance between the two markings |
| `metric.view_depth_m` | How much road the warped view covers, bottom to top |
| `tracking.lane_width_px` | That same lane width, in warped pixels |

For `view_depth_m`, put an object on the road at the point that appears at the
very top of the warped pane and measure to the front axle. The calibration
script prints the measured lane width in pixels when both boundaries are
detected.

Get these wrong and nothing crashes — the car just drives to the wrong speed
for corners that are not the radius it thinks they are.

## 3. Detection

Only touch these if boundaries are being missed or phantom ones appear. Watch
the state readout: frequent `COASTING` means detection is dropping out,
frequent `LEFT ONLY` / `RIGHT ONLY` means one side genuinely is not seen.

| Symptom | Try |
| --- | --- |
| Markings missed on dark or low-contrast track | lower `detection.mask.brightness_offset` (more negative) |
| Noise and speckle everywhere | raise `detection.mask.open_kernel`, raise `gradient_threshold` |
| Coloured clutter surviving the mask | lower `detection.mask.max_saturation` |
| Boundary lost halfway up the image | raise `detection.window.margin` or `max_missed_windows` |
| Fit wanders onto the wrong marking | lower `detection.window.search_margin` |
| Confidence always low | the fit is short — check the warp and `min_span_frac` |

`--show-mask` puts the binary mask in the debug view, which answers most of
these questions faster than any parameter sweep.

## 4. Steering

**`control.law`** — `pure_pursuit` is the default and is smoother; `stanley`
holds a slightly tighter line and reacts faster, at the cost of more
sensitivity to heading noise.

**`control.feedforward`** — leave it on `auto`. It enables the curvature
feedforward for Stanley and disables it for pure pursuit, which already follows
curvature through its goal point. Overriding it is the single most expensive
mistake available here; `scripts/simulate.py --compare` shows the cost.

**`control.lookahead_base_m` / `lookahead_time_s`** — the lookahead is
`base + time × speed`, so the car looks a roughly constant *time* ahead.
Raise `lookahead_time_s` if the car weaves on straights; lower it if it enters
corners late or cuts them.

**`control.butter_cutoff`** (default `0.15`) trades smoothness for
responsiveness. Lower it if the wheel twitches frame to frame; raise it if the
car reacts late. Watch `steering effort` in the simulator report — it catches
a controller that holds the line by sawing at the wheel, which mean error does
not.

## 5. Speed

**`vehicle.max_lateral_accel_ms2`** sets how hard the car is willing to corner
(`v ≤ √(a·R)`). Lower it if the car slides or understeers in corners.

**`control.horizon_time_s`** is the safety parameter that matters most on a
tight track: the car will not travel further than its lane fit reaches in this
many seconds. Raising it brakes earlier for corners it cannot see round;
setting it to `0` disables the limit entirely.

The trade-off is real and worth measuring on your own track:

| `horizon_time_s` | Easy oval | Tight 0.8 m corners |
| --- | --- | --- |
| 0.0 (off) | 0.8 cm at 0.99 m/s | **leaves the lane** |
| 1.1 (default) | 0.5 cm at 0.48 m/s | 6.8 cm, completes |

If the car is too slow for your liking, the fix is a wider field of view rather
than a smaller `horizon_time_s` — the limit is only telling you how little road
the camera can actually see.

## 6. Checking the encoder

```bash
python -m serial.tools.miniterm /dev/ttyUSB0 115200
```

You should see one number per line, roughly every 100 ms, rising as you spin
the wheel. Set `encoder.wheel_circumference_m` to the wheel diameter times pi,
measured rather than guessed — an error here scales every speed reading and
therefore the lookahead.

Push the car a measured distance and check `speed × time` roughly matches it.
If the reported speed is consistently double or half, check `PULSES_PER_REV` in
the Arduino sketch against the number of magnets on the wheel.

With the encoder disabled the controller uses `control.assumed_speed_ms` so the
lookahead stays sane; set it near the speed you actually expect to drive.
