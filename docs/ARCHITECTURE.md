# Architecture

## Layering

Modules are split so that the parts worth testing do not depend on hardware:

| Layer | Modules | Dependencies |
| --- | --- | --- |
| Maths | `geometry`, `fitting`, `polyfit` | numpy |
| Decision | `tracking`, `states`, `control` | numpy, scipy |
| Image | `thresholding`, `perception`, `overlay` | OpenCV |
| Hardware | `camera`, `telemetry` | OpenCV, picamera2, pyserial |
| Offline | `simulator`, `evaluation` | OpenCV, numpy |
| Wiring | `config`, `pipeline`, `cli` | all of the above |

Everything outside the hardware row runs on a laptop, which is what makes the
test suite meaningful without a car on a track.

## Data flow

```
frame (BGR)
  │
  ├─ PerspectiveWarp ─────────────▶ warped frame
  │                                     │
  │                                 lane_mask          (thresholding)
  │                                     │
  │                          SlidingWindowFinder       (polyfit)
  │                                     │
  │                       LanePolynomial | None  ×2
  │                                     │
  │                            LaneTracker.update      (tracking)
  │                                     │
  │                               LaneSnapshot
  │            (boundaries, centreline, state, curvature, offset,
  │             heading, sensing horizon, confidence)
  │                                     │
  └─ SpeedSensor.speed_ms ──▶ SteeringController.update (control)
                                        │
                                  DriveCommand
                        (steering angle, target speed, goal point)
```

## Coordinate conventions

All lane coordinates are pixels in the **warped** frame:

- `x` grows to the right, `y` grows **downwards**
- `y = height` is the bottom of the image, closest to the vehicle
- looking further ahead means *decreasing* `y`

Three consequences worth remembering:

- The **left** boundary has a **negative** slope in `x = f(y)` and the right one
  a positive slope, which is how the Hough detector tells them apart.
- Positive **curvature** means the road bends right; positive **heading** means
  the road runs off to the right; positive **steering** means steer right;
  positive **lateral offset** means the car sits right of the centreline. All
  four share a sign so they can be added without thinking.
- In the world frame the simulator uses, positive steering is *clockwise*, so
  it **decreases** yaw. Getting that backwards turns every correction into its
  opposite and the car diverges on a dead-straight road.

## The lane model

A boundary is `x = a·y² + b·y + c` over a validity range `[y_min, y_max]`. A
straight line is just `a = 0`, so Hough segments and sliding-window fits share
one representation and everything downstream works with either.

Fitting `x` as a function of `y` (rather than the reverse) is what makes this
work at all: a lane boundary is close to vertical in the warped view, so `x(y)`
is single-valued while `y(x)` is not.

The validity range matters as much as the coefficients. Extrapolating a
parabola beyond the road it was fitted on is wildly inaccurate exactly when it
matters most — in a corner, where the boundary curves out of frame and the
fitted range is short — so the controller never aims past `y_min`, and the
speed policy treats the span as the sensing horizon.

### Why `x = f(y)` eventually breaks

As a road turns towards horizontal, `x(y)` stops being a function of `y`: at
90° it is multivalued and the fit is meaningless. Three guards bound that,
all in `polyfit.py`:

| Guard | What it prevents |
| --- | --- |
| Stop the window walk at a wide sideways smear | A band holding a long horizontal stretch of boundary corrupting the fit |
| Exclude pixels at the frame edge | A clipped boundary leaving a vertical strip that bends the fit back the wrong way |
| Stop after consecutive empty windows | Jumping a gap to a different stretch of road that happens to be in line |

The last two are not hypothetical. A boundary that leaves the side of the image
leaves a vertical remnant against the border; fitting through it flips the sign
of the curvature, so the car is told a left-hand corner bends right.

## State machine

`LaneTracker` classifies each frame after memory and recovery have been applied:

| State | Meaning |
| --- | --- |
| `LANE_KEEPING` | Both boundaries detected this frame |
| `COASTING` | At least one boundary replayed from memory |
| `LEFT_ONLY` | Only the left boundary is real; the right was reconstructed |
| `RIGHT_ONLY` | Only the right boundary is real; the left was reconstructed |
| `STOP` | Nothing to follow, even from memory |

The order matters: reconstruction is checked before staleness, because a
boundary that had to be invented is weaker evidence than one merely replayed.

`STOP` commands a straight wheel and zero speed, but that zero still runs
through the Butterworth filter so the steering signal stays continuous and does
not snap when detection recovers.

## Plausibility checks

Two boundaries that are implausibly close together or far apart are not a lane.
This happens when both searches lock onto the same marking, and it is *worse*
than seeing nothing: the centreline lands on the marking itself and the car
confidently steers half a lane off. The tracker keeps whichever fit had more
support and rebuilds the other from the lane width.

Confidence carries the same idea more softly. It combines how many windows
found pixels with how far the fit reaches, and the speed policy spends it: a
short or sparse fit slows the car rather than steering it on a guess.

## Control

The steering command is feedforward plus feedback, but the two are **not**
independent — see the module docstring in `control.py` for the measurements.
Briefly: pure pursuit already follows curvature through its goal point, so
adding a curvature feedforward commands the same turn twice; Stanley only
regulates error, so it needs one. `feedforward: auto` pairs them correctly.

The speed command has three limits and takes the smallest:

1. `√(a_max · R)` for the tightest radius *in view*, so the car slows before a
   corner rather than inside it.
2. `visible_distance / horizon_time` — do not outdrive the sensor. The lane fit
   reaches a certain distance and no further, and in a corner that collapses as
   the boundaries swing out of frame.
3. `max_speed`, derated by confidence and by a degraded driving state.

## Extension points

- **New frame source**: implement `read()` and `close()` (the `FrameSource`
  protocol in `camera.py`) and pass it as `LaneKeepingPipeline(config, source=…)`.
- **New detector**: return a `LaneDetection` with two `LanePolynomial`s and
  register it in `create_detector`.
- **New control law**: `SteeringController._feedback` is the only place the
  steering angle is produced; the tracker and detector know nothing about it.
- **Actuation**: `pipeline.process()` returns a `DriveCommand` per frame.
  Wiring that to a servo is deliberately out of scope — this package computes a
  steering angle, it does not drive anything.
