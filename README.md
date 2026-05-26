# KITE: Kinematic Interpretable Trajectory Estimation

KITE is a KITScenes trajectory-prediction method that asks Gemma for interpretable driving intent and road geometry, then converts that language into physical waypoints with a bicycle model.

The main method is:

```text
geometry_actions_bicycle
```

The other two implemented methods are baselines used to understand where the gain comes from.

## Method Summary

Directly asking a vision-language model for numeric waypoints is possible, but raw coordinates are brittle. KITE instead asks Gemma for structured, human-readable driving semantics:

- what the scene looks like
- where the drivable corridor is
- whether the lane is straight or curved
- whether to maintain speed, accelerate, or decelerate
- whether to steer left, steer right, or continue along the lane

The numeric trajectory is then produced by deterministic kinematics. This separates semantic reasoning from physical motion generation.

## Problem Setting

For each KITScenes test sample, the model receives 12 camera images from the ego vehicle:

- `front_left`, `front`, and `front_right` cameras
- last 4 frames from each camera
- frame times: `0.6s`, `0.4s`, `0.2s`, and `now/current`

The output is a 5 second ego future trajectory:

- 25 waypoints
- sampled at 5 Hz
- local ego-frame coordinates, where `x` is forward and `y` is left

## Main Method And Baselines

| Role | Method | Gemma output | Rollout |
| --- | --- | --- | --- |
| Main KITE method | `geometry_actions_bicycle` | driving actions plus lane geometry | geometry-aware bicycle model |
| Baseline 1 | `direct_waypoints_egohistory` | 25 future waypoints | none |
| Baseline 2 | `language_actions_bicycle` | driving actions only | plain bicycle model |

### Main Method: `geometry_actions_bicycle`

This is the recommended KITE method and the one with the best result in this repo. Gemma receives front-camera image history, the route instruction, and current speed/heading from a Savitzky-Golay filter. It predicts both driving actions and lane geometry.

Gemma returns JSON with an `english` object:

```json
{
  "english": {
    "situational_awareness": "...",
    "lane_direction": "...",
    "lane_curvature_strength": "...",
    "ego_position_in_lane": "...",
    "drivable_corridor": "...",
    "trajectory_shape": "...",
    "acceleration_first_3s": "...",
    "steering_first_3s": "...",
    "acceleration_last_2s": "...",
    "steering_last_2s": "..."
  }
}
```

The geometry fields are important because `steer straight` means follow the current lane or drivable corridor, not zero wheel angle. If the road curves, the geometry-aware rollout can still curve while the language action remains `steer straight`.

### Baseline 1: `direct_waypoints_egohistory`

Gemma receives front-camera image history, the route instruction, and the full past ego trajectory as text. It directly predicts 25 future `(x, y)` waypoints.

No kinematic model is used. The parsed waypoints are written directly to the submission file.

### Baseline 2: `language_actions_bicycle`

Gemma receives front-camera image history, the route instruction, and current speed/heading from a Savitzky-Golay filter. It predicts acceleration and steering language for two time windows: the first 3 seconds and the last 2 seconds.

A plain bicycle model converts those labels into 25 future waypoints. This baseline tests whether language actions alone help before adding lane geometry.

## Language To Physical Values

The code normalizes Gemma's free-form text into canonical labels. Acceleration labels become longitudinal acceleration `a`:

| Label | Low-speed value | High-speed value |
| --- | ---: | ---: |
| `decelerate_strongly` | `-2.5 m/s^2` | `-5.0 m/s^2` |
| `decelerate_slightly` | `-0.6 m/s^2` | `-1.2 m/s^2` |
| `maintain_speed` | `0.0 m/s^2` | `0.0 m/s^2` |
| `accelerate_slightly` | `0.6 m/s^2` | `1.2 m/s^2` |
| `accelerate_strongly` | `2.5 m/s^2` | `5.0 m/s^2` |

Steering labels become a base steering angle `delta_action`:

| Label | Low-speed value | High-speed value |
| --- | ---: | ---: |
| `steer_left` | `15 deg` | `0.3 deg` |
| `steer_slightly_left` | `5 deg` | `0.1 deg` |
| `steer_straight` | `0 deg` | `0 deg` |
| `steer_slightly_right` | `-5 deg` | `-0.1 deg` |
| `steer_right` | `-15 deg` | `-0.3 deg` |

The low-speed table is used up to 60 km/h. The high-speed table is used above 60 km/h.

## Geometry Steering Bias

For the main KITE method, lane geometry is converted into a steering bias:

```text
lane_direction + lane_curvature_strength + trajectory_shape -> delta_lane_bias
```

The sign comes from the predicted direction:

- left curve: positive steering bias
- right curve: negative steering bias
- straight: zero or very small steering bias

The magnitude comes mostly from the predicted curvature strength:

| Curvature text | Low-speed bias | High-speed bias |
| --- | ---: | ---: |
| `sharp` or `high` curve | `4.0 deg` | `0.08 deg` |
| `medium`, `curve`, or `arc` | `2.5 deg` | `0.05 deg` |
| `slight` or `low` curve | `1.25 deg` | `0.025 deg` |
| fallback weak curve | `0.75 deg` | `0.015 deg` |

The final steering angle is:

```text
delta_total_t = clip(delta_action_t + delta_lane_bias_t)
```

The clipping cap is `16 deg` up to 60 km/h and `0.35 deg` above 60 km/h. Steering is smoothed over the action window with an ease-in-out profile.

## Bicycle Rollout

The initial state is estimated from the past ego trajectory using Savitzky-Golay derivatives:

```text
v0 = sqrt(vx^2 + vy^2)
theta0 = atan2(vy, vx)
```

Then KITE rolls forward for 25 steps:

```text
dt = 0.2 s
L = 2.7 m

v_{t+1} = max(0, v_t + a_t dt)
theta_{t+1} = theta_t + (v_{t+1} / L) tan(delta_total_t) dt
x_{t+1} = x_t + v_{t+1} cos(theta_{t+1}) dt
y_{t+1} = y_t + v_{t+1} sin(theta_{t+1}) dt
```

The final output is a 25-point local ego-frame trajectory.

## Results

KITScenes leaderboard results:

| Method | MMS | MMS selected | MMS heavy rain | MMS construction | MMS overtake | MMS intersection | MMS night time | MMS snow | L2 mean | L2 median | S. Coherence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Kinematics V2 / KITE | 4.31 | 4.18 | 4.39 | 4.45 | 4.23 | 4.21 | 4.75 | 3.95 | 3.57 | 2.87 | 0.84 |
| Kinematic | 4.18 | 4.03 | 4.50 | 4.36 | 3.91 | 4.25 | 4.68 | 3.54 | 4.08 | 3.32 | 0.83 |
| Gemma 4 vanilla model | 3.98 | 4.15 | 4.42 | 3.96 | 3.82 | 3.61 | 4.34 | 3.57 | 5.82 | 4.11 | 0.00 |

KITE improves the overall MMS from `3.98` to `4.31`, lowers mean L2 from `5.82` to `3.57`, lowers median L2 from `4.11` to `2.87`, and raises scene coherence from `0.00` to `0.84` compared with the vanilla Gemma 4 waypoint-style baseline.

## Current Test Artifacts

The local KITE run completed all 400 KITScenes test samples for the main method and both baselines.

| Role | Method | Rows | Output |
| --- | --- | ---: | --- |
| Main KITE method | `geometry_actions_bicycle` | 400 | geometry-aware bicycle rollout |
| Baseline 1 | `direct_waypoints_egohistory` | 400 | direct waypoint prediction |
| Baseline 2 | `language_actions_bicycle` | 400 | language actions plus plain bicycle rollout |

The final geometry-aware submission artifact is:

```text
outputs/kitscenes_test_front3_last4_geometry_allsteer_savgol_reasoning_geomkin_submission_final_with_reasoning.jsonl
```

## Quick Start

Install Python dependencies:

```bash
pip install -r requirements.txt
```

List available methods:

```bash
python3 run_kite.py --list-experiments
```

Run the main KITE method:

```bash
python3 run_kite.py \
  --config config.json \
  --experiment geometry_actions_bicycle \
  --model-root /path/to/gemma-4-31B-it \
  --parquet /path/to/kitscenes-data/data/test-00000-of-00001.parquet \
  --output-root outputs/kite \
  --limit 400
```

Run the main method followed by both baselines:

```bash
python3 run_all_kite.py \
  --config config.json \
  --model-root /path/to/gemma-4-31B-it \
  --parquet /path/to/kitscenes-data/data/test-00000-of-00001.parquet \
  --output-root outputs/kite \
  --limit 400
```

Run only selected baselines:

```bash
python3 run_all_kite.py \
  --config config.json \
  --experiments direct_waypoints_egohistory language_actions_bicycle \
  --model-root /path/to/gemma-4-31B-it \
  --parquet /path/to/kitscenes-data/data/test-00000-of-00001.parquet \
  --output-root outputs/kite \
  --limit 400
```

## Outputs

New runs write:

```text
outputs/kite/<method_output_name>/
outputs/kite/<method_output_name>_submission.jsonl
outputs/kite/<method_output_name>_submission_with_reasoning.jsonl
```

## Visualization

To compare the main method and both baselines on camera images:

```bash
python3 plot_three_method_camera_trajectories.py \
  --parquet /path/to/kitscenes-data/data/test-00000-of-00001.parquet \
  --direct-dir outputs/kite/kitscenes_test_kite_direct_waypoints_egohistory \
  --language-dir outputs/kite/kitscenes_test_kite_language_actions_bicycle \
  --geometry-jsonl outputs/kite/kitscenes_test_kite_geometry_actions_bicycle_submission_with_reasoning.jsonl \
  --geometry-dir outputs/kite/kitscenes_test_kite_geometry_actions_bicycle \
  --output-dir outputs/kite/three_method_camera_trajectory_plots \
  --limit 25
```

The plot script overlays the direct waypoint baseline, language-action baseline, and main KITE geometry-aware trajectory. It also writes the driving instruction and Gemma language output into the figure.

## Method Note

A PDF version of the method description is included at:

```text
docs/KITE_Method.pdf
```

Regenerate it with:

```bash
python3 scripts/make_method_pdf.py
```
