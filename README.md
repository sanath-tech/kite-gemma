# KITE: Kinematic Interpretable Trajectory Estimation

KITE is a KITScenes trajectory-prediction pipeline built around one idea:

```text
Ask Gemma for interpretable driving intent and road geometry, then let a bicycle model turn that language into physical waypoints.
```

The default and recommended method is `geometry_actions_bicycle`. It uses Gemma-4 with front-camera image history, current motion from a Savitzky-Golay filter, language driving actions, lane geometry, and a geometry-aware bicycle rollout.

All methods use the same KITScenes visual input:

- `front_left`, `front`, and `front_right` cameras
- last 4 frames from each camera
- 12 images total
- frame times: `0.6s`, `0.4s`, `0.2s`, and `now/current`
- output: 25 future `(x, y)` waypoints at 5 Hz, covering 5 seconds

## Quick Start

Install Python dependencies:

```bash
pip install -r requirements.txt
```

List experiments:

```bash
python3 run_kite.py --list-experiments
```

Run the default KITE method locally:

```bash
python3 run_kite.py \
  --config config.json \
  --experiment geometry_actions_bicycle \
  --model-root /path/to/gemma-4-31B-it \
  --parquet /path/to/kitscenes-data/data/test-00000-of-00001.parquet \
  --output-root outputs/kite \
  --limit 400
```

Run all experiments sequentially:

```bash
python3 run_all_kite.py \
  --config config.json \
  --model-root /path/to/gemma-4-31B-it \
  --parquet /path/to/kitscenes-data/data/test-00000-of-00001.parquet \
  --output-root outputs/kite \
  --limit 400
```

Run a subset:

```bash
python3 run_all_kite.py \
  --config config.json \
  --experiments language_actions_bicycle geometry_actions_bicycle \
  --model-root /path/to/gemma-4-31B-it \
  --parquet /path/to/kitscenes-data/data/test-00000-of-00001.parquet \
  --output-root outputs/kite \
  --limit 400
```

## Experiments

| Experiment | Gemma output | Rollout | Purpose |
| --- | --- | --- | --- |
| `direct_waypoints_egohistory` | 25 future waypoints | none | Direct coordinate baseline using image history plus ego-history text |
| `language_actions_bicycle` | acceleration and steering language | plain bicycle model | Tests whether Gemma is better at predicting driving intent than raw coordinates |
| `geometry_actions_bicycle` | lane geometry plus acceleration and steering language | geometry-aware bicycle model | KITE default; follows the drivable corridor while keeping the trajectory physically smooth |

## Current Results

The local three-method KITE run completed all 400 KITScenes test samples for each active method. Leaderboard or benchmark scores should be filled in after submission/evaluation.

| Method | Rows | Submission artifact | Notes |
| --- | ---: | --- | --- |
| `direct_waypoints_egohistory` | 400 | `outputs/kite/kitscenes_test_kite_direct_waypoints_egohistory_submission.jsonl` | Direct Gemma waypoint prediction |
| `language_actions_bicycle` | 400 | `outputs/kite/kitscenes_test_kite_language_actions_bicycle_submission.jsonl` | Language actions plus plain bicycle rollout |
| `geometry_actions_bicycle` | 400 | `outputs/kite/kitscenes_test_kite_geometry_actions_bicycle_submission.jsonl` | KITE geometry-aware run |
| Final geometry-aware submission | 400 | `outputs/kitscenes_test_front3_last4_geometry_allsteer_savgol_reasoning_geomkin_submission_final_with_reasoning.jsonl` | Best working submission artifact used for the final all-steer geometry pipeline |

New runs write to:

```text
outputs/kite/<method_output_name>/
outputs/kite/<method_output_name>_submission.jsonl
outputs/kite/<method_output_name>_submission_with_reasoning.jsonl
```

## Method 1: Direct Waypoints With Ego History

`direct_waypoints_egohistory` gives Gemma:

- front3 last4 images
- driving instruction
- full past ego trajectory as text

Gemma directly returns:

```json
{
  "trajectory_reasoning": "...",
  "waypoints_5s": [[1.2, 0.0], [2.4, 0.1]]
}
```

The real output contains 25 waypoints. No kinematic model is used. The parsed waypoints are written directly to the submission file.

## Method 2: Language Actions With Bicycle Rollout

`language_actions_bicycle` gives Gemma:

- front3 last4 images
- driving instruction
- current speed from Savitzky-Golay filtering
- current heading from Savitzky-Golay filtering

Gemma returns language fields:

```json
{
  "english": {
    "situational_awareness": "...",
    "acceleration_first_3s": "...",
    "steering_first_3s": "...",
    "acceleration_last_2s": "...",
    "steering_last_2s": "..."
  }
}
```

The first 15 rollout steps use the `first_3s` labels. The last 10 steps use the `last_2s` labels.

## Method 3: KITE Geometry Actions With Bicycle Rollout

`geometry_actions_bicycle` is the recommended KITE method. Gemma receives the same inputs as Method 2, but it also predicts road geometry:

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

This is the important difference: `steer straight` means follow the current lane or drivable corridor, not zero wheel angle. If the lane curves, the geometry-aware rollout can still curve while the language action remains `steer straight`.

## From Language To Physical Values

The code first normalizes Gemma text into canonical labels.

Acceleration labels become longitudinal acceleration `a`:

| Language label | Low-speed value | High-speed value |
| --- | ---: | ---: |
| `decelerate_strongly` | `-2.5 m/s^2` | `-5.0 m/s^2` |
| `decelerate_slightly` | `-0.6 m/s^2` | `-1.2 m/s^2` |
| `maintain_speed` | `0.0 m/s^2` | `0.0 m/s^2` |
| `accelerate_slightly` | `0.6 m/s^2` | `1.2 m/s^2` |
| `accelerate_strongly` | `2.5 m/s^2` | `5.0 m/s^2` |

Steering labels become base steering angle `delta_action`:

| Language label | Low-speed value | High-speed value |
| --- | ---: | ---: |
| `steer_left` | `15 deg` | `0.3 deg` |
| `steer_slightly_left` | `5 deg` | `0.1 deg` |
| `steer_straight` | `0 deg` | `0 deg` |
| `steer_slightly_right` | `-5 deg` | `-0.1 deg` |
| `steer_right` | `-15 deg` | `-0.3 deg` |

The low-speed table is used up to 60 km/h. The high-speed table is used above 60 km/h.

For `geometry_actions_bicycle`, the geometry fields add a lane-following steering bias:

```text
lane_direction + lane_curvature_strength + trajectory_shape -> delta_lane_bias
```

The sign comes from `lane_direction` and `trajectory_shape`:

```text
left curve  -> positive steering bias
right curve -> negative steering bias
straight    -> zero or very small bias
```

The magnitude comes mostly from `lane_curvature_strength`:

| Curvature language | Low-speed bias | High-speed bias |
| --- | ---: | ---: |
| `sharp` or `high` curve | `4.0 deg` | `0.08 deg` |
| `medium`, `curve`, or `arc` | `2.5 deg` | `0.05 deg` |
| `slight` or `low` curve | `1.25 deg` | `0.025 deg` |
| fallback weak curve | `0.75 deg` | `0.015 deg` |

The final steering used by the bicycle model is:

```text
delta_total_t = clip(delta_action_t + delta_lane_bias_t)
```

The clipping cap is `16 deg` up to 60 km/h and `0.35 deg` above 60 km/h. The steering is also smoothed over the action window, so the car does not jump instantly to the target steering angle.

## Bicycle Rollout

The initial physical state is estimated from the past ego trajectory using Savitzky-Golay derivatives:

```text
v0 = sqrt(vx^2 + vy^2)
theta0 = atan2(vy, vx)
```

Then the model rolls forward for 25 steps:

```text
dt = 0.2 s
L = 2.7 m

v_{t+1} = max(0, v_t + a_t dt)
theta_{t+1} = theta_t + (v_{t+1} / L) tan(delta_total_t) dt
x_{t+1} = x_t + v_{t+1} cos(theta_{t+1}) dt
y_{t+1} = y_t + v_{t+1} sin(theta_{t+1}) dt
```

That gives the final 25 future waypoints in local ego-frame meters.

## Visualization

To compare the three active methods on camera images:

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

The plot script overlays:

- direct waypoints with ego history
- language actions plus bicycle rollout
- KITE geometry actions plus geometry-aware bicycle rollout

It also writes the driving instruction and Gemma language output into the figure.
