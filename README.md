# KITE: Kinematic Interpretable Trajectory Estimation

KITE is a small KITScenes inference project for comparing direct waypoint prediction with interpretable language-and-geometry trajectory generation.

Project idea in one line:

```text
Use Gemma to predict either future waypoints directly, or interpretable driving intent/road geometry that a bicycle model converts into physical trajectories.
```

This folder contains one configurable inference runner for four KITScenes test-set trajectory prediction methods. All methods use the same visual input:

- `front_left`, `front`, `front_right` cameras
- last 4 frames from each camera
- 12 images total
- frames are labeled as `0.6s`, `0.4s`, `0.2s`, and `now/current`
- output trajectory has 25 future `(x, y)` waypoints at 5 Hz, covering 5 seconds

The main runner is:

```bash
python3 run_ablation.py
```

The experiment definitions are in:

```bash
config.json
```

List available methods:

```bash
python3 run_ablation.py --list-experiments
```

Run all four methods on Slurm:

```bash
sbatch run_all_ablation.slurm
```

## Method 1: `direct_waypoints_egohistory`

Gemma receives:

- front3 last4 images
- driving instruction
- full past ego trajectory as text

Gemma directly predicts:

```json
{
  "trajectory_reasoning": "...",
  "waypoints_5s": [[x1, y1], ..., [x25, y25]]
}
```

No kinematic model is used. The predicted waypoints are directly written to the submission.

This method tests whether Gemma can infer future motion directly from images plus ego-history.

## Method 2: `direct_waypoints_motion`

Gemma receives:

- front3 last4 images
- driving instruction
- current speed from Savitzky-Golay filtering
- current heading from Savitzky-Golay filtering

Gemma directly predicts:

```json
{
  "trajectory_reasoning": "...",
  "waypoints_5s": [[x1, y1], ..., [x25, y25]]
}
```

No kinematic model is used. This method tests direct waypoint prediction when Gemma is given compact physical motion state instead of full ego-history.

## Method 3: `language_actions_bicycle`

Gemma receives:

- front3 last4 images
- driving instruction
- current speed from Savitzky-Golay filtering
- current heading from Savitzky-Golay filtering

Gemma predicts language actions:

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

The code maps language to discrete actions:

```text
maintain speed        -> maintain_speed
accelerate slightly   -> accelerate_slightly
accelerate strongly   -> accelerate_strongly
decelerate slightly   -> decelerate_slightly
decelerate strongly   -> decelerate_strongly

steer straight        -> steer_straight
steer slightly left   -> steer_slightly_left
steer left            -> steer_left
steer slightly right  -> steer_slightly_right
steer right           -> steer_right
```

Then the plain bicycle model converts actions to 25 waypoints.

The first 15 steps use the `first_3s` actions. The last 10 steps use the `last_2s` actions.

State update:

```text
v_{t+1} = max(0, v_t + a_t dt)
theta_{t+1} = theta_t + (v_{t+1} / L) tan(delta_t) dt
x_{t+1} = x_t + v_{t+1} cos(theta_{t+1}) dt
y_{t+1} = y_t + v_{t+1} sin(theta_{t+1}) dt
```

Constants:

```text
dt = 0.2 s
L = 2.7 m
NUM_POINTS = 25
```

This method tests whether Gemma is better at predicting high-level driving intent than raw coordinates.

## Method 4: `geometry_actions_bicycle`

Gemma receives the same inputs as Method 3:

- front3 last4 images
- driving instruction
- Savitzky-Golay speed
- Savitzky-Golay heading

But Gemma predicts both language actions and lane geometry:

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

The action labels are mapped exactly as in `language_actions_bicycle`.

The geometry fields add a lane-curvature steering bias:

```text
lane_direction + lane_curvature_strength + trajectory_shape
    -> lane steering bias
```

Examples:

```text
sharp/high left curve   -> positive steering bias
slight/low right curve  -> negative steering bias
straight/none           -> zero bias
```

The final steering is:

```text
delta_total = delta_language_action + delta_lane_bias
```

This means `steer straight` can still follow a curved lane. In the plain bicycle method, `steer straight` always means zero steering.

This method is the most complete pipeline: Gemma predicts intent plus road geometry, and the kinematic model converts it to physically smooth waypoints.

## Outputs

Each method writes:

```text
outputs/kite/<method_output_name>/
outputs/kite/<method_output_name>_submission.jsonl
outputs/kite/<method_output_name>_submission_with_reasoning.jsonl
```

Per-sample JSON files include:

- prompt
- raw Gemma output
- parsed reasoning
- mapped actions
- geometry, if used
- controls, if a bicycle model is used
- final `pred_xyz`

Submission JSONL files include:

```json
{
  "scenario_id": "0000000",
  "future_trajectory": [[x1, y1], ..., [x25, y25]]
}
```

The `_with_reasoning` files additionally include Gemma reasoning and, for geometry runs, mapped actions and geometry fields.

## Recommended Comparison

The four methods form this ablation sequence:

```text
direct_waypoints_egohistory
  direct coordinate prediction with full past motion

direct_waypoints_motion
  direct coordinate prediction with compact velocity/heading

language_actions_bicycle
  language driving intent + physical bicycle model

geometry_actions_bicycle
  language driving intent + lane geometry + physical bicycle model
```

If `geometry_actions_bicycle` performs best, the result suggests that Gemma is more reliable when it predicts structured driving semantics and lets the kinematic model handle physical trajectory generation.
