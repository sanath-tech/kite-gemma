#!/usr/bin/env python3
"""Config-driven KITScenes front3-last4 ablation runner.

The four ablations differ only in prompt/output mode and rollout:
1. direct waypoints from images + ego history
2. direct waypoints from images + SavGol speed/heading
3. language actions from images + SavGol speed/heading, plain bicycle rollout
4. geometry language actions from images + SavGol speed/heading, geometry-aware bicycle rollout
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

if "--list-experiments" not in sys.argv:
    import numpy as np
    import pyarrow.parquet as pq
    import torch
    from PIL import Image
    from scipy.signal import savgol_filter
    from transformers import AutoModelForImageTextToText, AutoProcessor

    SLURM_DIR = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(SLURM_DIR))

    from evaluate_kitscenes_gemma_geometry_allsteer_kinematic import (  # noqa: E402
        geometry_from_parse,
        rollout_bicycle as rollout_allsteer,
    )
    from evaluate_kitscenes_gemma_reasoning_kinematic import (  # noqa: E402
        HIGH_SPEED_STEER_SCALE,
        LOW_SPEED_STEER_SCALE,
        STEER_PROFILE,
        WHEELBASE_M,
        actions_from_reasoning,
        extract_json_object,
        rollout_bicycle as rollout_plain,
    )


DT = 0.2
NUM_IMAGE_FRAMES = 4
NUM_WAYPOINTS = 25
CAMERA_COLUMNS = (
    ("frames_camera_front_left", "front left"),
    ("frames_camera_front", "front"),
    ("frames_camera_front_right", "front right"),
)
ACCEL_LABELS = (
    "maintain speed",
    "accelerate slightly",
    "accelerate strongly",
    "decelerate slightly",
    "decelerate strongly",
)
STEER_LABELS = (
    "steer straight",
    "steer slightly left",
    "steer left",
    "steer slightly right",
    "steer right",
)
REASONING_KEYS = (
    "situational_awareness",
    "acceleration_first_3s",
    "reason_acceleration_first_3s",
    "steering_first_3s",
    "reason_steering_first_3s",
    "acceleration_last_2s",
    "reason_acceleration_last_2s",
    "steering_last_2s",
    "reason_steering_last_2s",
)
GEOMETRY_REASONING_KEYS = (
    "situational_awareness",
    "lane_direction",
    "lane_curvature_strength",
    "ego_position_in_lane",
    "drivable_corridor",
    "trajectory_shape",
    *REASONING_KEYS[1:],
)

DEFAULT_CONFIG: dict[str, Any] = {
    "default_experiment": "geometry_actions_bicycle",
    "experiments": {
        "direct_waypoints_egohistory": {
            "description": "Gemma sees front3 last4 images plus full ego history and directly predicts 25 future xy waypoints.",
            "prompt_mode": "direct_waypoints",
            "motion_context": False,
            "ego_history": True,
            "rollout": "direct",
            "max_new_tokens": 1200,
            "output_name": "kitscenes_test_ablation_direct_waypoints_egohistory",
        },
        "direct_waypoints_motion": {
            "description": "Gemma sees front3 last4 images plus Savitzky-Golay speed/heading and directly predicts 25 future xy waypoints.",
            "prompt_mode": "direct_waypoints",
            "motion_context": True,
            "ego_history": False,
            "rollout": "direct",
            "max_new_tokens": 1200,
            "output_name": "kitscenes_test_ablation_direct_waypoints_motion",
        },
        "language_actions_bicycle": {
            "description": "Gemma sees front3 last4 images plus Savitzky-Golay speed/heading, outputs language acceleration/steering actions, then a plain bicycle rollout creates waypoints.",
            "prompt_mode": "reasoning",
            "motion_context": True,
            "ego_history": False,
            "rollout": "bestkin",
            "max_new_tokens": 450,
            "output_name": "kitscenes_test_ablation_language_actions_bicycle",
        },
        "geometry_actions_bicycle": {
            "description": "Gemma sees front3 last4 images plus Savitzky-Golay speed/heading, outputs lane geometry plus language actions, then a geometry-aware bicycle rollout creates waypoints.",
            "prompt_mode": "geometry_reasoning",
            "motion_context": True,
            "ego_history": False,
            "rollout": "allsteer_bestkin",
            "max_new_tokens": 600,
            "output_name": "kitscenes_test_ablation_geometry_actions_bicycle",
        },
    },
}


def points_xy(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    arr = np.squeeze(arr)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim > 2:
        arr = arr.reshape(-1, arr.shape[-1])
    return arr[:, :2]


def estimate_speed_heading_savgol(past_xy: np.ndarray) -> dict[str, Any]:
    window_length = min(9, len(past_xy))
    if window_length % 2 == 0:
        window_length -= 1
    if window_length < 5:
        raise ValueError(f"Need at least 5 past points for Savitzky-Golay, got {len(past_xy)}")
    polyorder = min(3, window_length - 1)
    vx = savgol_filter(past_xy[:, 0], window_length=window_length, polyorder=polyorder, deriv=1, delta=DT)
    vy = savgol_filter(past_xy[:, 1], window_length=window_length, polyorder=polyorder, deriv=1, delta=DT)
    speed = float(np.hypot(vx[-1], vy[-1]))
    heading = float(np.arctan2(vy[-1], vx[-1])) if speed > 1e-6 else 0.0
    return {
        "speed_mps": speed,
        "speed_kph": speed * 3.6,
        "heading_rad": heading,
        "source": "savgol",
        "window_length": window_length,
        "polyorder": polyorder,
        "vx_mps": float(vx[-1]),
        "vy_mps": float(vy[-1]),
    }


def format_ego_history(past_xy: np.ndarray) -> str:
    lines = [
        "Past ego trajectory sampled at 5 Hz in local coordinates, meters.",
        "x is forward/backward motion; y is lateral motion. The latest point is near the ego car now.",
    ]
    last_index = len(past_xy) - 1
    for idx, (x, y) in enumerate(past_xy):
        seconds_before_now = (last_index - idx) * DT
        time_text = "now/current" if seconds_before_now <= 1e-9 else f"{seconds_before_now:.1f} seconds before now"
        lines.append(f"- {time_text}: x={x:.2f} m, y={y:.2f} m")
    return "\n".join(lines)


def motion_text(motion: dict[str, Any]) -> str:
    return (
        f"Current speed: {motion['speed_mps']:.2f} m/s ({motion['speed_kph']:.1f} km/h)\n"
        f"Current heading angle: {motion['heading_rad']:.3f} rad\n"
    )


def build_direct_prompt(row: dict[str, Any], experiment: dict[str, Any], past_xy: np.ndarray, motion: dict[str, Any]) -> str:
    instruction = row.get("driving_instruction") or "follow the route instruction"
    context = ""
    if experiment.get("motion_context"):
        context += motion_text(motion)
    if experiment.get("ego_history"):
        context += format_ego_history(past_xy) + "\n"
    return (
        "You are driving the ego car.\n"
        "You are given 12 camera images from the ego vehicle.\n"
        "Camera layout: front left is the left-forward view, front is the straight-forward view, "
        "and front right is the right-forward view.\n"
        "Time order: images are sampled at 5 Hz, so adjacent frames are 0.2 seconds apart. "
        "The four image times are 0.6 seconds before now, 0.4 seconds before now, "
        "0.2 seconds before now, and now/current.\n"
        f"Driving instruction: {instruction}\n"
        f"{context}\n"
        "Predict the ego vehicle future trajectory for the next 5 seconds directly.\n"
        "Return exactly 25 waypoints sampled at 5 Hz. The first waypoint is 0.2 seconds in the future, "
        "and the last waypoint is 5.0 seconds in the future.\n"
        "Coordinates must be local ego-frame meters, where x is forward and y is left. "
        "Keep the waypoints smooth, physically plausible, and consistent with the images and instruction.\n\n"
        "Return only valid JSON with exactly this structure:\n"
        "{\n"
        '  "trajectory_reasoning": "...",\n'
        '  "waypoints_5s": [[x1, y1], [x2, y2], ..., [x25, y25]]\n'
        "}"
    )


def build_reasoning_prompt(row: dict[str, Any], motion: dict[str, Any]) -> str:
    instruction = row.get("driving_instruction") or "follow the route instruction"
    return (
        "You are driving the ego car.\n"
        "You are given 12 camera images from the ego vehicle.\n"
        "Camera layout: front left is the left-forward view, front is the straight-forward view, "
        "and front right is the right-forward view.\n"
        "Time order: images are sampled at 5 Hz, so adjacent frames are 0.2 seconds apart. "
        "The four image times are 0.6 seconds before now, 0.4 seconds before now, "
        "0.2 seconds before now, and now/current.\n"
        f"Driving instruction: {instruction}\n"
        f"{motion_text(motion)}"
        "Your task is to predict the vehicle's motion for the next 5 seconds.\n"
        "Choose exactly one label for each field.\n"
        f"Acceleration labels: {', '.join(ACCEL_LABELS)}\n"
        f"Steering labels: {', '.join(STEER_LABELS)}\n"
        "Answer in English. First describe what you notice in one sentence. "
        "Then choose exactly one acceleration label and one steering label for each time window. "
        "For each chosen action, give a short reason that explains the driving goal.\n\n"
        "Return only valid JSON with exactly this structure:\n"
        "{\n"
        '  "english": {\n'
        '    "situational_awareness": "...",\n'
        '    "acceleration_first_3s": "...",\n'
        '    "reason_acceleration_first_3s": "...",\n'
        '    "steering_first_3s": "...",\n'
        '    "reason_steering_first_3s": "...",\n'
        '    "acceleration_last_2s": "...",\n'
        '    "reason_acceleration_last_2s": "...",\n'
        '    "steering_last_2s": "...",\n'
        '    "reason_steering_last_2s": "..."\n'
        "  }\n"
        "}"
    )


def build_geometry_prompt(row: dict[str, Any], motion: dict[str, Any]) -> str:
    instruction = row.get("driving_instruction") or "follow the route instruction"
    return (
        "You are driving the ego car.\n"
        "You are given 12 camera images from the ego vehicle.\n"
        "Camera layout: front left is the left-forward view, front is the straight-forward view, "
        "and front right is the right-forward view.\n"
        "Time order: images are sampled at 5 Hz, so adjacent frames are 0.2 seconds apart. "
        "The four image times are 0.6 seconds before now, 0.4 seconds before now, "
        "0.2 seconds before now, and now/current.\n\n"
        f"Driving instruction: {instruction}\n"
        f"{motion_text(motion)}\n"
        "Your task is to predict the vehicle motion for the next 5 seconds.\n"
        "First estimate the road/lane geometry ahead from the camera images. "
        "Treat the drivable corridor centerline as the base path the vehicle should follow.\n\n"
        "Important steering rule:\n"
        "'steer straight' does NOT mean zero yaw rate. "
        "'steer straight' means follow the current lane or drivable corridor centerline. "
        "If the road curves, the future trajectory should curve with the road while still being considered straight driving.\n\n"
        "Action timing rule:\n"
        "Do not start a turn or lane change just because the instruction mentions it. "
        "Start the maneuver only when the images and current motion show the ego vehicle is at the maneuver point.\n\n"
        "Estimate these extra fields:\n"
        "- lane_direction: one of [straight, slight_left_curve, left_curve, sharp_left_curve, "
        "slight_right_curve, right_curve, sharp_right_curve]\n"
        "- lane_curvature_strength: one of [none, very_low, low, medium, high]\n"
        "- ego_position_in_lane: one of [centered, slightly_left, left, slightly_right, right, unknown]\n"
        "- drivable_corridor: short description of where the vehicle can safely drive\n"
        "- trajectory_shape: one of [straight_line, slight_left_arc, left_arc, slight_right_arc, right_arc, s_curve]\n\n"
        "Choose exactly one label for each action field.\n"
        "Choose steering labels as the ego vehicle's planned motion relative to the drivable corridor, "
        "not as raw wheel angle.\n"
        f"Acceleration labels: {', '.join(ACCEL_LABELS)}\n"
        f"Steering labels: {', '.join(STEER_LABELS)}\n\n"
        "Return only valid JSON with exactly this structure:\n"
        "{\n"
        '  "english": {\n'
        '    "situational_awareness": "...",\n'
        '    "lane_direction": "...",\n'
        '    "lane_curvature_strength": "...",\n'
        '    "ego_position_in_lane": "...",\n'
        '    "drivable_corridor": "...",\n'
        '    "trajectory_shape": "...",\n'
        '    "acceleration_first_3s": "...",\n'
        '    "reason_acceleration_first_3s": "...",\n'
        '    "steering_first_3s": "...",\n'
        '    "reason_steering_first_3s": "...",\n'
        '    "acceleration_last_2s": "...",\n'
        '    "reason_acceleration_last_2s": "...",\n'
        '    "steering_last_2s": "...",\n'
        '    "reason_steering_last_2s": "..."\n'
        "  }\n"
        "}"
    )


def build_prompt(row: dict[str, Any], experiment: dict[str, Any], past_xy: np.ndarray, motion: dict[str, Any]) -> str:
    prompt_mode = experiment["prompt_mode"]
    if prompt_mode == "direct_waypoints":
        return build_direct_prompt(row, experiment, past_xy, motion)
    if prompt_mode == "reasoning":
        return build_reasoning_prompt(row, motion)
    if prompt_mode == "geometry_reasoning":
        return build_geometry_prompt(row, motion)
    raise ValueError(f"Unknown prompt_mode={prompt_mode!r}")


def decode_frame(frame: dict[str, Any]) -> Image.Image:
    if frame.get("bytes") is not None:
        return Image.open(io.BytesIO(frame["bytes"])).convert("RGB")
    if frame.get("path") is not None:
        return Image.open(frame["path"]).convert("RGB")
    raise ValueError("Image frame has neither bytes nor path")


def build_messages(row: dict[str, Any], prompt: str) -> list[dict[str, Any]]:
    content = []
    time_labels = ["0.6 seconds before now", "0.4 seconds before now", "0.2 seconds before now", "now/current"]
    for frame_offset, time_label in enumerate(time_labels):
        for column, name in CAMERA_COLUMNS:
            frames = row[column][-NUM_IMAGE_FRAMES:]
            content.append({"type": "text", "text": f"{name} camera, {time_label}:"})
            content.append({"type": "image", "image": decode_frame(frames[frame_offset])})
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def normalize_english(parsed: dict[str, Any], keys: tuple[str, ...]) -> dict[str, str]:
    english = parsed.get("english", parsed)
    if not isinstance(english, dict):
        raise ValueError("Gemma output does not contain an english object")
    return {key: str(english.get(key) or "") for key in keys}


def parse_waypoints(raw_output: str) -> tuple[list[list[float]], dict[str, Any], str]:
    parsed = extract_json_object(raw_output)
    waypoints = parsed.get("waypoints_5s", parsed.get("future_trajectory"))
    if not isinstance(waypoints, list) or not waypoints:
        raise ValueError("Direct waypoint output did not contain waypoints_5s or future_trajectory")
    xy: list[list[float]] = []
    for point in waypoints:
        if isinstance(point, dict):
            x = point.get("x", point.get("x_m"))
            y = point.get("y", point.get("y_m"))
        else:
            x, y = point[0], point[1]
        xy.append([float(x), float(y)])
    if len(xy) > NUM_WAYPOINTS:
        xy = xy[:NUM_WAYPOINTS]
    while len(xy) < NUM_WAYPOINTS:
        xy.append(list(xy[-1]))
    reasoning = str(parsed.get("trajectory_reasoning") or parsed.get("reasoning") or "")
    return xy, parsed, reasoning


def pad_xyz(xy: np.ndarray | list[list[float]]) -> list[list[float]]:
    arr = np.asarray(xy, dtype=np.float64)
    zeros = np.zeros((len(arr), 1), dtype=np.float64)
    return np.concatenate([arr[:, :2], zeros], axis=1).tolist()


def iter_parquet_rows(parquet_path: Path, start: int, limit: int | None) -> Any:
    parquet_file = pq.ParquetFile(parquet_path)
    seen = 0
    emitted = 0
    for batch in parquet_file.iter_batches(batch_size=1):
        for row in batch.to_pylist():
            if seen < start:
                seen += 1
                continue
            if limit is not None and emitted >= limit:
                return
            yield seen, row
            seen += 1
            emitted += 1


def generate_raw_output(row: dict[str, Any], prompt: str, processor: Any, model: Any, max_new_tokens: int) -> str:
    messages = build_messages(row, prompt)
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    input_len = inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.decode(outputs[0][input_len:], skip_special_tokens=True).strip()


def apply_rollout(raw_output: str, experiment: dict[str, Any], past_xy: np.ndarray) -> dict[str, Any]:
    rollout_mode = experiment["rollout"]
    if rollout_mode == "direct":
        xy, parsed, reasoning = parse_waypoints(raw_output)
        return {
            "pred_xy": xy,
            "parsed_output": parsed,
            "reasoning": {"trajectory_reasoning": reasoning},
            "mapped_actions": None,
            "geometry": None,
            "controls": None,
            "initial_state": None,
        }

    parsed = extract_json_object(raw_output)
    if rollout_mode == "bestkin":
        english = normalize_english(parsed, REASONING_KEYS)
        actions = actions_from_reasoning(english)
        pred_xy, controls, initial_state = rollout_plain(past_xy, actions)
        return {
            "pred_xy": pred_xy.tolist(),
            "parsed_output": parsed,
            "reasoning": {"english": english},
            "mapped_actions": actions,
            "geometry": None,
            "controls": controls,
            "initial_state": initial_state,
        }

    if rollout_mode == "allsteer_bestkin":
        english = normalize_english(parsed, GEOMETRY_REASONING_KEYS)
        actions = actions_from_reasoning(english)
        gemma_parse = {"parsed": parsed, "english": english}
        geometry = geometry_from_parse(gemma_parse)
        pred_xy, controls, initial_state = rollout_allsteer(past_xy, actions, geometry)
        return {
            "pred_xy": pred_xy.tolist(),
            "parsed_output": parsed,
            "reasoning": {"english": english},
            "mapped_actions": actions,
            "geometry": geometry,
            "controls": controls,
            "initial_state": initial_state,
        }

    raise ValueError(f"Unknown rollout={rollout_mode!r}")


def generate_one(
    sample_idx: int,
    row: dict[str, Any],
    experiment_name: str,
    experiment: dict[str, Any],
    processor: Any,
    model: Any,
    output_dir: Path,
    max_new_tokens: int,
) -> dict[str, Any]:
    scenario_id = str(row.get("scenario_id", f"{sample_idx:07d}")).zfill(7)
    past_xy = points_xy(row["trajectory"]["past"])
    motion = estimate_speed_heading_savgol(past_xy)
    prompt = build_prompt(row, experiment, past_xy, motion)
    raw_output = generate_raw_output(row, prompt, processor, model, max_new_tokens)
    rollout_result = apply_rollout(raw_output, experiment, past_xy)

    result = {
        "status": "ok",
        "sample_idx": sample_idx,
        "scenario_id": scenario_id,
        "driving_instruction": row.get("driving_instruction"),
        "scenario_type": row.get("scenario_type"),
        "experiment_name": experiment_name,
        "experiment": experiment,
        "include_images": True,
        "camera_columns": [column for column, _ in CAMERA_COLUMNS],
        "num_image_frames_per_camera": NUM_IMAGE_FRAMES,
        "motion_context": motion if experiment.get("motion_context") else None,
        "ego_history_in_prompt": bool(experiment.get("ego_history")),
        "prompt": prompt,
        "raw_output": raw_output,
        "parsed_output": rollout_result["parsed_output"],
        "reasoning": rollout_result["reasoning"],
        "mapped_actions": rollout_result["mapped_actions"],
        "geometry": rollout_result["geometry"],
        "kinematic_profile": {
            "rollout": experiment["rollout"],
            "steer_profile": STEER_PROFILE,
            "low_speed_steer_scale": LOW_SPEED_STEER_SCALE,
            "high_speed_steer_scale": HIGH_SPEED_STEER_SCALE,
            "wheelbase_m": WHEELBASE_M,
        },
        "initial_state": rollout_result["initial_state"],
        "controls": rollout_result["controls"],
        "pred_xyz": pad_xyz(rollout_result["pred_xy"]),
    }
    output_path = output_dir / f"{sample_idx:06d}_{scenario_id}.json"
    output_path.write_text(json.dumps(result, indent=2))
    return result


def write_submission(output_path: Path, results: list[dict[str, Any]], include_reasoning: bool) -> None:
    with output_path.open("w") as out_file:
        for result in sorted(results, key=lambda item: item["sample_idx"]):
            item = {
                "scenario_id": str(result["scenario_id"]).zfill(7),
                "future_trajectory": [[float(point[0]), float(point[1])] for point in result["pred_xyz"]],
            }
            if include_reasoning:
                item["reasoning"] = result["reasoning"]
                if result.get("geometry") is not None:
                    item["geometry"] = result["geometry"]
                if result.get("mapped_actions") is not None:
                    item["mapped_actions"] = result["mapped_actions"]
            out_file.write(json.dumps(item) + "\n")


def load_config(config_path: Path | None, experiment_override: str | None) -> tuple[str, dict[str, Any], dict[str, Any]]:
    if config_path is None:
        config = DEFAULT_CONFIG
    else:
        config = json.loads(config_path.read_text())
    experiment_name = experiment_override or config["default_experiment"]
    experiments = config.get("experiments", {})
    if experiment_name not in experiments:
        raise ValueError(f"Unknown experiment {experiment_name!r}. Choices: {', '.join(sorted(experiments))}")
    return experiment_name, experiments[experiment_name], config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None, help="Optional JSON config. If omitted, uses the built-in default config.")
    parser.add_argument("--experiment", default=None, help="Experiment/model name from the config.")
    parser.add_argument("--list-experiments", action="store_true", help="Print available experiment names and exit.")
    parser.add_argument("--model-root", type=Path, default=None)
    parser.add_argument("--parquet", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/kite"))
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    args = parser.parse_args()

    experiment_name, experiment, full_config = load_config(args.config, args.experiment)
    if args.list_experiments:
        for name, item in full_config["experiments"].items():
            default_marker = " (default)" if name == full_config.get("default_experiment") else ""
            print(f"{name}{default_marker}: {item.get('description', '')}")
        return
    if args.model_root is None:
        raise ValueError("--model-root is required unless --list-experiments is used")
    if args.parquet is None:
        raise ValueError("--parquet is required unless --list-experiments is used")

    output_dir = args.output_root / experiment["output_name"]
    submission = args.output_root / f"{experiment['output_name']}_submission.jsonl"
    submission_with_reasoning = args.output_root / f"{experiment['output_name']}_submission_with_reasoning.jsonl"
    max_new_tokens = int(args.max_new_tokens or experiment.get("max_new_tokens", 600))

    output_dir.mkdir(parents=True, exist_ok=True)
    submission.parent.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "selected_experiment": experiment_name,
                "experiment": experiment,
                "full_config": full_config,
                "start": args.start,
                "limit": args.limit,
                "max_new_tokens": max_new_tokens,
                "parquet": str(args.parquet),
            },
            indent=2,
        )
    )

    print(f"selected_experiment={experiment_name}", flush=True)
    print(f"description={experiment.get('description')}", flush=True)
    print(f"output_dir={output_dir}", flush=True)
    print(f"submission={submission}", flush=True)
    print("Loading processor...", flush=True)
    processor = AutoProcessor.from_pretrained(str(args.model_root), local_files_only=True)

    print("Loading model...", flush=True)
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "local_files_only": True,
    }
    attn_implementation = os.environ.get("GEMMA_ATTN_IMPLEMENTATION")
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation
        print(f"Using attn_implementation={attn_implementation}", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(str(args.model_root), **model_kwargs)
    model.eval()
    print("Model loaded.", flush=True)

    results = []
    for sample_idx, row in iter_parquet_rows(args.parquet, args.start, args.limit):
        print(f"Running {experiment_name} sample_idx={sample_idx} scenario_id={row.get('scenario_id')}", flush=True)
        result = generate_one(
            sample_idx,
            row,
            experiment_name,
            experiment,
            processor,
            model,
            output_dir,
            max_new_tokens,
        )
        results.append(result)
        result_path = output_dir / f"{sample_idx:06d}_{result['scenario_id']}.json"
        print(f"Wrote {result_path}", flush=True)
        write_submission(submission, results, include_reasoning=False)
        write_submission(submission_with_reasoning, results, include_reasoning=True)
        print(f"Updated partial submissions with {len(results)} samples", flush=True)

    write_submission(submission, results, include_reasoning=False)
    write_submission(submission_with_reasoning, results, include_reasoning=True)
    summary = {
        "selected_experiment": experiment_name,
        "experiment": experiment,
        "num_samples": len(results),
        "start": args.start,
        "limit": args.limit,
        "max_new_tokens": max_new_tokens,
        "output_dir": str(output_dir),
        "submission": str(submission),
        "submission_with_reasoning": str(submission_with_reasoning),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote {submission}", flush=True)
    print(f"Wrote {submission_with_reasoning}", flush=True)
    print(f"Wrote {output_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
