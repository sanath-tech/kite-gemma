#!/usr/bin/env python3
"""Plot camera images, three predicted trajectories, and model language outputs."""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import textwrap
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from PIL import Image


DT = 0.2
REFERENCE_SCORES = {
    "expert_like": 10.0,
    "wrong_speed": 7.0,
    "neglect_instruction": 4.0,
    "off_road": 1.0,
    "crash": 0.0,
}
COMFORT_CATEGORIES = {"expert_like", "wrong_speed", "neglect_instruction"}
K_FRONT = np.array([[1841.0, 0.0, 1765.0], [0.0, 1841.0, 1135.0], [0.0, 0.0, 1.0]], dtype=np.float64)
R_FRONT = np.array(
    [
        [0.01709344, -0.99983669, 0.00586657],
        [0.00538969, -0.0057752, -0.9999688],
        [0.99983937, 0.01712452, 0.00529009],
    ],
    dtype=np.float64,
)
T_FRONT = np.array([-0.0183397, -0.18646863, -0.20817565], dtype=np.float64)

METHODS = (
    ("direct_egohistory", "Gemma 4 vanilla model", "cyan"),
    ("language_bicycle", "Kinematic", "orange"),
    ("geometry_bicycle", "Kinematics V2 / KITE", "crimson"),
)
CAMERAS = (
    ("frames_camera_front_left", "front left"),
    ("frames_camera_front", "front"),
    ("frames_camera_front_right", "front right"),
)


def points_xy(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    arr = np.squeeze(arr)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim > 2:
        arr = arr.reshape(-1, arr.shape[-1])
    return arr[:, :2]


def valid_reference(value: Any) -> bool:
    try:
        pts = points_xy(value)
    except Exception:
        return False
    return len(pts) > 1 and not np.allclose(pts, -100.0)


def average_jerk(traj: np.ndarray, dt: float = DT) -> float:
    if len(traj) < 4:
        return 0.0
    return float(np.linalg.norm(np.diff(traj, n=3, axis=0) / (dt**3), axis=1).mean())


def tortuosity(traj: np.ndarray) -> float:
    if len(traj) < 2:
        return 1.0
    diffs = np.diff(traj, axis=0)
    path_len = float(np.linalg.norm(diffs, axis=1).sum())
    direct = float(np.linalg.norm(traj[-1] - traj[0]))
    return math.inf if direct < 1e-6 else path_len / direct


def trajectory_similarity(plan: np.ndarray, ref: np.ndarray) -> dict[str, Any]:
    lat_base = float(os.environ.get("MMS_LAT_BASE_M", "1.0"))
    lat_speed_gain = float(os.environ.get("MMS_LAT_SPEED_GAIN", "0.1"))
    lon_base = float(os.environ.get("MMS_LON_BASE_M", "2.0"))
    lon_speed_gain = float(os.environ.get("MMS_LON_SPEED_GAIN", "0.2"))
    reduction = os.environ.get("MMS_SIM_REDUCTION", "mean")
    n = min(len(plan), len(ref))
    plan = plan[:n]
    ref = ref[:n]
    ref_vel = np.gradient(ref, DT, axis=0)
    ref_speed = np.linalg.norm(ref_vel, axis=1)
    tangent = ref_vel / np.maximum(ref_speed[:, None], 1e-6)
    tangent[ref_speed < 1e-6] = np.array([1.0, 0.0])
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
    delta = plan - ref
    d_lon = np.abs((delta * tangent).sum(axis=1))
    d_lat = np.abs((delta * normal).sum(axis=1))
    lambda_lat = lat_base + lat_speed_gain * ref_speed
    lambda_lon = lon_base + lon_speed_gain * ref_speed
    sim_lat = np.where(d_lat <= lambda_lat, 1.0, np.maximum(0.0, 1.0 - (d_lat - lambda_lat) / lambda_lat))
    sim_lon = np.where(d_lon <= lambda_lon, 1.0, np.maximum(0.0, 1.0 - (d_lon - lambda_lon) / lambda_lon))
    per_step = np.minimum(sim_lat, sim_lon)
    sim = float(per_step.min()) if reduction == "min" else float(per_step.mean())
    return {
        "similarity": sim,
        "mean_lateral_error_m": float(d_lat.mean()),
        "mean_longitudinal_error_m": float(d_lon.mean()),
        "max_lateral_error_m": float(d_lat.max()),
        "max_longitudinal_error_m": float(d_lon.max()),
    }


def score_against_reference(plan: np.ndarray, ref: np.ndarray, category: str) -> dict[str, Any]:
    sim_info = trajectory_similarity(plan, ref)
    sim = sim_info["similarity"]
    ref_score = REFERENCE_SCORES[category]
    cp = 0
    if category in COMFORT_CATEGORIES:
        jerk_penalty = average_jerk(ref) > 1e-6 and average_jerk(plan) > 1.44 * average_jerk(ref)
        tort_penalty = math.isfinite(tortuosity(ref)) and tortuosity(plan) >= 1.06 * tortuosity(ref)
        cp = int(jerk_penalty) + int(tort_penalty)
    plan_v0 = plan[1] - plan[0]
    ref_v0 = ref[1] - ref[0]
    initial_consistency = float(np.dot(plan_v0, ref_v0))
    initial_threshold = float(0.5 * np.dot(ref_v0, ref_v0))
    if initial_consistency <= initial_threshold:
        mms = 0.0
        case = "initial_velocity_inconsistent"
    elif ref_score in {0.0, 1.0} and sim >= 0.4:
        mms = ref_score
        case = "matched_crash_or_offroad"
    elif sim * ref_score >= 3.5 - cp:
        mms = sim * ref_score
        case = "matched_reference_scaled"
    else:
        mms = 3.5 - cp
        case = "unmatched_floor"
    return {"category": category, "reference_score": ref_score, "mms": float(mms), "case": case, **sim_info}


def local_mms_for_plan(plan_xy: np.ndarray, trajectories: dict[str, Any]) -> dict[str, Any]:
    scored_refs = []
    for category in REFERENCE_SCORES:
        value = trajectories.get(category)
        if valid_reference(value):
            scored_refs.append(score_against_reference(plan_xy, points_xy(value), category))
    if not scored_refs:
        return {"status": "unavailable", "error": "no valid reference trajectories"}
    best = max(scored_refs, key=lambda item: item["mms"])
    return {
        "status": "ok",
        "local_mms": best["mms"],
        "best_category": best["category"],
        "best_similarity": best["similarity"],
        "best_case": best["case"],
        "reference_scores": scored_refs,
    }


def decode_image(frame: dict[str, Any]) -> np.ndarray:
    if frame.get("bytes") is not None:
        return np.asarray(Image.open(io.BytesIO(frame["bytes"])).convert("RGB"))
    if frame.get("path") is not None:
        return np.asarray(Image.open(frame["path"]).convert("RGB"))
    raise ValueError("Image frame has neither bytes nor path")


def project_trajectory(points: np.ndarray, z_value: float = -1.5) -> tuple[np.ndarray, np.ndarray]:
    points_3d = np.hstack([points[:, :2], np.full((len(points), 1), z_value)])
    points_cam = (R_FRONT @ points_3d.T).T + T_FRONT
    valid = points_cam[:, 2] > 0
    z = points_cam[:, 2:3].copy()
    z[~valid] = 1.0
    pixels = (K_FRONT @ (points_cam / z).T).T[:, :2]
    return pixels, valid


def draw_projected(ax: plt.Axes, pred_xy: np.ndarray, image_shape: tuple[int, ...], label: str, color: str) -> int:
    pixels, valid = project_trajectory(pred_xy)
    height, width = image_shape[:2]
    u = pixels[:, 0]
    v = pixels[:, 1]
    in_frame = valid & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    if np.any(in_frame):
        ax.plot(u[in_frame], v[in_frame], ".-", color=color, linewidth=2.1, markersize=5, label=label)
    return int(in_frame.sum())


def iter_selected_rows(parquet_path: Path, wanted_indices: set[int]) -> Any:
    parquet_file = pq.ParquetFile(parquet_path)
    idx = 0
    for batch in parquet_file.iter_batches(batch_size=1):
        for row in batch.to_pylist():
            if idx in wanted_indices:
                yield idx, row
            idx += 1


def load_dir_results(result_dir: Path) -> dict[int, dict[str, Any]]:
    results = {}
    for path in sorted(result_dir.glob("[0-9]*_*.json")):
        sample_idx = int(path.name.split("_", 1)[0])
        results[sample_idx] = json.loads(path.read_text())
    return results


def load_jsonl_by_index(path: Path) -> dict[int, dict[str, Any]]:
    results = {}
    with path.open() as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            item = json.loads(line)
            item["sample_idx"] = idx
            results[idx] = item
    return results


def trajectory_from_result(result: dict[str, Any]) -> np.ndarray:
    if "pred_xyz" in result:
        return points_xy(result["pred_xyz"])
    return points_xy(result["future_trajectory"])


def direct_text(result: dict[str, Any]) -> str:
    reasoning = result.get("reasoning") or {}
    return str(reasoning.get("trajectory_reasoning") or result.get("parsed_output", {}).get("trajectory_reasoning") or "")


def english_summary(result: dict[str, Any]) -> str:
    english = (result.get("reasoning") or {}).get("english") or {}
    parts = []
    notice = english.get("situational_awareness")
    if notice:
        parts.append(str(notice))
    actions = result.get("mapped_actions")
    if actions:
        parts.append(f"actions={actions}")
    else:
        action_fields = {
            key: english.get(key)
            for key in (
                "acceleration_first_3s",
                "steering_first_3s",
                "acceleration_last_2s",
                "steering_last_2s",
            )
            if english.get(key)
        }
        if action_fields:
            parts.append(f"actions={action_fields}")
    geometry = result.get("geometry")
    if geometry:
        parts.append(f"geometry={geometry}")
    else:
        geometry_fields = {
            key: english.get(key)
            for key in ("lane_direction", "lane_curvature_strength", "trajectory_shape")
            if english.get(key)
        }
        if geometry_fields:
            parts.append(f"geometry={geometry_fields}")
    return " | ".join(parts)


def wrap_block(title: str, text: str, width: int = 118, max_chars: int = 900) -> str:
    text = " ".join(str(text or "").split())
    if len(text) > max_chars:
        text = text[: max_chars - 3] + "..."
    wrapped = textwrap.fill(text, width=width)
    return f"{title}:\n{wrapped}"


def mms_for_result(row: dict[str, Any], result: dict[str, Any]) -> dict[str, Any] | None:
    try:
        return local_mms_for_plan(trajectory_from_result(result), row["trajectory"])
    except Exception as exc:
        return {"status": "unavailable", "error": str(exc)}


def format_mms(score: dict[str, Any] | None) -> str:
    if not score or score.get("status") != "ok":
        return "MMS n/a"
    return (
        f"MMS {float(score['local_mms']):.3f} "
        f"best_ref={score.get('best_category', 'n/a')} "
        f"sim={float(score.get('best_similarity', 0.0)):.3f}"
    )


def plot_one(
    sample_idx: int,
    row: dict[str, Any],
    direct_results: dict[int, dict[str, Any]],
    language_results: dict[int, dict[str, Any]],
    geometry_results: dict[int, dict[str, Any]],
    output_dir: Path,
) -> Path | None:
    method_results = {
        "direct_egohistory": direct_results.get(sample_idx),
        "language_bicycle": language_results.get(sample_idx),
        "geometry_bicycle": geometry_results.get(sample_idx),
    }
    if not all(method_results.values()):
        missing = [name for name, result in method_results.items() if result is None]
        print(f"skip sample_idx={sample_idx}: missing {missing}", flush=True)
        return None

    scenario_id = str(row.get("scenario_id", f"{sample_idx:07d}")).zfill(7)
    instruction = row.get("driving_instruction") or ""
    past_xy = points_xy(row["trajectory"]["past"])
    front_image = decode_image(row["frames_camera_front"][-1])
    method_mms = {key: mms_for_result(row, result) for key, result in method_results.items()}

    fig = plt.figure(figsize=(20, 15))
    grid = fig.add_gridspec(3, 3, height_ratios=[0.9, 1.15, 0.85], width_ratios=[1.0, 1.0, 1.0])
    camera_axes = [fig.add_subplot(grid[0, idx]) for idx in range(3)]
    ax_projected = fig.add_subplot(grid[1, :2])
    ax_bev = fig.add_subplot(grid[1, 2])
    ax_text = fig.add_subplot(grid[2, :])

    for ax, (column, name) in zip(camera_axes, CAMERAS):
        image = decode_image(row[column][-1])
        ax.imshow(image)
        ax.set_title(f"{name} camera now/current", fontsize=11)
        ax.axis("off")

    ax_projected.imshow(front_image)
    visible_parts = []
    for key, label, color in METHODS:
        pred_xy = trajectory_from_result(method_results[key])
        label_with_mms = f"{label} ({format_mms(method_mms[key]).split(' best_ref=')[0]})"
        visible = draw_projected(ax_projected, pred_xy, front_image.shape, label_with_mms, color)
        visible_parts.append(f"{label}: {visible}")
    height, width = front_image.shape[:2]
    ax_projected.set_xlim(0, width)
    ax_projected.set_ylim(height, 0)
    ax_projected.axis("off")
    ax_projected.legend(loc="upper right", fontsize=8)
    ax_projected.set_title("Front camera with projected trajectories | visible points: " + ", ".join(visible_parts), fontsize=11)

    ax_bev.plot(past_xy[:, 0], past_xy[:, 1], "o-", color="gold", linewidth=1.7, markersize=4, label="past")
    for key, label, color in METHODS:
        pred_xy = trajectory_from_result(method_results[key])
        label_with_mms = f"{label} ({format_mms(method_mms[key]).split(' best_ref=')[0]})"
        ax_bev.plot(pred_xy[:, 0], pred_xy[:, 1], ".-", color=color, linewidth=2.0, markersize=4, label=label_with_mms)
    ax_bev.scatter([0], [0], color="black", marker="s", s=45, label="ego now")
    ax_bev.set_aspect("equal", adjustable="box")
    ax_bev.grid(True, alpha=0.3)
    ax_bev.set_xlabel("x (m)")
    ax_bev.set_ylabel("y (m)")
    ax_bev.set_title("BEV trajectory comparison", fontsize=11)
    ax_bev.legend(loc="best", fontsize=7)

    text_blocks = [
        f"sample={sample_idx:06d} scenario={scenario_id} instruction={instruction}",
        wrap_block(
            f"Gemma 4 vanilla model | {format_mms(method_mms['direct_egohistory'])}",
            direct_text(method_results["direct_egohistory"]),
        ),
        wrap_block(
            f"Kinematic | {format_mms(method_mms['language_bicycle'])}",
            english_summary(method_results["language_bicycle"]),
        ),
        wrap_block(
            f"Kinematics V2 / KITE | {format_mms(method_mms['geometry_bicycle'])}",
            english_summary(method_results["geometry_bicycle"]),
        ),
    ]
    ax_text.axis("off")
    ax_text.text(
        0.01,
        0.98,
        "\n\n".join(text_blocks),
        ha="left",
        va="top",
        fontsize=8.5,
        family="monospace",
        transform=ax_text.transAxes,
    )

    fig.suptitle(f"KITScenes three-method comparison: {sample_idx:06d} / {scenario_id}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{sample_idx:06d}_{scenario_id}_three_methods_camera_trajectories.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--direct-dir", type=Path, required=True)
    parser.add_argument("--language-dir", type=Path, required=True)
    parser.add_argument("--geometry-jsonl", type=Path, required=True)
    parser.add_argument("--geometry-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=40)
    args = parser.parse_args()

    sample_indices = set(range(args.start, args.start + args.limit))
    direct_results = load_dir_results(args.direct_dir)
    language_results = load_dir_results(args.language_dir)
    geometry_results = load_jsonl_by_index(args.geometry_jsonl)
    if args.geometry_dir is not None and args.geometry_dir.exists():
        geometry_results.update(load_dir_results(args.geometry_dir))

    written = []
    seen_rows = set()
    for sample_idx, row in iter_selected_rows(args.parquet, sample_indices):
        seen_rows.add(sample_idx)
        output_path = plot_one(sample_idx, row, direct_results, language_results, geometry_results, args.output_dir)
        if output_path:
            written.append(str(output_path))
            print(f"wrote {output_path}", flush=True)
    for sample_idx in sorted(sample_indices - seen_rows):
        print(f"missing parquet row {sample_idx}", flush=True)

    summary = {
        "num_plots": len(written),
        "parquet": str(args.parquet),
        "direct_dir": str(args.direct_dir),
        "language_dir": str(args.language_dir),
        "geometry_jsonl": str(args.geometry_jsonl),
        "geometry_dir": str(args.geometry_dir) if args.geometry_dir else None,
        "output_dir": str(args.output_dir),
        "plots": written,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "plot_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {args.output_dir / 'plot_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
