#!/usr/bin/env python3
"""Plot camera images, three predicted trajectories, and model language outputs."""

from __future__ import annotations

import argparse
import io
import json
import textwrap
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from PIL import Image


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
        visible = draw_projected(ax_projected, pred_xy, front_image.shape, label, color)
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
        ax_bev.plot(pred_xy[:, 0], pred_xy[:, 1], ".-", color=color, linewidth=2.0, markersize=4, label=label)
    ax_bev.scatter([0], [0], color="black", marker="s", s=45, label="ego now")
    ax_bev.set_aspect("equal", adjustable="box")
    ax_bev.grid(True, alpha=0.3)
    ax_bev.set_xlabel("x (m)")
    ax_bev.set_ylabel("y (m)")
    ax_bev.set_title("BEV trajectory comparison", fontsize=11)
    ax_bev.legend(loc="best", fontsize=7)

    text_blocks = [
        f"sample={sample_idx:06d} scenario={scenario_id} instruction={instruction}",
        wrap_block("Gemma 4 vanilla model", direct_text(method_results["direct_egohistory"])),
        wrap_block("Kinematic", english_summary(method_results["language_bicycle"])),
        wrap_block("Kinematics V2 / KITE", english_summary(method_results["geometry_bicycle"])),
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
