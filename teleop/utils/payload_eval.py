#!/usr/bin/env python3
"""
Offline before/after payload compensation evaluation for G1_29 recordings.

Inputs:
- Two episode recordings produced by teleop_hand_and_arm.py --record
- Each input can be either an episode directory or a direct path to data.json

Outputs:
- summary.json
- report.md
- report.html
- SVG plots for key metrics
- Optional side-by-side video if ffmpeg is available
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


LOCKED_JOINTS_G1_29 = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
]


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _resolve_episode_paths(path: str) -> Tuple[str, str]:
    path = os.path.abspath(path)
    if os.path.isdir(path):
        json_path = os.path.join(path, "data.json")
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Episode directory missing data.json: {path}")
        return path, json_path

    if os.path.isfile(path) and os.path.basename(path) == "data.json":
        return os.path.dirname(path), path

    raise FileNotFoundError(f"Expected an episode directory or data.json, got: {path}")


def _load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _vector_or_nan(values, size: int) -> np.ndarray:
    out = np.full(size, np.nan, dtype=float)
    if values is None:
        return out

    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return out

    count = min(size, arr.size)
    out[:count] = arr[:count]
    return out


def _safe_rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(x)
    if not np.any(finite):
        return float("nan")
    return float(np.sqrt(np.mean(np.square(x[finite]))))


def _safe_mean(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(x)
    if not np.any(finite):
        return float("nan")
    return float(np.mean(x[finite]))


def _safe_std(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(x)
    if not np.any(finite):
        return float("nan")
    return float(np.std(x[finite]))


def _safe_max_abs(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(x)
    if not np.any(finite):
        return float("nan")
    return float(np.max(np.abs(x[finite])))


def _safe_percentile_abs(x: np.ndarray, q: float) -> float:
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(x)
    if not np.any(finite):
        return float("nan")
    return float(np.percentile(np.abs(x[finite]), q))


def _improvement_pct(before: float, after: float) -> float:
    if not np.isfinite(before) or abs(before) < 1e-12 or not np.isfinite(after):
        return float("nan")
    return float(100.0 * (before - after) / before)


def _nan_to_none(x):
    if isinstance(x, float) and not np.isfinite(x):
        return None
    if isinstance(x, np.ndarray):
        return [_nan_to_none(float(v)) for v in x.tolist()]
    if isinstance(x, (list, tuple)):
        return [_nan_to_none(v) for v in x]
    if isinstance(x, dict):
        return {k: _nan_to_none(v) for k, v in x.items()}
    return x


def _html_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _relative_or_empty(base_dir: str, path: Optional[str]) -> str:
    if not path or not os.path.exists(path):
        return ""
    return os.path.relpath(path, base_dir)


def _grouped_bar_svg(
    out_path: str,
    title: str,
    labels: Sequence[str],
    series: Sequence[Tuple[str, Sequence[float], str]],
    y_label: str,
) -> None:
    width = 960
    height = 420
    left = 80
    right = 30
    top = 50
    bottom = 80
    plot_w = width - left - right
    plot_h = height - top - bottom

    flat_vals = []
    for _, values, _ in series:
        flat_vals.extend([float(v) for v in values if np.isfinite(v)])
    ymax = max(flat_vals) if flat_vals else 1.0
    ymax = ymax * 1.15 if ymax > 0 else 1.0

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-size="22" font-family="Arial">{_html_escape(title)}</text>',
        f'<text x="18" y="{height / 2}" transform="rotate(-90 18,{height / 2})" text-anchor="middle" font-size="14" font-family="Arial">{_html_escape(y_label)}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333" stroke-width="1"/>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333" stroke-width="1"/>',
    ]

    for frac in np.linspace(0.0, 1.0, 5):
        y = top + plot_h * (1.0 - frac)
        val = ymax * frac
        svg.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#ddd" stroke-width="1"/>')
        svg.append(f'<text x="{left - 10}" y="{y + 5:.1f}" text-anchor="end" font-size="12" font-family="Arial">{val:.2f}</text>')

    n_groups = max(1, len(labels))
    n_series = max(1, len(series))
    group_w = plot_w / n_groups
    bar_w = group_w / (n_series + 1)

    for i, label in enumerate(labels):
        x_label = left + group_w * (i + 0.5)
        svg.append(f'<text x="{x_label:.1f}" y="{height - 28}" text-anchor="middle" font-size="12" font-family="Arial">{_html_escape(label)}</text>')

    for s_idx, (name, values, color) in enumerate(series):
        for i, raw in enumerate(values):
            value = float(raw) if np.isfinite(raw) else 0.0
            x = left + group_w * i + bar_w * (s_idx + 0.5)
            h = plot_h * (value / ymax) if ymax > 0 else 0.0
            y = top + plot_h - h
            svg.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w * 0.8:.1f}" height="{h:.1f}" fill="{color}" opacity="0.82"/>'
            )

    legend_x = width - 220
    legend_y = 44
    for idx, (name, _, color) in enumerate(series):
        y = legend_y + 24 * idx
        svg.append(f'<rect x="{legend_x}" y="{y - 10}" width="16" height="16" fill="{color}" opacity="0.82"/>')
        svg.append(f'<text x="{legend_x + 24}" y="{y + 3}" font-size="13" font-family="Arial">{_html_escape(name)}</text>')

    svg.append("</svg>")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(svg))


def _line_plot_svg(
    out_path: str,
    title: str,
    x_label: str,
    y_label: str,
    series: Sequence[Tuple[str, np.ndarray, np.ndarray, str]],
) -> None:
    width = 960
    height = 420
    left = 80
    right = 30
    top = 50
    bottom = 80
    plot_w = width - left - right
    plot_h = height - top - bottom

    all_x = []
    all_y = []
    for _, x, y, _ in series:
        finite = np.isfinite(x) & np.isfinite(y)
        if np.any(finite):
            all_x.extend(x[finite].tolist())
            all_y.extend(y[finite].tolist())

    xmin = min(all_x) if all_x else 0.0
    xmax = max(all_x) if all_x else 1.0
    ymin = min(all_y) if all_y else 0.0
    ymax = max(all_y) if all_y else 1.0

    if abs(xmax - xmin) < 1e-9:
        xmax = xmin + 1.0
    if abs(ymax - ymin) < 1e-9:
        ymax = ymin + 1.0

    ypad = 0.05 * (ymax - ymin)
    ymin -= ypad
    ymax += ypad

    def sx(xv: float) -> float:
        return left + plot_w * (xv - xmin) / (xmax - xmin)

    def sy(yv: float) -> float:
        return top + plot_h * (1.0 - (yv - ymin) / (ymax - ymin))

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-size="22" font-family="Arial">{_html_escape(title)}</text>',
        f'<text x="{width / 2}" y="{height - 18}" text-anchor="middle" font-size="14" font-family="Arial">{_html_escape(x_label)}</text>',
        f'<text x="18" y="{height / 2}" transform="rotate(-90 18,{height / 2})" text-anchor="middle" font-size="14" font-family="Arial">{_html_escape(y_label)}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333" stroke-width="1"/>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333" stroke-width="1"/>',
    ]

    for frac in np.linspace(0.0, 1.0, 5):
        y = top + plot_h * (1.0 - frac)
        val = ymin + (ymax - ymin) * frac
        svg.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#ddd" stroke-width="1"/>')
        svg.append(f'<text x="{left - 10}" y="{y + 5:.1f}" text-anchor="end" font-size="12" font-family="Arial">{val:.2f}</text>')

    for frac in np.linspace(0.0, 1.0, 6):
        x = left + plot_w * frac
        val = xmin + (xmax - xmin) * frac
        svg.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" stroke="#f1f1f1" stroke-width="1"/>')
        svg.append(f'<text x="{x:.1f}" y="{top + plot_h + 20}" text-anchor="middle" font-size="12" font-family="Arial">{val:.2f}</text>')

    legend_x = width - 220
    legend_y = 44
    for idx, (name, x, y, color) in enumerate(series):
        finite = np.isfinite(x) & np.isfinite(y)
        if not np.any(finite):
            continue
        points = " ".join(f"{sx(float(xv)):.1f},{sy(float(yv)):.1f}" for xv, yv in zip(x[finite], y[finite]))
        svg.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.2"/>')
        ly = legend_y + 24 * idx
        svg.append(f'<line x1="{legend_x}" y1="{ly}" x2="{legend_x + 18}" y2="{ly}" stroke="{color}" stroke-width="3"/>')
        svg.append(f'<text x="{legend_x + 24}" y="{ly + 4}" font-size="13" font-family="Arial">{_html_escape(name)}</text>')

    svg.append("</svg>")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(svg))


@dataclass
class EpisodeData:
    label: str
    episode_dir: str
    json_path: str
    fps: float
    time_s: np.ndarray
    state_q: np.ndarray
    action_q: np.ndarray
    state_dq: np.ndarray
    state_tau: np.ndarray
    action_tau: np.ndarray
    tau_nominal: np.ndarray
    tau_payload: np.ndarray
    mass_hat: np.ndarray
    image_paths: Dict[str, List[str]]
    raw: Dict


class G129Kinematics:
    def __init__(self, repo_root: str):
        import pinocchio as pin

        self.pin = pin
        urdf_path = os.path.join(repo_root, "assets", "g1", "g1_body29_hand14.urdf")
        model_dir = os.path.join(repo_root, "assets", "g1")
        self.robot = pin.RobotWrapper.BuildFromURDF(urdf_path, model_dir)
        self.reduced_robot = self.robot.buildReducedRobot(
            list_of_joints_to_lock=LOCKED_JOINTS_G1_29,
            reference_configuration=np.array([0.0] * self.robot.model.nq),
        )
        self._ensure_ee_frames(self.reduced_robot.model)
        self.data = self.reduced_robot.model.createData()
        self.L_hand_id = self.reduced_robot.model.getFrameId("L_ee")
        self.R_hand_id = self.reduced_robot.model.getFrameId("R_ee")

    def _ensure_ee_frames(self, model) -> None:
        frame_names = [frame.name for frame in model.frames]
        if "L_ee" not in frame_names:
            model.addFrame(
                self.pin.Frame(
                    "L_ee",
                    model.getJointId("left_wrist_yaw_joint"),
                    self.pin.SE3(np.eye(3), np.array([0.05, 0.0, 0.0]).T),
                    self.pin.FrameType.OP_FRAME,
                )
            )
        if "R_ee" not in frame_names:
            model.addFrame(
                self.pin.Frame(
                    "R_ee",
                    model.getJointId("right_wrist_yaw_joint"),
                    self.pin.SE3(np.eye(3), np.array([0.05, 0.0, 0.0]).T),
                    self.pin.FrameType.OP_FRAME,
                )
            )

    def ee_pose(self, q14: np.ndarray, side: str) -> Tuple[np.ndarray, np.ndarray]:
        frame_id = self.L_hand_id if side == "left" else self.R_hand_id
        self.pin.forwardKinematics(self.reduced_robot.model, self.data, q14)
        self.pin.updateFramePlacements(self.reduced_robot.model, self.data)
        pose = self.data.oMf[frame_id]
        return pose.translation.copy(), pose.rotation.copy()

    @staticmethod
    def orientation_error_deg(r_target: np.ndarray, r_actual: np.ndarray) -> float:
        r_err = r_target @ r_actual.T
        cos_angle = float(np.clip((np.trace(r_err) - 1.0) * 0.5, -1.0, 1.0))
        return float(np.rad2deg(np.arccos(cos_angle)))


def load_episode(path: str, label: str) -> EpisodeData:
    episode_dir, json_path = _resolve_episode_paths(path)
    raw = _load_json(json_path)
    data_items = raw.get("data", [])
    fps = float(raw.get("info", {}).get("image", {}).get("fps", 30.0))

    n = len(data_items)
    state_q = np.full((n, 14), np.nan, dtype=float)
    action_q = np.full((n, 14), np.nan, dtype=float)
    state_dq = np.full((n, 14), np.nan, dtype=float)
    state_tau = np.full((n, 14), np.nan, dtype=float)
    action_tau = np.full((n, 14), np.nan, dtype=float)
    tau_nominal = np.full((n, 14), np.nan, dtype=float)
    tau_payload = np.full((n, 14), np.nan, dtype=float)
    mass_hat = np.full(n, np.nan, dtype=float)
    image_paths: Dict[str, List[str]] = {}

    for i, item in enumerate(data_items):
        left_state = (item.get("states", {}) or {}).get("left_arm", {}) or {}
        right_state = (item.get("states", {}) or {}).get("right_arm", {}) or {}
        left_action = (item.get("actions", {}) or {}).get("left_arm", {}) or {}
        right_action = (item.get("actions", {}) or {}).get("right_arm", {}) or {}

        state_q[i, :7] = _vector_or_nan(left_state.get("qpos", []), 7)
        state_q[i, 7:] = _vector_or_nan(right_state.get("qpos", []), 7)
        action_q[i, :7] = _vector_or_nan(left_action.get("qpos", []), 7)
        action_q[i, 7:] = _vector_or_nan(right_action.get("qpos", []), 7)
        state_dq[i, :7] = _vector_or_nan(left_state.get("qvel", []), 7)
        state_dq[i, 7:] = _vector_or_nan(right_state.get("qvel", []), 7)
        state_tau[i, :7] = _vector_or_nan(left_state.get("torque", []), 7)
        state_tau[i, 7:] = _vector_or_nan(right_state.get("torque", []), 7)
        action_tau[i, :7] = _vector_or_nan(left_action.get("torque", []), 7)
        action_tau[i, 7:] = _vector_or_nan(right_action.get("torque", []), 7)

        payload_metrics = (item.get("metrics", {}) or {}).get("payload", {}) or {}
        tau_nominal[i, :] = _vector_or_nan(payload_metrics.get("tau_nominal", []), 14)
        tau_payload[i, :] = _vector_or_nan(payload_metrics.get("tau_payload", []), 14)
        if "mass_hat" in payload_metrics:
            try:
                mass_hat[i] = float(payload_metrics["mass_hat"])
            except (TypeError, ValueError):
                pass

        for key, rel_path in (item.get("colors", {}) or {}).items():
            image_paths.setdefault(key, [])
            image_paths[key].append(os.path.join(episode_dir, rel_path) if rel_path else "")

    time_s = np.arange(n, dtype=float) / max(fps, 1e-6)
    return EpisodeData(
        label=label,
        episode_dir=episode_dir,
        json_path=json_path,
        fps=fps,
        time_s=time_s,
        state_q=state_q,
        action_q=action_q,
        state_dq=state_dq,
        state_tau=state_tau,
        action_tau=action_tau,
        tau_nominal=tau_nominal,
        tau_payload=tau_payload,
        mass_hat=mass_hat,
        image_paths=image_paths,
        raw=raw,
    )


def _finite_diff(values: np.ndarray, dt: float) -> np.ndarray:
    out = np.full_like(values, np.nan, dtype=float)
    if values.shape[0] < 2:
        return out
    diff = np.diff(values, axis=0) / dt
    out[1:] = diff
    out[0] = out[1]
    return out


def compute_summary(
    episode: EpisodeData,
    side: str,
    command_speed_threshold: float,
    measured_speed_threshold: float,
    kin: Optional[G129Kinematics],
) -> Dict:
    active_slice = slice(0, 7) if side == "left" else slice(7, 14)
    state_q_active = episode.state_q[:, active_slice]
    action_q_active = episode.action_q[:, active_slice]

    joint_err_rad = action_q_active - state_q_active
    joint_err_deg = np.rad2deg(joint_err_rad)

    valid_joint_mask = np.all(np.isfinite(joint_err_deg), axis=1)

    state_dq_active = episode.state_dq[:, active_slice]
    if not np.any(np.isfinite(state_dq_active)):
        state_dq_active = _finite_diff(state_q_active, 1.0 / max(episode.fps, 1e-6))

    cmd_dq_active = _finite_diff(action_q_active, 1.0 / max(episode.fps, 1e-6))
    cmd_speed = np.linalg.norm(np.where(np.isfinite(cmd_dq_active), cmd_dq_active, np.nan), axis=1)
    meas_speed = np.linalg.norm(np.where(np.isfinite(state_dq_active), state_dq_active, np.nan), axis=1)

    hold_mask = valid_joint_mask.copy()
    hold_mask &= np.isfinite(cmd_speed)
    hold_mask &= np.isfinite(meas_speed)
    hold_mask &= cmd_speed < command_speed_threshold
    hold_mask &= meas_speed < measured_speed_threshold

    joint_err_norm_deg = np.linalg.norm(joint_err_deg, axis=1)
    joint_rms_per_joint_deg = np.array([_safe_rms(joint_err_deg[:, j]) for j in range(7)], dtype=float)

    pos_err_mm = np.full(episode.time_s.shape[0], np.nan, dtype=float)
    ori_err_deg = np.full(episode.time_s.shape[0], np.nan, dtype=float)
    if kin is not None:
        for i in range(episode.time_s.shape[0]):
            q_state = episode.state_q[i]
            q_cmd = episode.action_q[i]
            if not np.all(np.isfinite(q_state)) or not np.all(np.isfinite(q_cmd)):
                continue
            p_cmd, r_cmd = kin.ee_pose(q_cmd, side)
            p_state, r_state = kin.ee_pose(q_state, side)
            pos_err_mm[i] = float(np.linalg.norm(p_cmd - p_state) * 1000.0)
            ori_err_deg[i] = float(kin.orientation_error_deg(r_cmd, r_state))

    tau_payload_active = episode.tau_payload[:, active_slice]
    tau_cmd_active = episode.action_tau[:, active_slice]
    tau_est_active = episode.state_tau[:, active_slice]

    summary = {
        "label": episode.label,
        "episode_dir": episode.episode_dir,
        "json_path": episode.json_path,
        "fps": float(episode.fps),
        "num_frames": int(episode.time_s.shape[0]),
        "duration_s": float(episode.time_s[-1]) if episode.time_s.size else 0.0,
        "active_side": side,
        "joint_error_rms_deg": _safe_rms(joint_err_deg),
        "joint_error_max_deg": _safe_max_abs(joint_err_deg),
        "joint_error_p95_deg": _safe_percentile_abs(joint_err_deg, 95.0),
        "joint_error_norm_rms_deg": _safe_rms(joint_err_norm_deg),
        "joint_error_norm_max_deg": _safe_max_abs(joint_err_norm_deg),
        "joint_error_rms_per_joint_deg": joint_rms_per_joint_deg,
        "hold_frame_ratio": float(np.mean(hold_mask)) if hold_mask.size else 0.0,
        "hold_joint_error_rms_deg": _safe_rms(joint_err_deg[hold_mask]),
        "hold_joint_error_max_deg": _safe_max_abs(joint_err_deg[hold_mask]),
        "hold_joint_error_norm_rms_deg": _safe_rms(joint_err_norm_deg[hold_mask]),
        "ee_position_error_rms_mm": _safe_rms(pos_err_mm),
        "ee_position_error_max_mm": _safe_max_abs(pos_err_mm),
        "ee_orientation_error_rms_deg": _safe_rms(ori_err_deg),
        "ee_orientation_error_max_deg": _safe_max_abs(ori_err_deg),
        "hold_ee_position_error_rms_mm": _safe_rms(pos_err_mm[hold_mask]),
        "hold_ee_orientation_error_rms_deg": _safe_rms(ori_err_deg[hold_mask]),
        "mean_abs_cmd_tau_nm": _safe_mean(np.abs(tau_cmd_active)),
        "mean_abs_est_tau_nm": _safe_mean(np.abs(tau_est_active)),
        "max_abs_payload_tau_nm": _safe_max_abs(tau_payload_active),
        "mean_abs_payload_tau_nm": _safe_mean(np.abs(tau_payload_active)),
        "mass_hat_mean_kg": _safe_mean(episode.mass_hat),
        "mass_hat_std_kg": _safe_std(episode.mass_hat),
    }

    return {
        "summary": summary,
        "time_s": episode.time_s,
        "joint_error_norm_deg": joint_err_norm_deg,
        "ee_position_error_mm": pos_err_mm,
        "ee_orientation_error_deg": ori_err_deg,
        "hold_mask": hold_mask,
        "shared_images": episode.image_paths,
    }


def build_comparison(before: Dict, after: Dict) -> Dict:
    lower_better = [
        "joint_error_rms_deg",
        "joint_error_max_deg",
        "joint_error_p95_deg",
        "joint_error_norm_rms_deg",
        "joint_error_norm_max_deg",
        "hold_joint_error_rms_deg",
        "hold_joint_error_max_deg",
        "hold_joint_error_norm_rms_deg",
        "ee_position_error_rms_mm",
        "ee_position_error_max_mm",
        "ee_orientation_error_rms_deg",
        "ee_orientation_error_max_deg",
        "hold_ee_position_error_rms_mm",
        "hold_ee_orientation_error_rms_deg",
    ]

    comparison = {}
    for key in lower_better:
        comparison[key] = {
            "before": before["summary"].get(key, float("nan")),
            "after": after["summary"].get(key, float("nan")),
            "improvement_pct": _improvement_pct(
                float(before["summary"].get(key, float("nan"))),
                float(after["summary"].get(key, float("nan"))),
            ),
        }

    return comparison


def _pick_frame_index(before: Dict, after: Dict) -> int:
    min_len = min(before["time_s"].shape[0], after["time_s"].shape[0])
    if min_len <= 0:
        return 0

    before_ee = before["ee_position_error_mm"][:min_len]
    after_ee = after["ee_position_error_mm"][:min_len]
    before_joint = before["joint_error_norm_deg"][:min_len]
    after_joint = after["joint_error_norm_deg"][:min_len]
    hold_mask = before["hold_mask"][:min_len] & after["hold_mask"][:min_len]

    improvement = np.where(
        np.isfinite(before_ee) & np.isfinite(after_ee),
        before_ee - after_ee,
        before_joint - after_joint,
    )
    if np.any(hold_mask & np.isfinite(improvement)):
        idx = int(np.nanargmax(np.where(hold_mask, improvement, np.nan)))
        return idx
    if np.any(np.isfinite(improvement)):
        return int(np.nanargmax(improvement))
    return min_len // 2


def _write_summary_json(out_path: str, payload: Dict) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(_nan_to_none(payload), f, ensure_ascii=False, indent=2)


def _write_report_md(out_path: str, before: Dict, after: Dict, comparison: Dict) -> None:
    rows = [
        ("关节误差 RMS (deg)", "joint_error_rms_deg"),
        ("关节误差 P95 (deg)", "joint_error_p95_deg"),
        ("关节误差范数 RMS (deg)", "joint_error_norm_rms_deg"),
        ("末端位置误差 RMS (mm)", "ee_position_error_rms_mm"),
        ("末端位置误差 Max (mm)", "ee_position_error_max_mm"),
        ("末端姿态误差 RMS (deg)", "ee_orientation_error_rms_deg"),
        ("Hold 段关节误差 RMS (deg)", "hold_joint_error_rms_deg"),
        ("Hold 段末端位置误差 RMS (mm)", "hold_ee_position_error_rms_mm"),
        ("Hold 段末端姿态误差 RMS (deg)", "hold_ee_orientation_error_rms_deg"),
    ]

    lines = [
        "# G1_29 Payload Compensation 对比报告",
        "",
        "## 核心结论",
        "",
        f"- Before: `{before['summary']['label']}`",
        f"- After: `{after['summary']['label']}`",
        f"- Active side: `{before['summary']['active_side']}`",
        f"- Before episode: `{before['summary']['episode_dir']}`",
        f"- After episode: `{after['summary']['episode_dir']}`",
        "",
        "## 关键数字",
        "",
        "| 指标 | Before | After | 改进 |",
        "| --- | ---: | ---: | ---: |",
    ]

    for label, key in rows:
        b = comparison[key]["before"]
        a = comparison[key]["after"]
        imp = comparison[key]["improvement_pct"]
        imp_text = "N/A" if not np.isfinite(imp) else f"{imp:+.2f}%"
        b_text = "N/A" if not np.isfinite(b) else f"{b:.3f}"
        a_text = "N/A" if not np.isfinite(a) else f"{a:.3f}"
        lines.append(f"| {label} | {b_text} | {a_text} | {imp_text} |")

    lines.extend(
        [
            "",
            "## 说明",
            "",
            "- 关节误差来自 `actions.left/right_arm.qpos - states.left/right_arm.qpos`。",
            "- 末端误差来自把命令关节角和实测关节角分别做 FK 后得到的 ee 位置 / 姿态差。",
            "- Hold 段由命令速度和实测速度双阈值筛选，更适合观测 payload 静态支撑能力。",
            "- 如果某项显示 `N/A`，通常表示录制中缺少该字段，或当前环境缺少相应依赖。",
        ]
    )

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _write_report_html(
    out_path: str,
    out_dir: str,
    before: Dict,
    after: Dict,
    comparison: Dict,
    svg_paths: Dict[str, str],
    frame_assets: List[Dict[str, str]],
) -> None:
    table_rows = []
    ordered_keys = [
        ("关节误差 RMS (deg)", "joint_error_rms_deg"),
        ("关节误差 P95 (deg)", "joint_error_p95_deg"),
        ("关节误差范数 RMS (deg)", "joint_error_norm_rms_deg"),
        ("末端位置误差 RMS (mm)", "ee_position_error_rms_mm"),
        ("末端位置误差 Max (mm)", "ee_position_error_max_mm"),
        ("末端姿态误差 RMS (deg)", "ee_orientation_error_rms_deg"),
        ("Hold 段关节误差 RMS (deg)", "hold_joint_error_rms_deg"),
        ("Hold 段末端位置误差 RMS (mm)", "hold_ee_position_error_rms_mm"),
        ("Hold 段末端姿态误差 RMS (deg)", "hold_ee_orientation_error_rms_deg"),
    ]
    for label, key in ordered_keys:
        row = comparison.get(key, {})
        b = row.get("before", float("nan"))
        a = row.get("after", float("nan"))
        imp = row.get("improvement_pct", float("nan"))
        table_rows.append(
            f"<tr><td>{_html_escape(label)}</td><td>{'N/A' if not np.isfinite(b) else f'{b:.3f}'}</td>"
            f"<td>{'N/A' if not np.isfinite(a) else f'{a:.3f}'}</td>"
            f"<td>{'N/A' if not np.isfinite(imp) else f'{imp:+.2f}%'}"
            "</td></tr>"
        )

    plot_cards = []
    for title, key in [
        ("关节误差范数", "joint_error_norm"),
        ("末端位置误差", "ee_position_error"),
        ("末端姿态误差", "ee_orientation_error"),
        ("各关节 RMS 误差", "joint_rms_bar"),
    ]:
        rel = _relative_or_empty(out_dir, svg_paths.get(key))
        if rel:
            plot_cards.append(
                f'<section class="card"><h3>{_html_escape(title)}</h3><img src="{_html_escape(rel)}" alt="{_html_escape(title)}"/></section>'
            )

    frame_sections = []
    for frame in frame_assets:
        video_rel = frame.get("video_rel", "")
        if video_rel:
            frame_sections.append(
                "<section class=\"card\">"
                f"<h3>{_html_escape(frame['title'])}</h3>"
                f"<video controls style=\"max-width:100%; border-radius:8px;\" src=\"{_html_escape(video_rel)}\"></video>"
                "</section>"
            )
            continue
        before_img = frame.get("before_rel", "")
        after_img = frame.get("after_rel", "")
        if not before_img or not after_img:
            continue
        frame_sections.append(
            "<section class=\"card\">"
            f"<h3>{_html_escape(frame['title'])}</h3>"
            f"<p>frame index = {frame['frame_index']}</p>"
            "<div class=\"compare-row\">"
            f"<figure><figcaption>Before</figcaption><img src=\"{_html_escape(before_img)}\" alt=\"before frame\"/></figure>"
            f"<figure><figcaption>After</figcaption><img src=\"{_html_escape(after_img)}\" alt=\"after frame\"/></figure>"
            "</div></section>"
        )

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8"/>
  <title>G1_29 Payload Compensation 对比报告</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #1b1b1b; background: #f7f8fa; }}
    h1, h2, h3 {{ margin: 0 0 12px 0; }}
    .meta {{ margin-bottom: 24px; line-height: 1.7; }}
    .card {{ background: white; border-radius: 12px; padding: 18px; margin-bottom: 18px; box-shadow: 0 2px 10px rgba(0,0,0,0.06); }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #d7dbe2; padding: 10px 12px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ background: #f0f4f8; }}
    img {{ max-width: 100%; border-radius: 8px; background: #fff; }}
    .compare-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
    figure {{ margin: 0; }}
    figcaption {{ margin-bottom: 8px; font-weight: 600; }}
  </style>
</head>
<body>
  <section class="card">
    <h1>G1_29 Payload Compensation 对比报告</h1>
    <div class="meta">
      <div><strong>Before:</strong> {_html_escape(before['summary']['label'])} | {_html_escape(before['summary']['episode_dir'])}</div>
      <div><strong>After:</strong> {_html_escape(after['summary']['label'])} | {_html_escape(after['summary']['episode_dir'])}</div>
      <div><strong>Active side:</strong> {_html_escape(before['summary']['active_side'])}</div>
      <div><strong>说明:</strong> 所有误差都以录制 episode 中的命令关节角和实测关节角为基础计算；Hold 段由低命令速度和低实测速度联合筛选。</div>
    </div>
  </section>

  <section class="card">
    <h2>关键数字</h2>
    <table>
      <thead>
        <tr><th>指标</th><th>Before</th><th>After</th><th>改进</th></tr>
      </thead>
      <tbody>
        {"".join(table_rows)}
      </tbody>
    </table>
  </section>

  {"".join(plot_cards)}
  {"".join(frame_sections)}
</body>
</html>
"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)


def _maybe_make_video(
    out_path: str,
    before_episode_dir: str,
    after_episode_dir: str,
    fps: float,
    camera_key: str,
    num_frames: int,
) -> Optional[str]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return None

    before_pattern = os.path.join(before_episode_dir, "colors", f"%06d_{camera_key}.jpg")
    after_pattern = os.path.join(after_episode_dir, "colors", f"%06d_{camera_key}.jpg")
    if not os.path.exists(before_pattern % 0) or not os.path.exists(after_pattern % 0):
        return None

    cmd = [
        ffmpeg,
        "-y",
        "-framerate",
        f"{fps:.3f}",
        "-i",
        before_pattern,
        "-framerate",
        f"{fps:.3f}",
        "-i",
        after_pattern,
        "-frames:v",
        str(num_frames),
        "-filter_complex",
        "[0:v][1:v]hstack=inputs=2[v]",
        "-map",
        "[v]",
        "-pix_fmt",
        "yuv420p",
        out_path,
    ]
    subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out_path if os.path.exists(out_path) else None


def _shared_camera_key(before: EpisodeData, after: EpisodeData, preferred: str) -> Optional[str]:
    common = sorted(set(before.image_paths.keys()) & set(after.image_paths.keys()))
    if preferred in common:
        return preferred
    return common[0] if common else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare G1_29 payload compensation recordings.")
    parser.add_argument("--before", required=True, help="Episode directory or data.json without compensation")
    parser.add_argument("--after", required=True, help="Episode directory or data.json with compensation")
    parser.add_argument("--output-dir", required=True, help="Directory to store reports and plots")
    parser.add_argument("--before-label", default="before_compensation")
    parser.add_argument("--after-label", default="after_compensation")
    parser.add_argument("--active-side", choices=["left", "right"], default="right")
    parser.add_argument("--command-speed-threshold", type=float, default=0.08, help="rad/s threshold for command-side hold detection")
    parser.add_argument("--measured-speed-threshold", type=float, default=0.15, help="rad/s threshold for measured-side hold detection")
    parser.add_argument("--camera-key", default="color_0", help="Preferred camera key for side-by-side stills/video")
    parser.add_argument("--make-video", action="store_true", help="Try to generate a side-by-side mp4 if ffmpeg is available")
    args = parser.parse_args()

    _ensure_dir(args.output_dir)

    before_episode = load_episode(args.before, args.before_label)
    after_episode = load_episode(args.after, args.after_label)

    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    kin = None
    try:
        kin = G129Kinematics(repo_root)
    except Exception:
        kin = None

    before = compute_summary(
        before_episode,
        side=args.active_side,
        command_speed_threshold=args.command_speed_threshold,
        measured_speed_threshold=args.measured_speed_threshold,
        kin=kin,
    )
    after = compute_summary(
        after_episode,
        side=args.active_side,
        command_speed_threshold=args.command_speed_threshold,
        measured_speed_threshold=args.measured_speed_threshold,
        kin=kin,
    )
    comparison = build_comparison(before, after)

    svg_paths = {}
    joint_svg = os.path.join(args.output_dir, "joint_error_norm.svg")
    _line_plot_svg(
        joint_svg,
        title="Active-Side Joint Error Norm",
        x_label="Time (s)",
        y_label="deg",
        series=[
            (before["summary"]["label"], before["time_s"], before["joint_error_norm_deg"], "#d62728"),
            (after["summary"]["label"], after["time_s"], after["joint_error_norm_deg"], "#1f77b4"),
        ],
    )
    svg_paths["joint_error_norm"] = joint_svg

    ee_pos_svg = os.path.join(args.output_dir, "ee_position_error.svg")
    _line_plot_svg(
        ee_pos_svg,
        title="End-Effector Position Error",
        x_label="Time (s)",
        y_label="mm",
        series=[
            (before["summary"]["label"], before["time_s"], before["ee_position_error_mm"], "#d62728"),
            (after["summary"]["label"], after["time_s"], after["ee_position_error_mm"], "#1f77b4"),
        ],
    )
    svg_paths["ee_position_error"] = ee_pos_svg

    ee_ori_svg = os.path.join(args.output_dir, "ee_orientation_error.svg")
    _line_plot_svg(
        ee_ori_svg,
        title="End-Effector Orientation Error",
        x_label="Time (s)",
        y_label="deg",
        series=[
            (before["summary"]["label"], before["time_s"], before["ee_orientation_error_deg"], "#d62728"),
            (after["summary"]["label"], after["time_s"], after["ee_orientation_error_deg"], "#1f77b4"),
        ],
    )
    svg_paths["ee_orientation_error"] = ee_ori_svg

    joint_bar_svg = os.path.join(args.output_dir, "joint_rms_bar.svg")
    _grouped_bar_svg(
        joint_bar_svg,
        title="Per-Joint RMS Tracking Error",
        labels=[f"J{i + 1}" for i in range(7)],
        series=[
            (before["summary"]["label"], before["summary"]["joint_error_rms_per_joint_deg"], "#d62728"),
            (after["summary"]["label"], after["summary"]["joint_error_rms_per_joint_deg"], "#1f77b4"),
        ],
        y_label="deg",
    )
    svg_paths["joint_rms_bar"] = joint_bar_svg

    shared_key = _shared_camera_key(before_episode, after_episode, args.camera_key)
    frame_assets: List[Dict[str, str]] = []
    if shared_key is not None:
        best_idx = _pick_frame_index(before, after)
        candidate_indices = []
        if min(before_episode.time_s.shape[0], after_episode.time_s.shape[0]) > 0:
            candidate_indices = [
                0,
                best_idx,
                min(before_episode.time_s.shape[0], after_episode.time_s.shape[0]) - 1,
            ]
        used = set()
        for idx in candidate_indices:
            if idx in used:
                continue
            used.add(idx)
            before_imgs = before_episode.image_paths.get(shared_key, [])
            after_imgs = after_episode.image_paths.get(shared_key, [])
            before_path = before_imgs[idx] if idx < len(before_imgs) else ""
            after_path = after_imgs[idx] if idx < len(after_imgs) else ""
            frame_assets.append(
                {
                    "title": f"{shared_key} 对比",
                    "frame_index": str(idx),
                    "before_rel": _relative_or_empty(args.output_dir, before_path),
                    "after_rel": _relative_or_empty(args.output_dir, after_path),
                }
            )

        if args.make_video:
            num_frames = min(before_episode.time_s.shape[0], after_episode.time_s.shape[0])
            video_path = _maybe_make_video(
                os.path.join(args.output_dir, f"{shared_key}_side_by_side.mp4"),
                before_episode.episode_dir,
                after_episode.episode_dir,
                fps=min(before_episode.fps, after_episode.fps),
                camera_key=shared_key,
                num_frames=num_frames,
            )
            if video_path is not None:
                frame_assets.append(
                    {
                        "title": "Side-by-side video",
                        "frame_index": "video",
                        "before_rel": "",
                        "after_rel": "",
                        "video_rel": _relative_or_empty(args.output_dir, video_path),
                    }
                )

    payload = {
        "before": before["summary"],
        "after": after["summary"],
        "comparison": comparison,
        "artifacts": {
            "svg_paths": {k: _relative_or_empty(args.output_dir, v) for k, v in svg_paths.items()},
            "camera_key": shared_key,
        },
    }

    _write_summary_json(os.path.join(args.output_dir, "summary.json"), payload)
    _write_report_md(os.path.join(args.output_dir, "report.md"), before, after, comparison)
    _write_report_html(
        os.path.join(args.output_dir, "report.html"),
        args.output_dir,
        before,
        after,
        comparison,
        svg_paths,
        frame_assets,
    )

    print(f"Payload evaluation report written to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
