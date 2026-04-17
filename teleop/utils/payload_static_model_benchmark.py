#!/usr/bin/env python3
"""
Model-level static benchmark for G1_29 payload compensation.

This benchmark does not require DDS, XR input, or an external simulator.
It estimates the static pose bias caused by payload torque mismatch under a
simple joint-stiffness model derived from the controller kp gains.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pinocchio as pin


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


def _safe_rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(x)
    if not np.any(finite):
        return float("nan")
    return float(np.sqrt(np.mean(np.square(x[finite]))))


def _safe_p95_abs(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(x)
    if not np.any(finite):
        return float("nan")
    return float(np.percentile(np.abs(x[finite]), 95.0))


def _safe_max_abs(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(x)
    if not np.any(finite):
        return float("nan")
    return float(np.max(np.abs(x[finite])))


def _relative(out_dir: str, path: str) -> str:
    return os.path.relpath(path, out_dir)


@dataclass
class Sample:
    q_nominal: np.ndarray
    q_before: np.ndarray
    q_after: np.ndarray
    tau_nominal: np.ndarray
    tau_payload: np.ndarray
    residual_before: np.ndarray
    residual_after: np.ndarray
    ee_nominal: np.ndarray
    ee_before: np.ndarray
    ee_after: np.ndarray
    rot_nominal: np.ndarray
    rot_before: np.ndarray
    rot_after: np.ndarray


class G129StaticBenchmark:
    def __init__(self, repo_root: str):
        self.repo_root = repo_root
        urdf_path = os.path.join(repo_root, "assets", "g1", "g1_body29_hand14.urdf")
        model_dir = os.path.join(repo_root, "assets", "g1")
        self.robot = pin.RobotWrapper.BuildFromURDF(urdf_path, model_dir)
        self.reduced_robot = self.robot.buildReducedRobot(
            list_of_joints_to_lock=LOCKED_JOINTS_G1_29,
            reference_configuration=np.zeros(self.robot.model.nq),
        )
        self._ensure_ee_frames(self.reduced_robot.model)
        self.model = self.reduced_robot.model
        self.data = self.model.createData()
        self.L_ee_id = self.model.getFrameId("L_ee")
        self.R_ee_id = self.model.getFrameId("R_ee")
        self.lower = self.model.lowerPositionLimit.copy()
        self.upper = self.model.upperPositionLimit.copy()
        self.kp_full = np.array([80.0, 80.0, 80.0, 80.0, 40.0, 40.0, 40.0] * 2, dtype=float)

    def _ensure_ee_frames(self, model) -> None:
        names = [f.name for f in model.frames]
        if "L_ee" not in names:
            model.addFrame(
                pin.Frame(
                    "L_ee",
                    model.getJointId("left_wrist_yaw_joint"),
                    pin.SE3(np.eye(3), np.array([0.05, 0.0, 0.0]).T),
                    pin.FrameType.OP_FRAME,
                )
            )
        if "R_ee" not in names:
            model.addFrame(
                pin.Frame(
                    "R_ee",
                    model.getJointId("right_wrist_yaw_joint"),
                    pin.SE3(np.eye(3), np.array([0.05, 0.0, 0.0]).T),
                    pin.FrameType.OP_FRAME,
                )
            )

    def active_slice(self, side: str) -> slice:
        return slice(0, 7) if side == "left" else slice(7, 14)

    def ee_id(self, side: str) -> int:
        return self.L_ee_id if side == "left" else self.R_ee_id

    def forward(self, q: np.ndarray, side: str) -> Tuple[np.ndarray, np.ndarray]:
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        pose = self.data.oMf[self.ee_id(side)]
        return pose.translation.copy(), pose.rotation.copy()

    def jacobian(self, q: np.ndarray, side: str) -> np.ndarray:
        return pin.computeFrameJacobian(
            self.model,
            self.data,
            q,
            self.ee_id(side),
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )

    def tau_payload(self, q: np.ndarray, side: str, mass: float, com_ee: np.ndarray, scale: float = 1.0) -> np.ndarray:
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        frame_id = self.ee_id(side)
        J = pin.computeFrameJacobian(
            self.model,
            self.data,
            q,
            frame_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )
        ee_rotation = self.data.oMf[frame_id].rotation
        r_world = ee_rotation @ com_ee
        force_world = np.array([0.0, 0.0, -9.81 * mass], dtype=float)
        moment_world = np.cross(r_world, force_world)
        wrench = np.hstack([moment_world, force_world])
        tau = scale * (J.T @ wrench)
        active = self.active_slice(side)
        inactive = slice(7, 14) if side == "left" else slice(0, 7)
        tau[inactive] = 0.0
        tau[active] = np.asarray(tau[active], dtype=float)
        return tau

    def tau_nominal(self, q: np.ndarray) -> np.ndarray:
        return np.asarray(pin.rnea(self.model, self.data, q, np.zeros(self.model.nv), np.zeros(self.model.nv)), dtype=float)

    @staticmethod
    def clip_tau(tau: np.ndarray, limit: float | None) -> np.ndarray:
        if limit is None:
            return tau.copy()
        return np.clip(tau, -limit, limit)

    @staticmethod
    def orientation_error_deg(r_nom: np.ndarray, r_actual: np.ndarray) -> float:
        r_err = r_nom @ r_actual.T
        cos_angle = float(np.clip((np.trace(r_err) - 1.0) * 0.5, -1.0, 1.0))
        return float(np.rad2deg(np.arccos(cos_angle)))

    def sample_q(self, rng: np.random.Generator, side: str, range_scale: float, workspace_filter: bool) -> np.ndarray | None:
        q = np.zeros(self.model.nq, dtype=float)
        lower = self.lower.copy() * range_scale
        upper = self.upper.copy() * range_scale
        q[:] = rng.uniform(lower, upper)
        inactive = slice(7, 14) if side == "left" else slice(0, 7)
        q[inactive] = 0.0
        if not workspace_filter:
            return q
        pos, _ = self.forward(q, side)
        x, y, z = pos.tolist()
        if side == "right":
            if not (0.10 <= x <= 0.60 and -0.55 <= y <= -0.03 and -0.10 <= z <= 0.55):
                return None
        else:
            if not (0.10 <= x <= 0.60 and 0.03 <= y <= 0.55 and -0.10 <= z <= 0.55):
                return None
        return q

    def simulate_sample(
        self,
        q: np.ndarray,
        side: str,
        payload_mass: float,
        payload_com_ee: np.ndarray,
        payload_scale: float,
        arm_tau_limit: float | None,
    ) -> Sample:
        tau_nom = self.tau_nominal(q)
        tau_payload_full = self.tau_payload(q, side, payload_mass, payload_com_ee, scale=1.0)
        tau_ff_before = self.clip_tau(tau_nom, arm_tau_limit)
        tau_ff_after = self.clip_tau(tau_nom + payload_scale * tau_payload_full, arm_tau_limit)
        tau_required = tau_nom + tau_payload_full
        residual_before = tau_required - tau_ff_before
        residual_after = tau_required - tau_ff_after

        dq_before = residual_before / self.kp_full
        dq_after = residual_after / self.kp_full

        q_before = q + dq_before
        q_after = q + dq_after

        ee_nom, rot_nom = self.forward(q, side)
        ee_before, rot_before = self.forward(q_before, side)
        ee_after, rot_after = self.forward(q_after, side)

        return Sample(
            q_nominal=q.copy(),
            q_before=q_before.copy(),
            q_after=q_after.copy(),
            tau_nominal=tau_nom.copy(),
            tau_payload=tau_payload_full.copy(),
            residual_before=residual_before.copy(),
            residual_after=residual_after.copy(),
            ee_nominal=ee_nom,
            ee_before=ee_before,
            ee_after=ee_after,
            rot_nominal=rot_nom,
            rot_before=rot_before,
            rot_after=rot_after,
        )

    def chain_points(self, q: np.ndarray, side: str) -> np.ndarray:
        joint_names = [
            f"{side}_shoulder_pitch_joint",
            f"{side}_elbow_joint",
            f"{side}_wrist_roll_joint",
            f"{side}_wrist_pitch_joint",
            f"{side}_wrist_yaw_joint",
        ]
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        points = []
        for joint_name in joint_names:
            jid = self.model.getJointId(joint_name)
            points.append(self.data.oMi[jid].translation.copy())
        points.append(self.data.oMf[self.ee_id(side)].translation.copy())
        return np.asarray(points, dtype=float)


def make_summary(samples: List[Sample], benchmark: G129StaticBenchmark, side: str) -> Dict:
    active = benchmark.active_slice(side)
    joint_before = []
    joint_after = []
    ee_pos_before = []
    ee_pos_after = []
    ee_ori_before = []
    ee_ori_after = []
    residual_before = []
    residual_after = []
    ee_positions = []

    for sample in samples:
        dq_before = np.rad2deg(sample.q_before[active] - sample.q_nominal[active])
        dq_after = np.rad2deg(sample.q_after[active] - sample.q_nominal[active])
        joint_before.append(dq_before)
        joint_after.append(dq_after)
        ee_pos_before.append(np.linalg.norm(sample.ee_before - sample.ee_nominal) * 1000.0)
        ee_pos_after.append(np.linalg.norm(sample.ee_after - sample.ee_nominal) * 1000.0)
        ee_ori_before.append(benchmark.orientation_error_deg(sample.rot_nominal, sample.rot_before))
        ee_ori_after.append(benchmark.orientation_error_deg(sample.rot_nominal, sample.rot_after))
        residual_before.append(sample.residual_before[active])
        residual_after.append(sample.residual_after[active])
        ee_positions.append(sample.ee_nominal.copy())

    joint_before = np.asarray(joint_before, dtype=float)
    joint_after = np.asarray(joint_after, dtype=float)
    ee_pos_before = np.asarray(ee_pos_before, dtype=float)
    ee_pos_after = np.asarray(ee_pos_after, dtype=float)
    ee_ori_before = np.asarray(ee_ori_before, dtype=float)
    ee_ori_after = np.asarray(ee_ori_after, dtype=float)
    residual_before = np.asarray(residual_before, dtype=float)
    residual_after = np.asarray(residual_after, dtype=float)
    ee_positions = np.asarray(ee_positions, dtype=float)

    return {
        "count": len(samples),
        "joint_error_before_rms_deg": _safe_rms(joint_before),
        "joint_error_after_rms_deg": _safe_rms(joint_after),
        "joint_error_before_p95_deg": _safe_p95_abs(joint_before),
        "joint_error_after_p95_deg": _safe_p95_abs(joint_after),
        "joint_error_before_max_deg": _safe_max_abs(joint_before),
        "joint_error_after_max_deg": _safe_max_abs(joint_after),
        "joint_error_before_rms_per_joint_deg": np.sqrt(np.mean(joint_before ** 2, axis=0)).tolist(),
        "joint_error_after_rms_per_joint_deg": np.sqrt(np.mean(joint_after ** 2, axis=0)).tolist(),
        "ee_position_before_rms_mm": _safe_rms(ee_pos_before),
        "ee_position_after_rms_mm": _safe_rms(ee_pos_after),
        "ee_position_before_p95_mm": _safe_p95_abs(ee_pos_before),
        "ee_position_after_p95_mm": _safe_p95_abs(ee_pos_after),
        "ee_position_before_max_mm": _safe_max_abs(ee_pos_before),
        "ee_position_after_max_mm": _safe_max_abs(ee_pos_after),
        "ee_orientation_before_rms_deg": _safe_rms(ee_ori_before),
        "ee_orientation_after_rms_deg": _safe_rms(ee_ori_after),
        "ee_orientation_before_p95_deg": _safe_p95_abs(ee_ori_before),
        "ee_orientation_after_p95_deg": _safe_p95_abs(ee_ori_after),
        "ee_orientation_before_max_deg": _safe_max_abs(ee_ori_before),
        "ee_orientation_after_max_deg": _safe_max_abs(ee_ori_after),
        "residual_tau_before_rms_nm": _safe_rms(residual_before),
        "residual_tau_after_rms_nm": _safe_rms(residual_after),
        "residual_tau_before_p95_nm": _safe_p95_abs(residual_before),
        "residual_tau_after_p95_nm": _safe_p95_abs(residual_after),
        "residual_tau_before_max_nm": _safe_max_abs(residual_before),
        "residual_tau_after_max_nm": _safe_max_abs(residual_after),
        "ee_position_samples_mm_before": ee_pos_before.tolist(),
        "ee_position_samples_mm_after": ee_pos_after.tolist(),
        "ee_orientation_samples_deg_before": ee_ori_before.tolist(),
        "ee_orientation_samples_deg_after": ee_ori_after.tolist(),
        "joint_error_samples_deg_before": joint_before.tolist(),
        "joint_error_samples_deg_after": joint_after.tolist(),
        "ee_workspace_xyz": ee_positions.tolist(),
    }


def _improve(before: float, after: float) -> float:
    if abs(before) < 1e-12:
        return float("nan")
    return 100.0 * (before - after) / before


def save_plots(out_dir: str, summary: Dict, samples: List[Sample], benchmark: G129StaticBenchmark, side: str) -> Dict[str, str]:
    active = benchmark.active_slice(side)
    artifacts = {}

    before_joint = np.asarray(summary["joint_error_samples_deg_before"], dtype=float)
    after_joint = np.asarray(summary["joint_error_samples_deg_after"], dtype=float)
    ee_before = np.asarray(summary["ee_position_samples_mm_before"], dtype=float)
    ee_after = np.asarray(summary["ee_position_samples_mm_after"], dtype=float)
    ori_before = np.asarray(summary["ee_orientation_samples_deg_before"], dtype=float)
    ori_after = np.asarray(summary["ee_orientation_samples_deg_after"], dtype=float)
    workspace = np.asarray(summary["ee_workspace_xyz"], dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    labels = ["Joint RMS (deg)", "EE Pos RMS (mm)", "EE Ori RMS (deg)"]
    before_vals = [
        summary["joint_error_before_rms_deg"],
        summary["ee_position_before_rms_mm"],
        summary["ee_orientation_before_rms_deg"],
    ]
    after_vals = [
        summary["joint_error_after_rms_deg"],
        summary["ee_position_after_rms_mm"],
        summary["ee_orientation_after_rms_deg"],
    ]
    for ax, label, b, a in zip(axes, labels, before_vals, after_vals):
        ax.bar(["Before", "After"], [b, a], color=["#d95f02", "#1b9e77"])
        ax.set_title(label)
        ax.grid(alpha=0.25, axis="y")
    fig.suptitle("Static Payload Compensation: Before vs After")
    fig.tight_layout()
    path = os.path.join(out_dir, "summary_bars.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    artifacts["summary_bars"] = path

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].hist(ee_before, bins=30, alpha=0.75, color="#d95f02", label="Before")
    axes[0].hist(ee_after, bins=30, alpha=0.75, color="#1b9e77", label="After")
    axes[0].set_title("EE Position Error Distribution")
    axes[0].set_xlabel("mm")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].hist(ori_before, bins=30, alpha=0.75, color="#d95f02", label="Before")
    axes[1].hist(ori_after, bins=30, alpha=0.75, color="#1b9e77", label="After")
    axes[1].set_title("EE Orientation Error Distribution")
    axes[1].set_xlabel("deg")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    path = os.path.join(out_dir, "error_histograms.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    artifacts["error_histograms"] = path

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), subplot_kw={"projection": "3d"})
    s0 = axes[0].scatter(workspace[:, 0], workspace[:, 1], workspace[:, 2], c=ee_before, cmap="magma", s=18)
    axes[0].set_title("Before: EE Position Error (mm)")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_zlabel("z")
    fig.colorbar(s0, ax=axes[0], shrink=0.72)
    s1 = axes[1].scatter(workspace[:, 0], workspace[:, 1], workspace[:, 2], c=ee_after, cmap="viridis", s=18)
    axes[1].set_title("After: EE Position Error (mm)")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].set_zlabel("z")
    fig.colorbar(s1, ax=axes[1], shrink=0.72)
    fig.tight_layout()
    path = os.path.join(out_dir, "workspace_error_scatter.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    artifacts["workspace_error_scatter"] = path

    representative_idx = int(np.argmax(ee_before))
    sample = samples[representative_idx]
    pts_nom = benchmark.chain_points(sample.q_nominal, side)
    pts_before = benchmark.chain_points(sample.q_before, side)
    pts_after = benchmark.chain_points(sample.q_after, side)
    fig = plt.figure(figsize=(6.8, 6.2))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(pts_nom[:, 0], pts_nom[:, 1], pts_nom[:, 2], "-o", color="#4c4c4c", label="Desired")
    ax.plot(pts_before[:, 0], pts_before[:, 1], pts_before[:, 2], "-o", color="#d95f02", label="Before")
    ax.plot(pts_after[:, 0], pts_after[:, 1], pts_after[:, 2], "-o", color="#1b9e77", label="After")
    ax.set_title("Representative Pose: Static Deflection")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(out_dir, "representative_pose.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    artifacts["representative_pose"] = path

    fig, ax = plt.subplots(figsize=(10, 4.2))
    joints = np.arange(1, 8)
    width = 0.38
    before_rms = np.sqrt(np.mean(before_joint ** 2, axis=0))
    after_rms = np.sqrt(np.mean(after_joint ** 2, axis=0))
    ax.bar(joints - width / 2, before_rms, width=width, color="#d95f02", label="Before")
    ax.bar(joints + width / 2, after_rms, width=width, color="#1b9e77", label="After")
    ax.set_xticks(joints)
    ax.set_xticklabels([f"J{i}" for i in joints])
    ax.set_ylabel("deg")
    ax.set_title("Per-Joint Static Deflection RMS")
    ax.legend()
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    path = os.path.join(out_dir, "joint_rms.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    artifacts["joint_rms"] = path

    return artifacts


def write_reports(out_dir: str, cfg: Dict, summary: Dict, artifacts: Dict[str, str]) -> None:
    json_path = os.path.join(out_dir, "summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"config": cfg, "summary": summary, "artifacts": {k: _relative(out_dir, v) for k, v in artifacts.items()}}, f, ensure_ascii=False, indent=2)

    lines = [
        "# G1_29 Payload Compensation Static Model Benchmark",
        "",
        "## Configuration",
        "",
        f"- side: `{cfg['side']}`",
        f"- payload_mass: `{cfg['payload_mass']}` kg",
        f"- payload_com_ee: `{cfg['payload_com_ee']}` m",
        f"- payload_scale: `{cfg['payload_scale']}`",
        f"- arm_tau_limit: `{cfg['arm_tau_limit']}`",
        f"- samples: `{cfg['num_samples']}`",
        "",
        "## Key Numbers",
        "",
        "| Metric | Before | After | Improvement |",
        "| --- | ---: | ---: | ---: |",
    ]
    row_specs = [
        ("Joint RMS (deg)", "joint_error_before_rms_deg", "joint_error_after_rms_deg"),
        ("Joint P95 (deg)", "joint_error_before_p95_deg", "joint_error_after_p95_deg"),
        ("Joint Max (deg)", "joint_error_before_max_deg", "joint_error_after_max_deg"),
        ("EE Position RMS (mm)", "ee_position_before_rms_mm", "ee_position_after_rms_mm"),
        ("EE Position P95 (mm)", "ee_position_before_p95_mm", "ee_position_after_p95_mm"),
        ("EE Position Max (mm)", "ee_position_before_max_mm", "ee_position_after_max_mm"),
        ("EE Orientation RMS (deg)", "ee_orientation_before_rms_deg", "ee_orientation_after_rms_deg"),
        ("EE Orientation P95 (deg)", "ee_orientation_before_p95_deg", "ee_orientation_after_p95_deg"),
        ("EE Orientation Max (deg)", "ee_orientation_before_max_deg", "ee_orientation_after_max_deg"),
        ("Residual Tau RMS (Nm)", "residual_tau_before_rms_nm", "residual_tau_after_rms_nm"),
    ]
    for label, kb, ka in row_specs:
        b = float(summary[kb])
        a = float(summary[ka])
        lines.append(f"| {label} | {b:.4f} | {a:.4f} | {_improve(b, a):+.2f}% |")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- 这是一个**模型级**静态 benchmark，不是机器人实机数据，也不是 Isaac 闭环仿真数据。",
            "- Before 假设前馈中没有 payload torque；After 假设前馈中加入当前实现的 payload compensation。",
            "- 关节静态偏差用 `dq ~= residual_tau / kp` 近似，再通过 FK 估算 ee 位置 / 姿态偏差。",
            "- 因此这些数字最适合回答：在现有动力学模型和 controller 刚度下，payload compensation 理论上能消掉多大一部分静态持物偏差。",
        ]
    )
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8"/>
  <title>G1_29 Payload Compensation Static Benchmark</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #1c1c1c; background: #f7f8fa; }}
    .card {{ background: white; border-radius: 12px; padding: 18px; margin-bottom: 18px; box-shadow: 0 2px 10px rgba(0,0,0,0.06); }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ border: 1px solid #d7dbe2; padding: 10px 12px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ background: #eef3f7; }}
    img {{ width: 100%; border-radius: 8px; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
    code {{ background: #eef3f7; padding: 2px 6px; border-radius: 6px; }}
  </style>
</head>
<body>
  <section class="card">
    <h1>G1_29 Payload Compensation Static Benchmark</h1>
    <p>这是一个模型级静态 benchmark，用于近似量化 payload compensation 对静态持物偏差的改善幅度。</p>
    <p>
      <strong>side:</strong> <code>{cfg['side']}</code> |
      <strong>payload_mass:</strong> <code>{cfg['payload_mass']}</code> kg |
      <strong>payload_com_ee:</strong> <code>{cfg['payload_com_ee']}</code> m |
      <strong>samples:</strong> <code>{cfg['num_samples']}</code>
    </p>
  </section>
  <section class="card">
    <h2>Key Numbers</h2>
    <table>
      <thead><tr><th>Metric</th><th>Before</th><th>After</th><th>Improvement</th></tr></thead>
      <tbody>
"""
    for label, kb, ka in row_specs:
        b = float(summary[kb])
        a = float(summary[ka])
        html += f"<tr><td>{label}</td><td>{b:.4f}</td><td>{a:.4f}</td><td>{_improve(b, a):+.2f}%</td></tr>"
    html += """
      </tbody>
    </table>
  </section>
"""
    for key in ["summary_bars", "error_histograms", "workspace_error_scatter", "representative_pose", "joint_rms"]:
        html += f'<section class="card"><h2>{key}</h2><img src="{_relative(out_dir, artifacts[key])}" alt="{key}"/></section>'
    html += """
</body>
</html>
"""
    with open(os.path.join(out_dir, "report.html"), "w", encoding="utf-8") as f:
        f.write(html)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a model-level static payload compensation benchmark for G1_29.")
    parser.add_argument("--side", choices=["left", "right"], default="right")
    parser.add_argument("--payload-mass", type=float, default=0.8)
    parser.add_argument("--payload-com", nargs=3, type=float, default=[0.02, 0.0, 0.08], metavar=("X", "Y", "Z"))
    parser.add_argument("--payload-scale", type=float, default=1.0)
    parser.add_argument("--arm-tau-limit", type=float, default=None)
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--range-scale", type=float, default=0.55, help="Scale joint limits toward zero for workspace-focused sampling.")
    parser.add_argument("--no-workspace-filter", action="store_true")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    benchmark = G129StaticBenchmark(repo_root)
    rng = np.random.default_rng(args.seed)
    payload_com = np.asarray(args.payload_com, dtype=float)

    samples: List[Sample] = []
    attempts = 0
    max_attempts = max(2000, args.num_samples * 40)
    while len(samples) < args.num_samples and attempts < max_attempts:
        attempts += 1
        q = benchmark.sample_q(
            rng,
            side=args.side,
            range_scale=args.range_scale,
            workspace_filter=not args.no_workspace_filter,
        )
        if q is None:
            continue
        samples.append(
            benchmark.simulate_sample(
                q=q,
                side=args.side,
                payload_mass=args.payload_mass,
                payload_com_ee=payload_com,
                payload_scale=args.payload_scale,
                arm_tau_limit=args.arm_tau_limit,
            )
        )

    if len(samples) == 0:
        raise RuntimeError("Failed to collect any valid benchmark samples.")

    summary = make_summary(samples, benchmark, args.side)
    cfg = {
        "side": args.side,
        "payload_mass": args.payload_mass,
        "payload_com_ee": payload_com.tolist(),
        "payload_scale": args.payload_scale,
        "arm_tau_limit": args.arm_tau_limit,
        "num_samples": len(samples),
        "seed": args.seed,
        "range_scale": args.range_scale,
        "workspace_filter": not args.no_workspace_filter,
    }
    artifacts = save_plots(args.output_dir, summary, samples, benchmark, args.side)
    write_reports(args.output_dir, cfg, summary, artifacts)

    print(f"Static benchmark written to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
