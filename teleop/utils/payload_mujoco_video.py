#!/usr/bin/env python3
"""
Render a representative G1_29 payload-compensation comparison video with MuJoCo.

The video is generated from the same model-level static benchmark used for the
payload compensation report. It selects the worst cases (by pre-compensation ee
position error) and renders three synchronized panels:

1. Target pose
2. Without compensation
3. With compensation

This gives a compact visual summary of how much static deflection is removed by
the payload feedforward term.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import site
import sys
from dataclasses import dataclass
from typing import Dict, List

import cv2
import numpy as np

# Prefer the environment's MuJoCo over any user-site install.
USER_SITE = site.getusersitepackages()
if isinstance(USER_SITE, str) and USER_SITE in sys.path:
    sys.path.remove(USER_SITE)
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from teleop.utils.payload_static_model_benchmark import G129StaticBenchmark


ARM_JOINTS = {
    "left": [
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
    ],
    "right": [
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ],
}


@dataclass
class CaseData:
    index: int
    q_nominal: np.ndarray
    q_before: np.ndarray
    q_after: np.ndarray
    ee_before_mm: float
    ee_after_mm: float
    ee_ori_before_deg: float
    ee_ori_after_deg: float
    joint_rms_before_deg: float
    joint_rms_after_deg: float
    residual_tau_before_nm: float
    residual_tau_after_nm: float


def _safe_float(x: float) -> float:
    return float(np.asarray(x, dtype=float))


def _put_line(img: np.ndarray, text: str, xy: tuple[int, int], color: tuple[int, int, int], scale: float = 0.72, thick: int = 2) -> None:
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (20, 20, 20), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def select_cases(
    benchmark: G129StaticBenchmark,
    side: str,
    payload_mass: float,
    payload_com: np.ndarray,
    payload_scale: float,
    arm_tau_limit: float | None,
    num_samples: int,
    seed: int,
    range_scale: float,
    top_k: int,
) -> List[CaseData]:
    rng = np.random.default_rng(seed)
    samples = []
    attempts = 0
    max_attempts = max(2000, num_samples * 40)
    while len(samples) < num_samples and attempts < max_attempts:
        attempts += 1
        q = benchmark.sample_q(rng, side=side, range_scale=range_scale, workspace_filter=True)
        if q is None:
            continue
        samples.append(
            benchmark.simulate_sample(
                q=q,
                side=side,
                payload_mass=payload_mass,
                payload_com_ee=payload_com,
                payload_scale=payload_scale,
                arm_tau_limit=arm_tau_limit,
            )
        )

    if not samples:
        raise RuntimeError("No valid benchmark samples were generated for video rendering.")

    active = benchmark.active_slice(side)
    ranked = []
    for idx, sample in enumerate(samples):
        ee_before_mm = np.linalg.norm(sample.ee_before - sample.ee_nominal) * 1000.0
        ee_after_mm = np.linalg.norm(sample.ee_after - sample.ee_nominal) * 1000.0
        ee_ori_before_deg = benchmark.orientation_error_deg(sample.rot_nominal, sample.rot_before)
        ee_ori_after_deg = benchmark.orientation_error_deg(sample.rot_nominal, sample.rot_after)
        joint_rms_before_deg = np.sqrt(np.mean(np.rad2deg(sample.q_before[active] - sample.q_nominal[active]) ** 2))
        joint_rms_after_deg = np.sqrt(np.mean(np.rad2deg(sample.q_after[active] - sample.q_nominal[active]) ** 2))
        residual_tau_before_nm = np.sqrt(np.mean(sample.residual_before[active] ** 2))
        residual_tau_after_nm = np.sqrt(np.mean(sample.residual_after[active] ** 2))
        ranked.append(
            CaseData(
                index=idx,
                q_nominal=sample.q_nominal.copy(),
                q_before=sample.q_before.copy(),
                q_after=sample.q_after.copy(),
                ee_before_mm=_safe_float(ee_before_mm),
                ee_after_mm=_safe_float(ee_after_mm),
                ee_ori_before_deg=_safe_float(ee_ori_before_deg),
                ee_ori_after_deg=_safe_float(ee_ori_after_deg),
                joint_rms_before_deg=_safe_float(joint_rms_before_deg),
                joint_rms_after_deg=_safe_float(joint_rms_after_deg),
                residual_tau_before_nm=_safe_float(residual_tau_before_nm),
                residual_tau_after_nm=_safe_float(residual_tau_after_nm),
            )
        )

    ranked.sort(key=lambda case: case.ee_before_mm, reverse=True)
    return ranked[: max(1, top_k)]


class MujocoRenderer:
    def __init__(self, xml_path: str, side: str, width: int, height: int):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.side = side
        self.width = width
        self.height = height
        self.renderer = mujoco.Renderer(self.model, width=width, height=height)
        self.camera = mujoco.MjvCamera()
        self.camera.distance = 1.72
        self.camera.azimuth = 160.0 if side == "right" else 20.0
        self.camera.elevation = -16.0
        self.camera.lookat[:] = np.array([0.18, -0.02 if side == "left" else 0.02, 0.98], dtype=float)
        self._joint_qpos_addr = self._build_joint_qpos_addr()
        # Match the nominal floating-base pose in the MJCF model.
        self._base_qpos = np.array([0.0, 0.0, 0.793, 1.0, 0.0, 0.0, 0.0], dtype=float)

    def _build_joint_qpos_addr(self) -> Dict[str, int]:
        mapping = {}
        for joint_name in ARM_JOINTS["left"] + ARM_JOINTS["right"]:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0:
                raise KeyError(f"Joint not found in MuJoCo model: {joint_name}")
            mapping[joint_name] = int(self.model.jnt_qposadr[joint_id])
        return mapping

    def render_pose(self, q14: np.ndarray) -> np.ndarray:
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self.data.qpos[:7] = self._base_qpos
        left_vals = np.asarray(q14[:7], dtype=float)
        right_vals = np.asarray(q14[7:14], dtype=float)
        for joint_name, joint_val in zip(ARM_JOINTS["left"], left_vals):
            self.data.qpos[self._joint_qpos_addr[joint_name]] = float(joint_val)
        for joint_name, joint_val in zip(ARM_JOINTS["right"], right_vals):
            self.data.qpos[self._joint_qpos_addr[joint_name]] = float(joint_val)
        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, camera=self.camera)
        return self.renderer.render().copy()

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


def compose_frame(
    target_img: np.ndarray,
    before_img: np.ndarray,
    after_img: np.ndarray,
    case_idx: int,
    total_cases: int,
    case: CaseData,
    side: str,
    payload_mass: float,
    payload_com: np.ndarray,
) -> np.ndarray:
    panel_h, panel_w = target_img.shape[:2]
    header_h = 120
    footer_h = 56
    out = np.full((header_h + panel_h + footer_h, panel_w * 3, 3), 247, dtype=np.uint8)
    out[header_h : header_h + panel_h, 0:panel_w] = cv2.cvtColor(target_img, cv2.COLOR_RGB2BGR)
    out[header_h : header_h + panel_h, panel_w : 2 * panel_w] = cv2.cvtColor(before_img, cv2.COLOR_RGB2BGR)
    out[header_h : header_h + panel_h, 2 * panel_w : 3 * panel_w] = cv2.cvtColor(after_img, cv2.COLOR_RGB2BGR)

    cv2.rectangle(out, (0, 0), (out.shape[1], header_h), (20, 26, 38), -1)
    _put_line(out, "G1_29 Payload Compensation Visual Comparison", (32, 42), (238, 240, 244), scale=1.0, thick=2)
    subtitle = (
        f"Case {case_idx + 1}/{total_cases} | side={side} | payload={payload_mass:.2f} kg | "
        f"com_ee=[{payload_com[0]:.3f}, {payload_com[1]:.3f}, {payload_com[2]:.3f}] m"
    )
    _put_line(out, subtitle, (32, 80), (190, 204, 220), scale=0.62, thick=1)

    x_target = 22
    x_before = panel_w + 22
    x_after = 2 * panel_w + 22
    y0 = header_h + 34
    _put_line(out, "Target pose", (x_target, y0), (255, 255, 255), scale=0.78, thick=2)
    _put_line(out, "Without compensation", (x_before, y0), (75, 92, 235), scale=0.78, thick=2)
    _put_line(out, "With compensation", (x_after, y0), (46, 181, 125), scale=0.78, thick=2)

    _put_line(out, f"EE pos err: {case.ee_before_mm:.1f} mm", (x_before, y0 + 36), (54, 83, 227), scale=0.62, thick=2)
    _put_line(out, f"EE ori err: {case.ee_ori_before_deg:.1f} deg", (x_before, y0 + 66), (54, 83, 227), scale=0.62, thick=2)
    _put_line(out, f"Joint RMS: {case.joint_rms_before_deg:.2f} deg", (x_before, y0 + 96), (54, 83, 227), scale=0.62, thick=2)

    _put_line(out, f"EE pos err: {case.ee_after_mm:.2f} mm", (x_after, y0 + 36), (28, 158, 106), scale=0.62, thick=2)
    _put_line(out, f"EE ori err: {case.ee_ori_after_deg:.4f} deg", (x_after, y0 + 66), (28, 158, 106), scale=0.62, thick=2)
    _put_line(out, f"Joint RMS: {case.joint_rms_after_deg:.4f} deg", (x_after, y0 + 96), (28, 158, 106), scale=0.62, thick=2)

    footer_y = header_h + panel_h + 34
    improve = 100.0 * (case.ee_before_mm - case.ee_after_mm) / max(case.ee_before_mm, 1e-9)
    footer = (
        f"Static benchmark replay | ee error improvement: {improve:.2f}% | "
        f"residual tau RMS: {case.residual_tau_before_nm:.2f} -> {case.residual_tau_after_nm:.4f} Nm"
    )
    _put_line(out, footer, (24, footer_y), (36, 44, 57), scale=0.68, thick=2)
    return out


def render_video(
    renderer: MujocoRenderer,
    cases: List[CaseData],
    out_path: str,
    side: str,
    payload_mass: float,
    payload_com: np.ndarray,
    fps: int,
    neutral_q: np.ndarray,
) -> str:
    panel_w = renderer.width
    panel_h = renderer.height
    total_w = panel_w * 3
    total_h = panel_h + 176
    writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (total_w, total_h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for: {out_path}")

    try:
        for case_idx, case in enumerate(cases):
            neutral_hold = int(round(0.45 * fps))
            ramp = int(round(1.30 * fps))
            hold = int(round(1.10 * fps))
            pause = int(round(0.45 * fps))
            total_frames = neutral_hold + ramp + hold + pause

            for frame_idx in range(total_frames):
                if frame_idx < neutral_hold:
                    alpha = 0.0
                elif frame_idx < neutral_hold + ramp:
                    t = (frame_idx - neutral_hold) / max(ramp - 1, 1)
                    alpha = 0.5 - 0.5 * math.cos(math.pi * t)
                else:
                    alpha = 1.0

                q_target = (1.0 - alpha) * neutral_q + alpha * case.q_nominal
                q_before = (1.0 - alpha) * neutral_q + alpha * case.q_before
                q_after = (1.0 - alpha) * neutral_q + alpha * case.q_after

                target_img = renderer.render_pose(q_target)
                before_img = renderer.render_pose(q_before)
                after_img = renderer.render_pose(q_after)
                frame = compose_frame(
                    target_img=target_img,
                    before_img=before_img,
                    after_img=after_img,
                    case_idx=case_idx,
                    total_cases=len(cases),
                    case=case,
                    side=side,
                    payload_mass=payload_mass,
                    payload_com=payload_com,
                )
                writer.write(frame)
    finally:
        writer.release()
    return out_path


def write_case_report(out_dir: str, cases: List[CaseData], cfg: Dict[str, object], video_relpath: str) -> None:
    cases_json = []
    lines = [
        "# MuJoCo Payload Compensation Video Cases",
        "",
        f"- side: `{cfg['side']}`",
        f"- payload_mass: `{cfg['payload_mass']}` kg",
        f"- payload_com_ee: `{cfg['payload_com']}` m",
        f"- payload_scale: `{cfg['payload_scale']}`",
        f"- arm_tau_limit: `{cfg['arm_tau_limit']}`",
        f"- seed: `{cfg['seed']}`",
        f"- num_samples: `{cfg['num_samples']}`",
        f"- top_k: `{cfg['top_k']}`",
        f"- video: `{video_relpath}`",
        "",
        "| Case | EE Pos Before (mm) | EE Pos After (mm) | EE Ori Before (deg) | EE Ori After (deg) | Joint RMS Before (deg) | Joint RMS After (deg) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rank, case in enumerate(cases, start=1):
        lines.append(
            f"| {rank} | {case.ee_before_mm:.3f} | {case.ee_after_mm:.3f} | "
            f"{case.ee_ori_before_deg:.3f} | {case.ee_ori_after_deg:.6f} | "
            f"{case.joint_rms_before_deg:.3f} | {case.joint_rms_after_deg:.6f} |"
        )
        cases_json.append(
            {
                "rank": rank,
                "sample_index": case.index,
                "ee_pos_before_mm": case.ee_before_mm,
                "ee_pos_after_mm": case.ee_after_mm,
                "ee_ori_before_deg": case.ee_ori_before_deg,
                "ee_ori_after_deg": case.ee_ori_after_deg,
                "joint_rms_before_deg": case.joint_rms_before_deg,
                "joint_rms_after_deg": case.joint_rms_after_deg,
                "residual_tau_before_nm": case.residual_tau_before_nm,
                "residual_tau_after_nm": case.residual_tau_after_nm,
            }
        )

    with open(os.path.join(out_dir, "cases.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    with open(os.path.join(out_dir, "cases.json"), "w", encoding="utf-8") as f:
        json.dump({"config": cfg, "cases": cases_json, "video": video_relpath}, f, ensure_ascii=False, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a MuJoCo comparison video for G1_29 payload compensation.")
    parser.add_argument("--side", choices=["left", "right"], default="right")
    parser.add_argument("--payload-mass", type=float, default=0.8)
    parser.add_argument("--payload-com", nargs=3, type=float, default=[0.02, 0.0, 0.08], metavar=("X", "Y", "Z"))
    parser.add_argument("--payload-scale", type=float, default=1.0)
    parser.add_argument("--arm-tau-limit", type=float, default=None)
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--range-scale", type=float, default=0.55)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--panel-width", type=int, default=480)
    parser.add_argument("--panel-height", type=int, default=360)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    benchmark = G129StaticBenchmark(REPO_ROOT)
    payload_com = np.asarray(args.payload_com, dtype=float)
    cases = select_cases(
        benchmark=benchmark,
        side=args.side,
        payload_mass=args.payload_mass,
        payload_com=payload_com,
        payload_scale=args.payload_scale,
        arm_tau_limit=args.arm_tau_limit,
        num_samples=args.num_samples,
        seed=args.seed,
        range_scale=args.range_scale,
        top_k=args.top_k,
    )

    xml_path = os.path.join(REPO_ROOT, "assets", "g1", "g1_body29_hand14.xml")
    renderer = MujocoRenderer(xml_path=xml_path, side=args.side, width=args.panel_width, height=args.panel_height)
    neutral_q = np.zeros(14, dtype=float)
    out_path = os.path.join(args.output_dir, "payload_compensation_mujoco.mp4")
    try:
        render_video(
            renderer=renderer,
            cases=cases,
            out_path=out_path,
            side=args.side,
            payload_mass=args.payload_mass,
            payload_com=payload_com,
            fps=args.fps,
            neutral_q=neutral_q,
        )
    finally:
        renderer.close()

    cfg = {
        "side": args.side,
        "payload_mass": args.payload_mass,
        "payload_com": [float(x) for x in payload_com.tolist()],
        "payload_scale": args.payload_scale,
        "arm_tau_limit": args.arm_tau_limit,
        "seed": args.seed,
        "num_samples": args.num_samples,
        "top_k": args.top_k,
    }
    write_case_report(args.output_dir, cases, cfg=cfg, video_relpath=os.path.basename(out_path))
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
