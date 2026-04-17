import os
import sys
import time
import pickle
from typing import Dict, Tuple

import numpy as np
import pinocchio as pin

import logging_mp
logger_mp = logging_mp.getLogger(__name__)

parent2_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parent2_dir)


# ==================== XR_TELEOPERATE PATCH START ====================
# Relative to upstream: this file is the added G1_29 payload estimator module.
# It owns mass_hat estimation plus the structured debug snapshot interface.
# ===================== XR_TELEOPERATE PATCH END =====================
class PayloadEstimatorG1_29:
    """
    Minimal payload mass estimator for G1_29.

    Design scope:
    - Only supports G1_29.
    - Only estimates a single scalar `mass_hat`.
    - Uses static / quasi-static approximation:
        tau_nominal = rnea(q, v=0, a=0)
    - Uses fixed payload_com_ee.
    - Uses active side only:
        left  -> q[0:7]
        right -> q[7:14]

    Path convention:
    - This class intentionally follows the current repo convention and expects to
      be launched from the `teleop/` directory.
    - `cache_path = "g1_29_model_cache.pkl"` is therefore cwd-relative by design.
    """

    LOCKED_JOINTS = [
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

    def __init__(
        self,
        payload_side: str,
        payload_com_ee,
        mass_min: float = 0.0,
        mass_max: float = 3.0,
        dq_threshold: float = 0.05,
        observability_threshold: float = 0.5,
        ema_alpha: float = 0.2,
        hold_last_on_invalid: bool = True,
        debug: bool = False,
        log_every: int = 30,
        unit_test: bool = False,
    ):
        if payload_side not in ("left", "right"):
            raise ValueError(f"payload_side must be 'left' or 'right', got {payload_side}")

        self.payload_com_ee = np.asarray(payload_com_ee, dtype=float).reshape(-1)
        if self.payload_com_ee.size != 3:
            raise ValueError(f"payload_com_ee must have 3 elements, got shape {self.payload_com_ee.shape}")
        if not np.all(np.isfinite(self.payload_com_ee)):
            raise ValueError("payload_com_ee must be finite")

        if mass_min < 0.0:
            raise ValueError(f"mass_min must be >= 0, got {mass_min}")
        if mass_max < mass_min:
            raise ValueError(f"mass_max must be >= mass_min, got mass_min={mass_min}, mass_max={mass_max}")
        if dq_threshold < 0.0:
            raise ValueError(f"dq_threshold must be >= 0, got {dq_threshold}")
        if observability_threshold < 0.0:
            raise ValueError(f"observability_threshold must be >= 0, got {observability_threshold}")
        if not (0.0 <= ema_alpha <= 1.0):
            raise ValueError(f"ema_alpha must be in [0, 1], got {ema_alpha}")

        self.payload_side = payload_side
        self.mass_min = float(mass_min)
        self.mass_max = float(mass_max)
        self.dq_threshold = float(dq_threshold)
        self.observability_threshold = float(observability_threshold)
        self.ema_alpha = float(ema_alpha)
        self.hold_last_on_invalid = bool(hold_last_on_invalid)
        self.debug = bool(debug)
        self.log_every = max(1, int(log_every))
        self.unit_test = bool(unit_test)

        self.gravity = 9.81
        self.model_n = 14

        # Keep the same relative-path convention as the rest of the repo.
        # This estimator is expected to run from `teleop/`.
        self.cache_path = "g1_29_model_cache.pkl"
        if not self.unit_test:
            self.urdf_path = "../assets/g1/g1_body29_hand14.urdf"
            self.model_dir = "../assets/g1/"
            logger_mp.info(
                "[PayloadEstimatorG1_29] path convention: expected launch directory is `teleop/`; "
                f"cache_path is cwd-relative: {self.cache_path}"
            )
            if os.path.basename(os.getcwd()) != "teleop":
                logger_mp.warning(
                    "[PayloadEstimatorG1_29] current working directory is not `teleop/`. "
                    "This class follows the repo's existing relative-path convention."
                )
        else:
            self.urdf_path = "../../assets/g1/g1_body29_hand14.urdf"
            self.model_dir = "../../assets/g1/"

        self.robot = None
        self.reduced_robot = None
        self._build_or_load_model()

        self.model = self.reduced_robot.model
        self.data = self.reduced_robot.data

        if self.model.nq != self.model_n or self.model.nv != self.model_n:
            raise RuntimeError(
                f"G1_29 reduced model dimension mismatch: nq={self.model.nq}, nv={self.model.nv}, expected 14"
            )

        self.L_hand_id = self.model.getFrameId("L_ee")
        self.R_hand_id = self.model.getFrameId("R_ee")

        if self.L_hand_id >= self.model.nframes or self.R_hand_id >= self.model.nframes:
            raise RuntimeError(
                f"Invalid ee frame ids: L_ee={self.L_hand_id}, R_ee={self.R_hand_id}, nframes={self.model.nframes}"
            )

        if self.payload_side == "left":
            self.active_slice = slice(0, 7)
            self.inactive_slice = slice(7, 14)
            self.active_frame_id = self.L_hand_id
        else:
            self.active_slice = slice(7, 14)
            self.inactive_slice = slice(0, 7)
            self.active_frame_id = self.R_hand_id

        self.reset()

        logger_mp.info(
            "[PayloadEstimatorG1_29] init ok: "
            f"payload_side={self.payload_side}, "
            f"active_slice={self.active_slice}, "
            f"active_frame_id={self.active_frame_id}, "
            f"L_ee={self.L_hand_id}, "
            f"R_ee={self.R_hand_id}, "
            f"com_ee={self.payload_com_ee.tolist()}, "
            f"mass_range=[{self.mass_min}, {self.mass_max}]"
        )

    def _build_or_load_model(self):
        if os.path.exists(self.cache_path):
            logger_mp.info(f"[PayloadEstimatorG1_29] loading cache: {self.cache_path}")
            self.robot, self.reduced_robot = self._load_cache()

            # Explicit design:
            # Even if the cache is old and does not contain L_ee / R_ee, we patch
            # the frames in memory and continue. Manual cache deletion is NOT required.
            patched = self._ensure_ee_frames(self.reduced_robot.model)
            if patched:
                logger_mp.warning(
                    "[PayloadEstimatorG1_29] loaded cache was missing L_ee / R_ee. "
                    "Patched frames in memory and refreshing cache file. "
                    "Manual cache deletion is not required."
                )
                self._save_cache()
        else:
            logger_mp.info("[PayloadEstimatorG1_29] loading URDF (slow)...")
            self.robot = pin.RobotWrapper.BuildFromURDF(self.urdf_path, self.model_dir)
            self.reduced_robot = self.robot.buildReducedRobot(
                list_of_joints_to_lock=self.LOCKED_JOINTS,
                reference_configuration=np.array([0.0] * self.robot.model.nq),
            )
            self._ensure_ee_frames(self.reduced_robot.model)
            self._save_cache()

        # Recreate data AFTER ensuring frames so old caches stay safe and frame
        # placement buffers always match the final model.
        self.robot.data = self.robot.model.createData()
        self.reduced_robot.data = self.reduced_robot.model.createData()

    def _ensure_ee_frames(self, model) -> bool:
        frame_names = [frame.name for frame in model.frames]
        patched = False

        if "L_ee" not in frame_names:
            model.addFrame(
                pin.Frame(
                    "L_ee",
                    model.getJointId("left_wrist_yaw_joint"),
                    pin.SE3(np.eye(3), np.array([0.05, 0.0, 0.0]).T),
                    pin.FrameType.OP_FRAME,
                )
            )
            patched = True

        if "R_ee" not in frame_names:
            model.addFrame(
                pin.Frame(
                    "R_ee",
                    model.getJointId("right_wrist_yaw_joint"),
                    pin.SE3(np.eye(3), np.array([0.05, 0.0, 0.0]).T),
                    pin.FrameType.OP_FRAME,
                )
            )
            patched = True

        return patched

    def _save_cache(self):
        data = {
            "robot_model": self.robot.model,
            "reduced_model": self.reduced_robot.model,
        }
        with open(self.cache_path, "wb") as f:
            pickle.dump(data, f)

    def _load_cache(self):
        with open(self.cache_path, "rb") as f:
            data = pickle.load(f)

        robot = pin.RobotWrapper()
        robot.model = data["robot_model"]

        reduced_robot = pin.RobotWrapper()
        reduced_robot.model = data["reduced_model"]

        return robot, reduced_robot

    def _validate_arm_obs(self, arm_obs: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not isinstance(arm_obs, dict):
            raise TypeError(f"arm_obs must be a dict, got {type(arm_obs).__name__}")

        required_keys = ("q", "dq", "tau_est")
        missing = [k for k in required_keys if k not in arm_obs]
        if missing:
            raise KeyError(f"arm_obs missing required keys: {missing}")

        if arm_obs["tau_est"] is None:
            raise ValueError("arm_obs['tau_est'] must not be None")

        q = np.asarray(arm_obs["q"], dtype=float).reshape(-1)
        dq = np.asarray(arm_obs["dq"], dtype=float).reshape(-1)
        tau_est = np.asarray(arm_obs["tau_est"], dtype=float).reshape(-1)

        if q.size != self.model_n:
            raise ValueError(f"arm_obs['q'] must have 14 elements, got {q.size}")
        if dq.size != self.model_n:
            raise ValueError(f"arm_obs['dq'] must have 14 elements, got {dq.size}")
        if tau_est.size != self.model_n:
            raise ValueError(f"arm_obs['tau_est'] must have 14 elements, got {tau_est.size}")

        return q, dq, tau_est

    def _compute_tau_nominal(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=float).reshape(self.model_n)
        v_zero = np.zeros(self.model.nv)
        a_zero = np.zeros(self.model.nv)
        tau_nominal = pin.rnea(self.model, self.data, q, v_zero, a_zero)
        return np.asarray(tau_nominal, dtype=float).reshape(self.model_n)

    def _compute_tau_unit_mass(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=float).reshape(self.model_n)

        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

        J = pin.computeFrameJacobian(
            self.model,
            self.data,
            q,
            self.active_frame_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )

        ee_rotation = self.data.oMf[self.active_frame_id].rotation
        r_world = ee_rotation @ self.payload_com_ee

        force_world_unit = np.array([0.0, 0.0, -self.gravity])
        moment_world_unit = np.cross(r_world, force_world_unit)

        # Pinocchio spatial wrench ordering: [mx, my, mz, fx, fy, fz]
        wrench_unit = np.hstack([moment_world_unit, force_world_unit])
        tau_unit_full = J.T @ wrench_unit
        return np.asarray(tau_unit_full, dtype=float).reshape(self.model_n)

    def _safe_max_abs(self, x) -> float:
        if x is None:
            return float("nan")
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.size == 0 or not np.any(np.isfinite(x)):
            return float("nan")
        return float(np.nanmax(np.abs(x)))

    # ===== XR_TELEOPERATE PATCH: estimator debug-state cache =====
    def _update_last_debug_state(
        self,
        mode: str,
        reason: str,
        dq_active=None,
        tau_nominal_active=None,
        tau_est_active=None,
        tau_residual=None,
        tau_unit_active=None,
        mass_raw=None,
    ) -> None:
        self.last_mode = str(mode)
        self.last_reason = str(reason)

        if dq_active is not None:
            self.last_dq_active = np.asarray(dq_active, dtype=float).reshape(-1).copy()
        if tau_nominal_active is not None:
            self.last_tau_nominal = np.asarray(tau_nominal_active, dtype=float).reshape(-1).copy()
        if tau_est_active is not None:
            self.last_tau_est = np.asarray(tau_est_active, dtype=float).reshape(-1).copy()
        if tau_residual is not None:
            self.last_tau_residual = np.asarray(tau_residual, dtype=float).reshape(-1).copy()
        if tau_unit_active is not None:
            self.last_tau_unit_mass = np.asarray(tau_unit_active, dtype=float).reshape(-1).copy()
        if mass_raw is not None:
            self.last_raw_mass = float(mass_raw)

    def _maybe_log_debug(
        self,
        mode: str,
        reason: str,
        dq_active=None,
        tau_nominal_active=None,
        tau_est_active=None,
        tau_residual=None,
        tau_unit_active=None,
        mass_raw=None,
    ):
        if not self.debug:
            return
        if self.update_count % self.log_every != 0:
            return

        logger_mp.info(
            "[payload_est] "
            f"mode={mode} "
            f"reason={reason} "
            f"side={self.payload_side} "
            f"dq_max={self._safe_max_abs(dq_active):.5f} "
            f"tau_nom_max={self._safe_max_abs(tau_nominal_active):.5f} "
            f"tau_est_max={self._safe_max_abs(tau_est_active):.5f} "
            f"tau_res_max={self._safe_max_abs(tau_residual):.5f} "
            f"tau_unit_max={self._safe_max_abs(tau_unit_active):.5f} "
            f"mass_raw={float(mass_raw) if mass_raw is not None else float('nan'):.5f} "
            f"mass_hat={float(self.mass_hat):.5f} "
            f"is_valid={self.is_valid}"
        )

    def _hold_or_fallback(self, reason: str) -> float:
        self.last_mode = "hold"
        self.last_reason = str(reason)
        if self.hold_last_on_invalid and self.is_valid:
            self._maybe_log_debug(mode="hold", reason=reason)
            return float(self.mass_hat)

        self.mass_hat = 0.0
        self.is_valid = False
        self._maybe_log_debug(mode="hold", reason=reason)
        return float(self.mass_hat)

    def update(self, arm_obs: Dict) -> float:
        self.update_count += 1

        try:
            q, dq, tau_est = self._validate_arm_obs(arm_obs)
        except (TypeError, KeyError, ValueError):
            return self._hold_or_fallback("invalid_arm_obs")

        dq_active = dq[self.active_slice]
        tau_est_active = tau_est[self.active_slice]

        if not np.all(np.isfinite(q)):
            return self._hold_or_fallback("nonfinite_q")
        if not np.all(np.isfinite(dq)):
            self._update_last_debug_state(mode="hold", reason="nonfinite_dq", dq_active=dq_active)
            return self._hold_or_fallback("nonfinite_dq")
        if not np.all(np.isfinite(tau_est_active)):
            self._update_last_debug_state(mode="hold", reason="nonfinite_tau_est", dq_active=dq_active, tau_est_active=tau_est_active)
            return self._hold_or_fallback("nonfinite_tau_est")

        if np.max(np.abs(dq_active)) > self.dq_threshold:
            self._update_last_debug_state(mode="hold", reason="dq_gate", dq_active=dq_active, tau_est_active=tau_est_active)
            return self._hold_or_fallback("dq_gate")

        tau_nominal_full = self._compute_tau_nominal(q)
        tau_unit_full = self._compute_tau_unit_mass(q)

        tau_nominal_active = tau_nominal_full[self.active_slice]
        tau_unit_active = tau_unit_full[self.active_slice]
        tau_residual = tau_est_active - tau_nominal_active

        # Observability gate:
        # use the scalar tau_unit_active.T @ tau_unit_active directly.
        denom = float(tau_unit_active.T @ tau_unit_active)
        if denom < self.observability_threshold:
            self._update_last_debug_state(
                mode="hold",
                reason="low_observability",
                dq_active=dq_active,
                tau_nominal_active=tau_nominal_active,
                tau_est_active=tau_est_active,
                tau_residual=tau_residual,
                tau_unit_active=tau_unit_active,
            )
            return self._hold_or_fallback("low_observability")

        # Scalar projection:
        # m_raw = (tau_unit^T tau_residual) / (tau_unit^T tau_unit)
        numer = float(tau_unit_active.T @ tau_residual)
        mass_raw = numer / denom
        mass_clipped = float(np.clip(mass_raw, self.mass_min, self.mass_max))

        if self.is_valid:
            self.mass_hat = self.ema_alpha * mass_clipped + (1.0 - self.ema_alpha) * self.mass_hat
        else:
            self.mass_hat = mass_clipped

        self.last_valid_mass_hat = float(self.mass_hat)
        self.is_valid = True
        self.last_update_time = time.time()

        self._update_last_debug_state(
            mode="update",
            reason="ok",
            dq_active=dq_active,
            tau_nominal_active=tau_nominal_active,
            tau_est_active=tau_est_active,
            tau_residual=tau_residual,
            tau_unit_active=tau_unit_active,
            mass_raw=mass_raw,
        )

        self._maybe_log_debug(
            mode="update",
            reason="ok",
            dq_active=dq_active,
            tau_nominal_active=tau_nominal_active,
            tau_est_active=tau_est_active,
            tau_residual=tau_residual,
            tau_unit_active=tau_unit_active,
            mass_raw=mass_raw,
        )

        return float(self.mass_hat)

    def get_mass_hat(self) -> float:
        return float(self.mass_hat)

    # ===== XR_TELEOPERATE PATCH: structured estimator snapshot =====
    def get_debug_snapshot(self) -> Dict:
        return {
            "payload_side": self.payload_side,
            "mass_hat": float(self.mass_hat),
            "last_valid_mass_hat": float(self.last_valid_mass_hat),
            "is_valid": bool(self.is_valid),
            "update_count": int(self.update_count),
            "last_update_time": self.last_update_time,
            "mode": self.last_mode,
            "reason": self.last_reason,
            "mass_raw": float(self.last_raw_mass),
            "dq_active_max": self._safe_max_abs(self.last_dq_active),
            "tau_nominal_max": self._safe_max_abs(self.last_tau_nominal),
            "tau_est_max": self._safe_max_abs(self.last_tau_est),
            "tau_residual_max": self._safe_max_abs(self.last_tau_residual),
            "tau_unit_max": self._safe_max_abs(self.last_tau_unit_mass),
        }

    def reset(self) -> None:
        self.mass_hat = 0.0
        self.last_valid_mass_hat = 0.0
        self.is_valid = False
        self.last_update_time = None
        self.update_count = 0

        self.last_mode = "hold"
        self.last_reason = "reset"
        self.last_raw_mass = 0.0
        self.last_dq_active = np.zeros(7)
        self.last_tau_nominal = np.zeros(7)
        self.last_tau_est = np.zeros(7)
        self.last_tau_residual = np.zeros(7)
        self.last_tau_unit_mass = np.zeros(7)


if __name__ == "__main__":
    print("PayloadEstimatorG1_29 smoke test")
    print("This is only an interface / initialization smoke test, not an accuracy validation.")
    print("Under mock zero input, mass_hat may stay at 0.0 or another small bounded value.")

    estimator = PayloadEstimatorG1_29(
        payload_side="right",
        payload_com_ee=[0.02, 0.0, 0.08],
        debug=True,
        log_every=1,
    )

    print("init ok, mass_hat =", estimator.get_mass_hat())

    estimator.reset()
    print("reset ok, mass_hat =", estimator.get_mass_hat())

    # This mock input is only for smoke testing the interface and control flow.
    # It is NOT meant to validate estimation quality.
    arm_obs = {
        "q": np.zeros(14),
        "dq": np.zeros(14),
        "tau_est": np.zeros(14),
    }

    mass_hat = estimator.update(arm_obs)
    print("update ok, mass_hat =", mass_hat)
    print("A small non-zero value here can still be expected for a smoke test.")
