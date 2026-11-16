"""
6-DOF Robot Arm Kinematics Engine
===================================
Target  : ESP32 over WiFi (Option 2 control architecture)
Method  : Denavit-Hartenberg (DH) convention
IK      : Numerical — Jacobian pseudo-inverse with damped least squares

─────────────────────────────────────────────────────────────────────────────
DH PARAMETER GUIDE
─────────────────────────────────────────────────────────────────────────────
Each joint i is described by 4 parameters:

  a_i     : Link length      — distance along x_i from z_i to z_{i+1}
  alpha_i : Link twist       — angle from z_i to z_{i+1} about x_i
  d_i     : Link offset      — distance along z_i from x_{i-1} to x_i
  theta_i : Joint angle      — angle from x_{i-1} to x_i about z_i (VARIABLE)

Standard anthropomorphic 6R layout (e.g. scaled-down UR3):

        z1 (up)
        |
       [J1] — base rotation
        |
       [J2] — shoulder pitch        a2 is your upper arm length
        |
       [J3] — elbow pitch           a3 is your forearm length
        |
       [J4] — wrist 1 (axis PARALLEL to J2/J3, i.e. a pitch — UR-style)
        |
       [J5] — wrist 2 (axis perpendicular to J4)
        |
       [J6] — wrist 3 (spin about the tool axis)

NOTE: with all joints at 0 the arm points straight up and the tool points
along -Y. That home pose is triple-singular (straight arm, J4 || J6, Euler
gimbal lock) — never use it as an IK seed. Make sure your physical zero and
ESP32 firmware zero match this convention (J2 theta_off = +pi/2).

─────────────────────────────────────────────────────────────────────────────
HOW TO UPDATE LINK LENGTHS (once CAD is finalised)
─────────────────────────────────────────────────────────────────────────────
Edit the DH_PARAMS table in RobotConfig. Units are in METRES.

  d1      → height from ground to shoulder pivot
  a2      → upper arm length  (shoulder to elbow)
  a3      → forearm length    (elbow to wrist)
  d4      → wrist offset      (along arm axis into wrist assembly)
  d6      → tool/end-effector offset from wrist

Everything else is computed automatically.
─────────────────────────────────────────────────────────────────────────────
"""

import sys
import socket
import time
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401 — registers 3D projection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from dataclasses import dataclass, field
from typing import Optional, List

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("RobotArm")

# Avoid UnicodeEncodeError on consoles that are not UTF-8 (e.g. some Windows terminals)
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# ROBOT CONFIGURATION  ← Edit this section when CAD is finalised
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class RobotConfig:
    """
    All physical constants for the robot arm.

    DH table layout: each row is [a, alpha, d, theta_offset]
      - a           (metres)   link length
      - alpha       (radians)  link twist
      - d           (metres)   link offset
      - theta_off   (radians)  constant offset added to joint angle
                               (used to zero the home position)

    PLACEHOLDER VALUES — replace with your measured link lengths.
    These approximate a compact desktop 6R arm at ~700g total mass.
    """

    # ── Link lengths (CHANGE THESE) ───────────────────────────────────────────
    d1: float = 0.100   # base-to-shoulder height          (m)
    a2: float = 0.120   # upper arm length shoulder→elbow   (m)
    a3: float = 0.100   # forearm length elbow→wrist        (m)
    d4: float = 0.00   # wrist offset along arm axis       (m)
    d6: float = 0.050   # end-effector / tool offset        (m)

    # ── Joint limits  [min, max]  in degrees ─────────────────────────────────
    joint_limits_deg: list = field(default_factory=lambda: [
        [-180,  180],   # J1 base
        [ -90,   90],   # J2 shoulder
        [-135,   45],   # J3 elbow
        [-180,  180],   # J4 wrist roll
        [ -90,   90],   # J5 wrist pitch
        [-180,  180],   # J6 wrist roll
    ])

    # ── IK solver settings ────────────────────────────────────────────────────
    ik_max_iterations:  int   = 500      # max Jacobian steps
    ik_tolerance_pos:   float = 1e-4     # position tolerance  (metres)
    ik_tolerance_ori:   float = 1e-3     # orientation tolerance (radians)
    ik_step_size:       float = 0.5      # step scaling factor  (0 < α ≤ 1)
    ik_damping:         float = 1e-3     # damped least-squares λ
    ik_restarts:        int   = 10       # random-seed retries if the first solve fails
    ik_singular_threshold: float = 0.01  # σ_min below this → extra damping (Wampler)
    ik_singular_damping:   float = 0.05  # max extra λ applied right at a singularity

    # ── WiFi / ESP32 settings ─────────────────────────────────────────────────
    esp32_ip:   str = "192.168.1.100"   # ← set your ESP32's IP here
    esp32_port: int = 5000
    esp32_timeout: float = 5.0          # socket timeout in seconds

    @property
    def dh_params(self) -> np.ndarray:
        """
        Returns the 6×4 DH table as a numpy array.
        Columns: [a, alpha, d, theta_offset]
        """
        π = np.pi
        return np.array([
            # a         alpha       d         theta_off
            [0.0,       π/2,        self.d1,  0.0   ],  # J1 base
            # theta_off = +π/2 → DH θ2 is the upper-arm ELEVATION from horizontal,
            # so user J2 = 0 means the upper arm points straight up.
            [self.a2,    0.0,        0.0,      π/2  ],  # J2 shoulder
        
 #           [self.a3,   -π/2,        0.0,      0.0   ],  # J3 elbow
#            [0.0,       -π/2,        self.d4,  0.0   ],  # J4 wrist roll
 #           [0.0,        π/2,        0.0,      0.0   ],  # J5 wrist pitch
#            [0.0,        0.0,        self.d6,  0.0   ],  # J6 wrist roll

            # ── Alternative (roll-pitch-roll) rows, currently disabled ──

            [self.a3,  -π/2, 0.0,      -π/2],  # J3 elbow   (a3 = 0, offset -π/2 so J3 = 0 is a straight arm)
            [0.0,  π/2, self.d4,  0.0 ],  # J4 forearm roll (d4 = forearm length)
            [0.0,   π/2, 0.0,      0.0 ],  # J5 wrist pitch
            [0.0,   0.0, self.d6,  0.0 ],  # J6 tool roll
        
        ])

    @property
    def joint_limits_rad(self) -> np.ndarray:
        """Joint limits converted to radians, shape (6, 2)."""
        return np.radians(self.joint_limits_deg)


# ─────────────────────────────────────────────────────────────────────────────
# DH TRANSFORM  (single joint)
# ─────────────────────────────────────────────────────────────────────────────
def dh_transform(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
    """
    Compute the 4×4 homogeneous DH transform for a single joint.

    T = Rot_z(θ) · Trans_z(d) · Trans_x(a) · Rot_x(α)

    Parameters
    ----------
    a     : link length       (metres)
    alpha : link twist        (radians)
    d     : link offset       (metres)
    theta : joint angle       (radians)

    Returns
    -------
    T : 4×4 numpy array — homogeneous transform from frame i-1 to frame i
    """
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)

    return np.array([
        [ct,   -st*ca,   st*sa,   a*ct],
        [st,    ct*ca,  -ct*sa,   a*st],
        [0.0,      sa,      ca,      d],
        [0.0,     0.0,     0.0,    1.0],
    ])


# ─────────────────────────────────────────────────────────────────────────────
# FORWARD KINEMATICS
# ─────────────────────────────────────────────────────────────────────────────
class ForwardKinematics:
    """
    Computes end-effector pose from joint angles using DH convention.
    """

    def __init__(self, config: RobotConfig):
        self.config = config

    def compute(
        self,
        joint_angles_rad: np.ndarray,
        return_all_frames: bool = False
    ) -> dict:
        """
        Compute forward kinematics.

        Parameters
        ----------
        joint_angles_rad  : array of 6 joint angles in radians
        return_all_frames : if True, returns transforms for all joints
                            (useful for visualisation / debugging)

        Returns
        -------
        dict with keys:
          'T'          : 4×4 end-effector transform (base → EE)
          'position'   : (x, y, z) in metres
          'rotation'   : 3×3 rotation matrix
          'euler_xyz'  : (roll, pitch, yaw) in degrees
          'frames'     : list of intermediate 4×4 transforms (if requested)
        """
        dh = self.config.dh_params
        T = np.eye(4)
        frames = [T.copy()]

        for i in range(6):
            a, alpha, d, theta_off = dh[i]
            theta = joint_angles_rad[i] + theta_off
            T = T @ dh_transform(a, alpha, d, theta)
            if return_all_frames:
                frames.append(T.copy())

        position = T[:3, 3]
        rotation = T[:3, :3]
        euler    = self._rotation_to_euler_xyz(rotation)

        result = {
            "T":        T,
            "position": position,
            "rotation": rotation,
            "euler_xyz": np.degrees(euler),   # convert to degrees for readability
        }
        if return_all_frames:
            result["frames"] = frames

        return result

    @staticmethod
    def _rotation_to_euler_xyz(R: np.ndarray) -> np.ndarray:
        """
        Extract roll-pitch-yaw (XYZ Euler) angles from a rotation matrix.
        Handles gimbal lock (pitch = ±90°) gracefully.
        """
        sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
        singular = sy < 1e-6

        if not singular:
            roll  = np.arctan2( R[2, 1], R[2, 2])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw   = np.arctan2( R[1, 0], R[0, 0])
        else:
            # Gimbal lock: yaw absorbed into roll
            roll  = np.arctan2(-R[1, 2], R[1, 1])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw   = 0.0

        return np.array([roll, pitch, yaw])


# ─────────────────────────────────────────────────────────────────────────────
# INVERSE KINEMATICS  (Numerical — Damped Jacobian Pseudo-inverse)
# ─────────────────────────────────────────────────────────────────────────────
class InverseKinematics:
    """
    Solves IK numerically using the Jacobian pseudo-inverse method
    with damped least squares (Levenberg-Marquardt style).

    Why damped least squares?
    - Near singularities the plain pseudo-inverse blows up.
    - Adding λI to J·Jᵀ before inversion keeps joint velocities bounded.
    - λ (ik_damping in config) controls the trade-off: larger λ = safer
      near singularities but slower convergence.
    """

    def __init__(self, config: RobotConfig, fk: ForwardKinematics):
        self.config = config
        self.fk     = fk

    def solve(
        self,
        target_position: np.ndarray,
        target_rotation: Optional[np.ndarray] = None,
        initial_angles:  Optional[np.ndarray] = None,
    ) -> dict:
        """
        Solve IK from the given seed; if that fails, retry from up to
        cfg.ik_restarts random in-limit seeds and return the first success
        (or the lowest-error attempt). Deterministic: the RNG is seeded.
        """
        best = self._solve_once(target_position, target_rotation, initial_angles)
        if best["success"]:
            return best
        lim = self.config.joint_limits_rad
        rng = np.random.default_rng(0)
        for _ in range(self.config.ik_restarts):
            seed = rng.uniform(lim[:, 0], lim[:, 1])
            res  = self._solve_once(target_position, target_rotation, seed)
            if res["success"]:
                res["message"] += " (random restart)"
                return res
            if res["position_error"] < best["position_error"]:
                best = res
        return best

    def _solve_once(
        self,
        target_position: np.ndarray,
        target_rotation: Optional[np.ndarray] = None,
        initial_angles:  Optional[np.ndarray] = None,
    ) -> dict:
        """
        Solve IK for a target pose.

        Parameters
        ----------
        target_position : (x, y, z) in metres — required
        target_rotation : 3×3 rotation matrix  — optional
                          If None, only position is controlled (3-DOF task).
        initial_angles  : starting guess in radians (6,)
                          If None, uses the home position (all zeros).

        Returns
        -------
        dict with keys:
          'success'        : bool
          'joint_angles'   : solution in radians (6,)
          'joint_angles_deg': solution in degrees (6,)
          'position_error' : final Euclidean position error (metres)
          'iterations'     : number of iterations taken
          'message'        : human-readable status string
        """
        cfg = self.config

        # ── Initial guess ─────────────────────────────────────────────────────
        q = (
            np.array(initial_angles, dtype=float)
            if initial_angles is not None
            else np.zeros(6)
        )

        position_only = target_rotation is None

        for iteration in range(cfg.ik_max_iterations):

            # ── Current FK ────────────────────────────────────────────────────
            fk_result = self.fk.compute(q)
            current_pos = fk_result["position"]
            current_rot = fk_result["rotation"]

            # ── Position error ────────────────────────────────────────────────
            pos_error = target_position - current_pos

            # ── Orientation error (axis-angle representation) ─────────────────
            if not position_only:
                R_error   = target_rotation @ current_rot.T
                ori_error = self._rotation_error_vector(R_error)
                error_vec = np.concatenate([pos_error, ori_error])   # 6D
            else:
                error_vec = pos_error                                 # 3D

            # ── Convergence check ─────────────────────────────────────────────
            pos_err_norm = np.linalg.norm(pos_error)
            ori_err_norm = (
                np.linalg.norm(ori_error) if not position_only else 0.0
            )

            if (pos_err_norm < cfg.ik_tolerance_pos and
                    ori_err_norm < cfg.ik_tolerance_ori):
                return self._result(
                    success=True,
                    q=q,
                    pos_error=pos_err_norm,
                    iterations=iteration,
                    message=f"Converged in {iteration} iterations. "
                            f"Pos error: {pos_err_norm*1000:.3f} mm"
                )

            # ── Jacobian ──────────────────────────────────────────────────────
            J = self._compute_jacobian(q, position_only)

            # ── Damped least-squares step: Δq = Jᵀ(JJᵀ + λ²I)⁻¹ · e ─────────
            λ2  = cfg.ik_damping ** 2
            sigma_min = np.linalg.svd(J, compute_uv=False)[-1]
            if sigma_min < cfg.ik_singular_threshold:
                # Wampler adaptive damping: ramps up smoothly toward the singularity
                λ2 += cfg.ik_singular_damping ** 2 * (
                    1.0 - (sigma_min / cfg.ik_singular_threshold) ** 2
                )
            JJt = J @ J.T
            dq  = cfg.ik_step_size * (
                J.T @ np.linalg.solve(JJt + λ2 * np.eye(JJt.shape[0]), error_vec)
            )

            # ── Update and clamp to joint limits ──────────────────────────────
            q = self._clamp_joints(q + dq)

        # ── Did not converge ──────────────────────────────────────────────────
        final_fk  = self.fk.compute(q)
        final_err = np.linalg.norm(target_position - final_fk["position"])

        return self._result(
            success=False,
            q=q,
            pos_error=final_err,
            iterations=cfg.ik_max_iterations,
            message=f"Did not converge. Final pos error: {final_err*1000:.3f} mm. "
                    "Try a different initial guess or check target reachability."
        )

    # ── Jacobian (numerical, central differences) ─────────────────────────────
    def _compute_jacobian(
        self,
        q: np.ndarray,
        position_only: bool,
        delta: float = 1e-6
    ) -> np.ndarray:
        """
        Compute the Jacobian numerically via central differences.

        Numerical Jacobian is preferred here over an analytical one because:
        - It automatically adapts when you change link lengths in config.
        - No need to re-derive when geometry changes during prototyping.
        - Central differences give O(δ²) accuracy vs O(δ) for forward diffs.

        For a production system you would derive the analytical Jacobian,
        but for a 6R arm at this scale the numerical version is fast enough.
        """
        rows = 3 if position_only else 6
        J    = np.zeros((rows, 6))

        for i in range(6):
            q_plus  = q.copy(); q_plus[i]  += delta
            q_minus = q.copy(); q_minus[i] -= delta

            fk_plus  = self.fk.compute(q_plus)
            fk_minus = self.fk.compute(q_minus)

            # Position Jacobian rows
            J[:3, i] = (fk_plus["position"] - fk_minus["position"]) / (2 * delta)

            # Orientation Jacobian rows (only if 6D task)
            if not position_only:
                R_plus  = fk_plus["rotation"]
                R_minus = fk_minus["rotation"]
                R_diff  = R_plus @ R_minus.T
                J[3:, i] = self._rotation_error_vector(R_diff) / (2 * delta)

        return J

    @staticmethod
    def _rotation_error_vector(R_error: np.ndarray) -> np.ndarray:
        """
        Convert a rotation matrix error into a 3D axis-angle error vector.
        Uses the skew-symmetric part: e = 0.5 * [R32-R23, R13-R31, R21-R12]
        """
        return 0.5 * np.array([
            R_error[2, 1] - R_error[1, 2],
            R_error[0, 2] - R_error[2, 0],
            R_error[1, 0] - R_error[0, 1],
        ])

    def _clamp_joints(self, q: np.ndarray) -> np.ndarray:
        """Clamp joint angles to their configured limits."""
        limits = self.config.joint_limits_rad
        return np.clip(q, limits[:, 0], limits[:, 1])

    @staticmethod
    def _result(
        success: bool,
        q: np.ndarray,
        pos_error: float,
        iterations: int,
        message: str
    ) -> dict:
        return {
            "success":          success,
            "joint_angles":     q,
            "joint_angles_deg": np.degrees(q),
            "position_error":   pos_error,
            "iterations":       iterations,
            "message":          message,
        }


# ─────────────────────────────────────────────────────────────────────────────
# DECOUPLED IK  (Analytical — Positional + Orientational subproblems)
# ─────────────────────────────────────────────────────────────────────────────
class DecoupledIK:
    """
    Analytical inverse kinematics using kinematic decoupling.

      Subproblem 1 — Positional IK  (J1, J2, J3)
        Places the wrist centre (origin of frames 3-6, since d4 = 0) using
        the law of cosines for the planar shoulder/elbow pair.
      Subproblem 2 — Orientational IK  (J4, J5, J6)
        With this DH table the wrist rotation is
            R_36 = Rz(θ4) · Ry(-θ5) · Rz(θ6)
        (note the MINUS on θ5, from alpha4 = +π/2, alpha5 = -π/2), i.e. a ZYZ
        Euler sequence with middle angle b = -θ5.

    Requires d4 = 0 (spherical wrist). All 2 (elbow) x 2 (wrist flip) x 2
    (front/back) branches are generated, wrapped to (-π, π], filtered by the
    joint limits, and the one closest to `current_angles` is returned.
    """

    def __init__(self, config: RobotConfig, fk: ForwardKinematics):
        self.config = config
        self.fk     = fk

    @staticmethod
    def _wrap(q: np.ndarray) -> np.ndarray:
        return (q + np.pi) % (2 * np.pi) - np.pi

    @staticmethod
    def _wrist_solutions(R36: np.ndarray):
        """Return (a, b, c) ZYZ triples with R36 = Rz(a)·Ry(b)·Rz(c); θ5 = -b."""
        sb = np.hypot(R36[0, 2], R36[1, 2])
        if sb < 1e-6:                                   # wrist singularity
            if R36[2, 2] > 0:                           # b ≈ 0 → only a + c known
                return [(np.arctan2(R36[1, 0], R36[0, 0]), 0.0, 0.0)]
            return [(0.0, np.pi, np.arctan2(R36[1, 0], R36[1, 1]))]   # b ≈ π
        b = np.arctan2(sb, R36[2, 2])
        a = np.arctan2(R36[1, 2],  R36[0, 2])
        c = np.arctan2(R36[2, 1], -R36[2, 0])
        return [(a, b, c), (a + np.pi, -b, c + np.pi)]  # wrist "flip" pair

    def solve(
        self,
        target_position: np.ndarray,
        target_rotation: np.ndarray,
        elbow_up: Optional[bool] = None,
        current_angles: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Parameters
        ----------
        target_position : (x, y, z) in metres
        target_rotation : 3×3 rotation matrix (required)
        elbow_up        : True/False forces the elbow above/below the
                          shoulder→wrist chord; None = best branch overall
        current_angles  : joint angles (rad) used to pick the nearest branch
                          (default: zeros)

        Returns the same dict as InverseKinematics.solve plus 'elbow_up',
        'orientation_error' and 'branches_in_limits'.
        """
        cfg = self.config
        dh  = cfg.dh_params
        off = dh[:, 3]
        lim = cfg.joint_limits_rad
        p   = np.asarray(target_position, dtype=float)
        R   = np.asarray(target_rotation,  dtype=float)
        cur = np.zeros(6) if current_angles is None else np.asarray(current_angles, dtype=float)

        # Wrist centre: step back from the tool tip along the tool z-axis
        p_wc = p - cfg.d6 * R[:, 2]
        r    = np.hypot(p_wc[0], p_wc[1])
        s    = p_wc[2] - cfg.d1

        cos_j3      = (r**2 + s**2 - cfg.a2**2 - cfg.a3**2) / (2 * cfg.a2 * cfg.a3)
        unreachable = abs(cos_j3) > 1.0 + 1e-6
        c3          = float(np.clip(cos_j3, -1.0, 1.0))

        # Shoulder singularity (wrist centre on the J1 axis): keep current J1
        base_j1 = np.arctan2(p_wc[1], p_wc[0]) if r > 1e-9 else cur[0] + off[0]

        candidates = []   # (violation, distance_to_current, q_user, elbow_is_up)
        for j1_raw, rr in ((base_j1, r), (base_j1 + np.pi, -r)):   # front / back
            for sgn in (+1.0, -1.0):                                # elbow branch
                j3 = np.arctan2(sgn * np.sqrt(1.0 - c3**2), c3)
                j2 = np.arctan2(s, rr) - np.arctan2(
                    cfg.a3 * np.sin(j3), cfg.a2 + cfg.a3 * np.cos(j3))

                # FK adds theta_off itself → feed it (raw - theta_off)
                q03    = np.array([j1_raw, j2, j3, 0.0, 0.0, 0.0]) - off
                frames = self.fk.compute(q03, return_all_frames=True)["frames"]
                R03    = frames[3][:3, :3]

                # Is the elbow above the shoulder→wrist chord (in the arm plane)?
                u      = np.array([np.cos(j1_raw), np.sin(j1_raw), 0.0])
                e      = frames[2][:3, 3] - frames[1][:3, 3]
                cross  = rr * e[2] - s * float(u @ e)
                is_up  = (cross > 0) if abs(cross) > 1e-9 else (sgn < 0)

                R36 = R03.T @ R
                for a, b, c in self._wrist_solutions(R36):
                    q_raw = np.array([j1_raw, j2, j3, a, -b, c])    # θ5 = -b
                    q     = self._wrap(q_raw - off)
                    viol  = float(np.sum(np.maximum(lim[:, 0] - q, 0) +
                                         np.maximum(q - lim[:, 1], 0)))
                    dist  = float(np.sum(np.abs(self._wrap(q - cur))))
                    candidates.append((viol, dist, q, bool(is_up)))

        pool = candidates
        if elbow_up is not None:
            filtered = [c for c in candidates if c[3] == elbow_up]
            pool = filtered or candidates
        in_limits = [c for c in pool if c[0] < 1e-9]
        best = min(in_limits, key=lambda c: c[1]) if in_limits else min(pool, key=lambda c: (c[0], c[1]))
        _, _, q_best, use_up = best

        q_clamped = np.clip(q_best, lim[:, 0], lim[:, 1])
        fk_check  = self.fk.compute(q_clamped)
        pos_error = float(np.linalg.norm(fk_check["position"] - p))
        ori_error = float(np.linalg.norm(InverseKinematics._rotation_error_vector(
            R @ fk_check["rotation"].T)))

        warnings = []
        if unreachable:
            warnings.append("target out of reach (cos_j3 clamped)")
        if not in_limits:
            warnings.append("no branch satisfies joint limits")
        if abs(np.sin(q_best[4] + off[4])) < 1e-6:
            warnings.append("wrist singularity (J5≈0 or π)")
        warning_str = "  Warnings: " + ", ".join(warnings) if warnings else ""

        success = (not unreachable
                   and pos_error < cfg.ik_tolerance_pos * 10
                   and ori_error < cfg.ik_tolerance_ori * 10)
        message = (f"Decoupled IK — {'elbow-up' if use_up else 'elbow-down'}. "
                   f"Pos error: {pos_error*1000:.3f} mm, ori error: {ori_error:.4f} rad."
                   + warning_str)

        return {
            "success":            success,
            "joint_angles":       q_clamped,
            "joint_angles_deg":   np.degrees(q_clamped),
            "position_error":     pos_error,
            "orientation_error":  ori_error,
            "iterations":         1,
            "message":            message,
            "elbow_up":           use_up,
            "branches_in_limits": len(in_limits),
        }


# ─────────────────────────────────────────────────────────────────────────────
# ESP32 WIFI CONTROLLER
# ─────────────────────────────────────────────────────────────────────────────
class ESP32Controller:
    """
    Sends joint angle commands to the ESP32 over a TCP WiFi socket.

    Command protocol (plain text, newline-terminated):
      ANGLES <j1> <j2> <j3> <j4> <j5> <j6>
      Values are in degrees (easier for the Arduino servo/stepper code).

    ESP32 responds with "OK\n" on success or "ERR <msg>\n" on failure.
    """

    def __init__(self, config: RobotConfig):
        self.config = config
        self.sock   = None

    def connect(self) -> bool:
        """Open TCP connection to the ESP32."""
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(self.config.esp32_timeout)
            self.sock.connect((self.config.esp32_ip, self.config.esp32_port))
            log.info(f"Connected to ESP32 at {self.config.esp32_ip}:{self.config.esp32_port}")
            return True
        except (socket.timeout, ConnectionRefusedError, OSError) as e:
            log.error(f"Could not connect to ESP32: {e}")
            self.sock = None
            return False

    def disconnect(self):
        """Close the TCP connection."""
        if self.sock:
            self.sock.close()
            self.sock = None
            log.info("Disconnected from ESP32.")

    def send_joint_angles(self, angles_deg: np.ndarray) -> bool:
        """
        Send joint angles (degrees) to the ESP32.

        Parameters
        ----------
        angles_deg : array of 6 joint angles in DEGREES

        Returns
        -------
        True if ESP32 acknowledged with "OK", False otherwise.
        """
        if self.sock is None:
            log.error("Not connected. Call connect() first.")
            return False

        # Round to 2 decimal places — enough precision for servo/stepper control
        formatted = " ".join(f"{a:.2f}" for a in angles_deg)
        command   = f"ANGLES {formatted}\n"

        try:
            self.sock.sendall(command.encode("utf-8"))
            buf = b""
            while not buf.endswith(b"\n") and len(buf) < 256:
                chunk = self.sock.recv(64)
                if not chunk:
                    raise OSError("connection closed by ESP32")
                buf += chunk
            response = buf.decode("utf-8").strip()

            if response.startswith("OK"):
                log.info(f"ESP32 ACK | Sent: {command.strip()}")
                return True
            else:
                log.warning(f"ESP32 response: {response}")
                return False

        except socket.timeout:
            log.error("ESP32 response timed out.")
            return False
        except OSError as e:
            log.error(f"Socket error while sending: {e}")
            self.sock = None
            return False

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *_):
        self.disconnect()


# ─────────────────────────────────────────────────────────────────────────────
# ROBOT ARM  (top-level API)
# ─────────────────────────────────────────────────────────────────────────────
class RobotArm:
    """
    High-level API combining FK, IK, and ESP32 communication.

    Typical usage
    -------------
        arm = RobotArm()

        # Move to a target XYZ position
        arm.move_to_position([0.15, 0.05, 0.18])

        # Check where the arm currently is (FK)
        pose = arm.get_current_pose()
        print(pose)

        # Connect and stream a trajectory
        with arm.controller:
            arm.move_to_position([0.10, 0.10, 0.15], send=True)
    """

    def __init__(self, config: Optional[RobotConfig] = None):
        self.config     = config or RobotConfig()
        self.fk         = ForwardKinematics(self.config)
        self.ik         = InverseKinematics(self.config, self.fk)
        self.ik_decoupled = DecoupledIK(self.config, self.fk)
        self.controller = ESP32Controller(self.config)

        # Track current joint state (radians)
        self._joint_angles = np.zeros(6)

    # ── Forward Kinematics ────────────────────────────────────────────────────
    def get_current_pose(self) -> dict:
        """Return FK result for the current joint state."""
        return self.fk.compute(self._joint_angles)

    def fk_from_angles(self, angles_deg: list) -> dict:
        """
        Compute FK for any given joint angles (in degrees).
        Useful for testing and visualisation.
        """
        return self.fk.compute(np.radians(angles_deg))

    # ── Inverse Kinematics ────────────────────────────────────────────────────
    def move_to_position(
        self,
        target_xyz:     list,
        target_rot:     Optional[np.ndarray] = None,
        initial_guess:  Optional[list]       = None,
        send:           bool                 = False,
    ) -> dict:
        """
        Solve IK for a Cartesian target and optionally send to ESP32.

        Parameters
        ----------
        target_xyz    : [x, y, z] in metres
        target_rot    : 3×3 rotation matrix (optional, for full 6D control)
        initial_guess : joint angles in degrees to seed the solver (optional)
        send          : if True, sends the result to the ESP32 over WiFi

        Returns
        -------
        IK result dict (see InverseKinematics.solve)
        """
        target    = np.array(target_xyz, dtype=float)
        guess_rad = np.radians(initial_guess) if initial_guess is not None else None

        result = self.ik.solve(target, target_rot, guess_rad)

        if result["success"]:
            self._joint_angles = result["joint_angles"]
            log.info(f"IK solved → {np.round(result['joint_angles_deg'], 2)} deg")

            if send:
                self.controller.send_joint_angles(result["joint_angles_deg"])
        else:
            log.warning(f"IK failed: {result['message']}")

        return result

    def move_to_position_decoupled(
        self,
        target_xyz:  list,
        target_rot:  np.ndarray,
        elbow_up:    Optional[bool] = None,
        send:        bool           = False,
    ) -> dict:
        """
        Solve IK analytically using kinematic decoupling.

        Requires target_rot — orientation must always be specified because
        the wrist center calculation depends on the target Z-axis direction.

        Parameters
        ----------
        target_xyz : [x, y, z] in metres
        target_rot : 3×3 rotation matrix — required
        elbow_up   : True = elbow-up, False = elbow-down, None = auto
        send       : if True, sends result to ESP32 over WiFi

        Returns
        -------
        Same dict as move_to_position(), plus 'elbow_up' key.

        When to use this vs move_to_position():
        ─────────────────────────────────────────
        move_to_position()           — numerical, works without target_rot,
                                       slower, handles any geometry
        move_to_position_decoupled() — analytical, needs target_rot,
                                       instant, more predictable configs
        """
        target = np.array(target_xyz, dtype=float)
        result = self.ik_decoupled.solve(
            target, target_rot, elbow_up, current_angles=self._joint_angles
        )

        if result["success"]:
            self._joint_angles = result["joint_angles"]
            log.info(
                f"Decoupled IK solved ({'elbow-up' if result['elbow_up'] else 'elbow-down'}) "
                f"→ {np.round(result['joint_angles_deg'], 2)} deg"
            )
            if send:
                self.controller.send_joint_angles(result["joint_angles_deg"])
        else:
            log.warning(f"Decoupled IK failed: {result['message']}")

        return result

    def set_joint_angles(self, angles_deg: list, send: bool = False):
        """
        Directly set joint angles (degrees), bypassing IK.
        Useful for manual control or homing sequences.
        """
        q = np.radians(angles_deg)
        q = self.ik._clamp_joints(q)
        self._joint_angles = q

        log.info(f"Joint angles set → {np.round(np.degrees(q), 2)} deg")

        if send:
            self.controller.send_joint_angles(np.degrees(q))

    # ── Workspace check ───────────────────────────────────────────────────────
    def is_reachable(self, target_xyz: list) -> bool:
        """
        Quick NECESSARY check: is the target within max reach of the SHOULDER?
        (Sphere centred on the shoulder pivot at height d1, matching the plot.
        Passing does not guarantee a solution — joint limits and the inner
        dead zone are not modelled.)
        """
        cfg        = self.config
        max_reach  = cfg.a2 + cfg.a3 + cfg.d6
        shoulder   = np.array([0.0, 0.0, cfg.d1])
        distance   = np.linalg.norm(np.asarray(target_xyz, dtype=float) - shoulder)
        reachable  = distance <= max_reach

        if not reachable:
            log.warning(
                f"Target {target_xyz} is out of reach. "
                f"Distance: {distance*1000:.1f} mm, Max: {max_reach*1000:.1f} mm"
            )
        return reachable


# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZER
# ─────────────────────────────────────────────────────────────────────────────
class ArmVisualizer:
    """
    Matplotlib-based 3D visualizer for the robot arm.

    Three modes
    -----------
    1. show_pose(angles_deg)
       Draws the arm at a single static pose. Good for quickly checking
       whether an FK or IK result looks physically correct.

    2. show_trajectory(waypoints_deg)
       Draws the arm stepping through a list of poses. Each pose is shown
       as a faded ghost so you can see the full motion arc at once.

    3. animate_trajectory(waypoints_deg, fps, interpolation_steps)
       Smoothly animates the arm moving between waypoints.
       Safe to run before touching hardware — if the motion looks wrong
       here, your IK target or joint limits need adjusting.

    Visual conventions
    ------------------
    - Links          : thick blue line connecting joint origins
    - Joint origins  : orange dots
    - End-effector   : red star
    - EE axes frame  : RGB arrows (X=red, Y=green, Z=blue)
    - Workspace sphere: dashed sphere at max reach (shown in pose mode)
    - Trajectory path : thin dotted line tracing EE positions
    - Ghost poses     : faded blue links showing previous waypoints
    """

    # Colours kept as class constants — easy to change
    LINK_COLOR       = "#2E86DE"   # blue
    JOINT_COLOR      = "#F39C12"   # orange
    EE_COLOR         = "#E74C3C"   # red
    GHOST_COLOR      = "#AED6F1"   # light blue for trajectory ghosts
    PATH_COLOR       = "#E74C3C"   # red dotted path
    WORKSPACE_COLOR  = "#95A5A6"   # gray sphere

    def __init__(self, arm: "RobotArm"):
        self.arm    = arm
        self.config = arm.config
        self.fk     = arm.fk

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────────────────────────────────────

    def show_pose(
        self,
        angles_deg: List[float],
        show_workspace: bool = True,
        title: str = "Robot arm pose"
    ) -> None:
        """
        Render a single static pose.

        Parameters
        ----------
        angles_deg      : 6 joint angles in degrees
        show_workspace  : draw the max-reach sphere if True
        title           : window title string
        """
        fig, ax = self._make_figure(title)
        joints  = self._joint_positions(np.radians(angles_deg))

        self._draw_arm(ax, joints, alpha=1.0)
        self._draw_ee_frame(ax, np.radians(angles_deg))

        if show_workspace:
            self._draw_workspace_sphere(ax)

        self._set_axes(ax, joints, np.radians(angles_deg))
        plt.tight_layout()
        plt.show()

    def show_trajectory(
        self,
        waypoints_deg: List[List[float]],
        title: str = "Robot arm trajectory"
    ) -> None:
        """
        Draw all waypoints at once as overlapping ghost poses.

        The final pose is drawn fully opaque; earlier poses are faded.
        The end-effector path is shown as a dotted red line.

        Parameters
        ----------
        waypoints_deg : list of 6-element angle lists, one per waypoint
        title         : window title string
        """
        fig, ax   = self._make_figure(title)
        n         = len(waypoints_deg)
        ee_path   = []

        for i, angles_deg in enumerate(waypoints_deg):
            q      = np.radians(angles_deg)
            joints = self._joint_positions(q)
            # Final pose is full opacity; earlier poses fade out
            alpha  = 0.15 + 0.85 * (i / max(n - 1, 1))
            color  = self.LINK_COLOR if i == n - 1 else self.GHOST_COLOR
            self._draw_arm(ax, joints, alpha=alpha, link_color=color)
            ee_path.append(joints[-1])

        # Draw EE frame only for the final pose
        self._draw_ee_frame(ax, np.radians(waypoints_deg[-1]))

        # Dotted path connecting all EE positions
        ee_path = np.array(ee_path)
        ax.plot(
            ee_path[:, 0], ee_path[:, 1], ee_path[:, 2],
            color=self.PATH_COLOR, linewidth=1.0,
            linestyle="dotted", label="EE path"
        )
        ax.legend(fontsize=8)

        all_joints = np.vstack([
            self._joint_positions(np.radians(a)) for a in waypoints_deg
        ])
        self._set_axes(ax, all_joints, np.radians(waypoints_deg[-1]))
        plt.tight_layout()
        plt.show()

    def animate_trajectory(
        self,
        waypoints_deg: List[List[float]],
        fps: int = 30,
        interpolation_steps: int = 20,
        title: str = "Robot arm animation"
    ) -> "animation.FuncAnimation":
        """
        Smoothly animate the arm moving through waypoints.

        Joint angles are linearly interpolated between consecutive waypoints
        so the motion looks fluid rather than jumping frame-to-frame.

        Parameters
        ----------
        waypoints_deg        : list of 6-element angle lists
        fps                  : playback speed in frames per second
        interpolation_steps  : number of intermediate frames between each
                               pair of waypoints (higher = smoother)
        title                : window title string

        Notes
        -----
        Close the window to stop the animation.
        The animation loops continuously so you can watch the full motion
        arc as many times as needed before committing to hardware.
        """
        # Build the full interpolated frame list
        frames = self._interpolate_waypoints(waypoints_deg, interpolation_steps)
        n      = len(frames)

        fig, ax = self._make_figure(title)

        # Compute axis bounds from all frames up front so they don't jump
        all_joints = np.vstack([self._joint_positions(f) for f in frames])
        self._set_axes(ax, all_joints, frames[0])

        # Collect EE path across all frames for the dotted trail
        ee_path = np.array([self._joint_positions(f)[-1] for f in frames])

        # Plot objects we'll update each frame — store as list so
        # we can clear and redraw without recreating the axes
        arm_lines   = []
        joint_dots  = []
        ee_dot      = []
        ee_frame_artists = []

        def _init():
            return []

        def _update(frame_idx: int):
            # Clear previous frame's artists
            for artist in arm_lines + joint_dots + ee_dot + ee_frame_artists:
                artist.remove()
            arm_lines.clear()
            joint_dots.clear()
            ee_dot.clear()
            ee_frame_artists.clear()

            q      = frames[frame_idx]
            joints = self._joint_positions(q)

            # Links
            for seg in self._link_segments(joints):
                line, = ax.plot(
                    [seg[0][0], seg[1][0]],
                    [seg[0][1], seg[1][1]],
                    [seg[0][2], seg[1][2]],
                    color=self.LINK_COLOR, linewidth=3.5, solid_capstyle="round"
                )
                arm_lines.append(line)

            # Joint dots
            for j in joints[:-1]:
                dot, = ax.plot(
                    [j[0]], [j[1]], [j[2]],
                    "o", color=self.JOINT_COLOR, markersize=7, zorder=5
                )
                joint_dots.append(dot)

            # End-effector star
            ee = joints[-1]
            star, = ax.plot(
                [ee[0]], [ee[1]], [ee[2]],
                "*", color=self.EE_COLOR, markersize=12, zorder=6
            )
            ee_dot.append(star)

            # EE coordinate frame arrows
            for artist in self._ee_frame_artists(ax, q):
                ee_frame_artists.append(artist)

            # Faded path trail up to current frame
            trail, = ax.plot(
                ee_path[:frame_idx+1, 0],
                ee_path[:frame_idx+1, 1],
                ee_path[:frame_idx+1, 2],
                color=self.PATH_COLOR, linewidth=1.0,
                linestyle="dotted", alpha=0.6
            )
            ee_frame_artists.append(trail)

            # Frame counter text in corner
            ax.set_title(
                f"{title}  |  frame {frame_idx+1}/{n}",
                fontsize=10, pad=4
            )
            return arm_lines + joint_dots + ee_dot + ee_frame_artists

        interval_ms = int(1000 / fps)
        ani = animation.FuncAnimation(
            fig, _update, frames=n,
            init_func=_init,
            interval=interval_ms,
            blit=False,   # blit=True causes glitches with 3D axes
            repeat=True
        )
        self._anim = ani          # keep a reference so it isn't garbage-collected

        plt.tight_layout()
        plt.show()
        return ani

    # ─────────────────────────────────────────────────────────────────────────
    # INTERNAL HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    def _joint_positions(self, q: np.ndarray) -> np.ndarray:
        """
        Return (7,3) array of XYZ positions for the base + 6 joint origins.
        Index 0 = world origin, index 6 = end-effector.
        Uses return_all_frames=True from FK to get every intermediate frame.
        """
        result = self.fk.compute(q, return_all_frames=True)
        # Each frame's origin is its translation column T[:3, 3]
        return np.array([T[:3, 3] for T in result["frames"]])

    def _link_segments(self, joints: np.ndarray):
        """Yield (start, end) pairs for each link."""
        for i in range(len(joints) - 1):
            yield joints[i], joints[i + 1]

    def _draw_arm(
        self,
        ax,
        joints:     np.ndarray,
        alpha:      float = 1.0,
        link_color: str   = None
    ) -> None:
        """Draw links and joint dots for one pose."""
        color = link_color or self.LINK_COLOR

        for seg in self._link_segments(joints):
            ax.plot(
                [seg[0][0], seg[1][0]],
                [seg[0][1], seg[1][1]],
                [seg[0][2], seg[1][2]],
                color=color, linewidth=3.5,
                alpha=alpha, solid_capstyle="round"
            )

        for j in joints[:-1]:
            ax.plot(
                [j[0]], [j[1]], [j[2]],
                "o", color=self.JOINT_COLOR,
                markersize=7, alpha=alpha, zorder=5
            )

        ee = joints[-1]
        ax.plot(
            [ee[0]], [ee[1]], [ee[2]],
            "*", color=self.EE_COLOR,
            markersize=12, alpha=alpha, zorder=6
        )

    def _ee_frame_artists(self, ax, q: np.ndarray, length: float = 0.025):
        """
        Draw the end-effector coordinate frame as three RGB arrows.
        Returns the list of quiver artists so the caller can remove them.
        X = red, Y = green, Z = blue — matches standard robotics convention.
        """
        result  = self.fk.compute(q)
        origin  = result["position"]
        R       = result["rotation"]
        artists = []

        colors = ["#E74C3C", "#2ECC71", "#2E86DE"]   # X, Y, Z
        for i, c in enumerate(colors):
            axis = R[:, i] * length
            q_art = ax.quiver(
                origin[0], origin[1], origin[2],
                axis[0],   axis[1],   axis[2],
                color=c, linewidth=1.5, arrow_length_ratio=0.3
            )
            artists.append(q_art)

        return artists

    def _draw_ee_frame(self, ax, q: np.ndarray, length: float = 0.025) -> None:
        """Draw EE frame arrows (static version — no return needed)."""
        self._ee_frame_artists(ax, q, length)

    def _draw_workspace_sphere(self, ax) -> None:
        """
        Draw a dashed wireframe sphere representing the arm's maximum reach.
        Gives an immediate visual sense of whether a target is reachable.
        """
        cfg       = self.config
        r         = cfg.a2 + cfg.a3 + cfg.d6
        u         = np.linspace(0, 2 * np.pi, 30)
        v         = np.linspace(0, np.pi, 20)
        xs        = r * np.outer(np.cos(u), np.sin(v))
        ys        = r * np.outer(np.sin(u), np.sin(v))
        zs        = r * np.outer(np.ones_like(u), np.cos(v))
        # Shift sphere up by d1 so it's centred on the shoulder, not the floor
        zs       += cfg.d1

        ax.plot_wireframe(
            xs, ys, zs,
            color=self.WORKSPACE_COLOR, linewidth=0.3,
            alpha=0.15, linestyle="dashed"
        )

    def _set_axes(self, ax, joints: np.ndarray, q: Optional[np.ndarray] = None) -> None:
        """
        Set equal-aspect 3D axes scaled to the arm's geometry.
        Adds a ground plane grid and labels the axes in mm.
        """
        cfg    = self.config
        extent = (cfg.a2 + cfg.a3 + cfg.d6) * 1.2

        ax.set_xlim(-extent, extent)
        ax.set_ylim(-extent, extent)
        zmin = min(0.0, cfg.d1 - (cfg.a2 + cfg.a3 + cfg.d6)) * 1.05
        ax.set_zlim(zmin, extent * 1.5)

        # Convert axis ticks to mm for readability
        ticks_m  = np.linspace(-extent, extent, 5)
        ticks_mm = [f"{int(t*1000)}" for t in ticks_m]
        ax.set_xticks(ticks_m); ax.set_xticklabels(ticks_mm, fontsize=7)
        ax.set_yticks(ticks_m); ax.set_yticklabels(ticks_mm, fontsize=7)
        z_ticks_m  = np.linspace(zmin, extent * 1.5, 5)
        z_ticks_mm = [f"{int(t*1000)}" for t in z_ticks_m]
        ax.set_zticks(z_ticks_m); ax.set_zticklabels(z_ticks_mm, fontsize=7)

        ax.set_xlabel("X (mm)", fontsize=8, labelpad=2)
        ax.set_ylabel("Y (mm)", fontsize=8, labelpad=2)
        ax.set_zlabel("Z (mm)", fontsize=8, labelpad=2)

        # Draw a faint ground plane grid
        ax.grid(True, linewidth=0.3, alpha=0.4)
        ax.set_box_aspect([1, 1, 1])

        # Joint labels alongside each dot
        q_label   = q if q is not None else self.arm._joint_angles
        fk_result = self.fk.compute(q_label, return_all_frames=True)
        names = ["Base", "Shoulder", "Elbow", "Wrist"]      # frames 3-6 share the wrist point
        for label, T in zip(names, fk_result["frames"][:4]):
            p = T[:3, 3]
            ax.text(
                p[0], p[1], p[2] + extent * 0.04,
                label, fontsize=7, ha="center", color="#555555"
            )

    @staticmethod
    def _make_figure(title: str):
        """Create a styled 3D figure and return (fig, ax)."""
        fig = plt.figure(figsize=(9, 7))
        ax  = fig.add_subplot(111, projection="3d")
        ax.set_facecolor("#F8F9FA")
        fig.patch.set_facecolor("#FFFFFF")
        ax.set_title(title, fontsize=11, pad=8)
        ax.view_init(elev=25, azim=45)   # comfortable default viewing angle
        return fig, ax

    @staticmethod
    def _interpolate_waypoints(
        waypoints_deg:      List[List[float]],
        interpolation_steps: int
    ) -> List[np.ndarray]:
        """
        Linearly interpolate joint angles between consecutive waypoints.
        Returns a flat list of angle arrays (in radians) ready for FK.

        Linear interpolation is used deliberately — it's the simplest
        motion primitive and lets you see exactly what the IK solver
        produced without any smoothing hiding errors.
        """
        frames = []
        for i in range(len(waypoints_deg) - 1):
            start = np.radians(waypoints_deg[i])
            end   = np.radians(waypoints_deg[i + 1])
            for s in range(interpolation_steps):
                t = s / interpolation_steps
                frames.append(start + t * (end - start))
        frames.append(np.radians(waypoints_deg[-1]))
        return frames


# ─────────────────────────────────────────────────────────────────────────────
# DEMO / QUICK TEST  (run this file directly to verify your setup)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    print("=" * 60)
    print("  6-DOF Robot Arm — Kinematics Self-Test + Visualizer")
    print("=" * 60)

    arm = RobotArm()
    viz = ArmVisualizer(arm)

    # ── Test 1: Forward Kinematics at home position ────────────────────────────
    print("\n[TEST 1] FK at home position (all zeros)")
    pose = arm.fk_from_angles([0, 0, 0, 0, 0, 0])
    print(f"  Position  : {np.round(pose['position']*1000, 2)} mm")
    print(f"  Euler XYZ : {np.round(pose['euler_xyz'], 2)} deg")

    # ── Test 2: FK at a known configuration ───────────────────────────────────
    print("\n[TEST 2] FK at [0, -45, -60, 0, 90, 0] deg")
    "pose2 = arm.fk_from_angles([0, -45, -60, 0, 45, 0])"
    'remember to change j5 back to 45 degrees'
    pose2 = arm.fk_from_angles([0, -45, -60, 0, 90, 0])

    print(f"  Position  : {np.round(pose2['position']*1000, 2)} mm")
    print(f"  Euler XYZ : {np.round(pose2['euler_xyz'], 2)} deg")

    # ── Test 3: IK — solve for FK result from Test 2 ──────────────────────────
    print("\n[TEST 3] Numerical IK — recover joint angles from FK position above")
    target_pos = pose2["position"]
    result = arm.move_to_position(
        target_pos.tolist(),
        target_rot=pose2["rotation"],
        initial_guess=[0, -30, -45, 0, 30, 0]   # non-singular seed
    )
    print(f"  Success     : {result['success']}")
    print(f"  Angles (deg): {np.round(result['joint_angles_deg'], 2)}")
    print(f"  Pos error   : {result['position_error']*1000:.4f} mm")
    print(f"  Iterations  : {result['iterations']}")
    print(f"  Message     : {result['message']}")

    # ── Test 3b: Decoupled IK — same target ───────────────────────────────────
    print("\n[TEST 3b] Decoupled IK — same target, analytical solution")
    result_d = arm.move_to_position_decoupled(
        target_pos.tolist(),
        target_rot=pose2["rotation"],
    )
    print(f"  Success     : {result_d['success']}")
    print(f"  Angles (deg): {np.round(result_d['joint_angles_deg'], 2)}")
    print(f"  Pos error   : {result_d['position_error']*1000:.4f} mm")
    print(f"  Elbow config: {'elbow-up' if result_d['elbow_up'] else 'elbow-down'}")
    print(f"  Message     : {result_d['message']}")

    # ── Test 4: Workspace / reachability ──────────────────────────────────────
    print("\n[TEST 4] Reachability checks")
    tests = [
        [0.10, 0.05, 0.15],
        [0.50, 0.50, 0.50],
    ]
    for t in tests:
        r = arm.is_reachable(t)
        print(f"  {[round(x*1000) for x in t]} mm → {'REACHABLE' if r else 'OUT OF REACH'}")

    # ── Test 5: decoupled IK round-trip over random in-limit poses ────────────
    print("\n[TEST 5] Decoupled IK round-trip, 200 random in-limit poses")
    _rng, _lims, _ok = np.random.default_rng(0), arm.config.joint_limits_rad, 0
    for _ in range(200):
        _q = _rng.uniform(_lims[:, 0], _lims[:, 1])
        _T = arm.fk.compute(_q)
        _ok += arm.ik_decoupled.solve(_T["position"], _T["rotation"])["success"]
    print(f"  exact pose recovered: {_ok}/200")

    if "--no-gui" in sys.argv:
        print("\n[DONE] --no-gui: skipping visualizations.")
        sys.exit(0)

    # ── Visualization 1: single static pose ───────────────────────────────────
    # Shows the arm at a specific pose with the workspace sphere.
    # Close the window to continue.
    print("\n[VIZ 1] Displaying static pose [0, -45, -60, 0, 30, 0] deg ...")
    print("        (close window to continue)")
    arm.set_joint_angles([0, -45, -30, 180, 0, 90])
    viz.show_pose(
        [0, -45, -30, 180, 0, 90],
        show_workspace=True,
        title="Static pose — [0, -45, -60, 0, 30, 0] deg"
    )

    # ── Visualization 2: trajectory ghost overlay ──────────────────────────────
    # Define a sequence of IK targets. The arm sweeps across a small arc.
    # Verify visually that the path looks physically plausible before
    # sending any commands to the ESP32.
    print("\n[VIZ 2] Solving IK for a 5-waypoint trajectory ...")
    targets = [
        [0.10,  0.050, 0.20],
        [0.10,  0.040, 0.18],
        [0.12,  0.030, 0.14],
        [0.15,  0.030, 0.13],
        [0.16,  0.020, 0.16],
    ]

    waypoints_deg = []
    prev_guess = [0, -45, -60, 0, 60, 0]

    for i, xyz in enumerate(targets):
        ik_result = arm.move_to_position(xyz, initial_guess=prev_guess)
        if ik_result["success"]:
            angles = ik_result["joint_angles_deg"].tolist()
            waypoints_deg.append(angles)
            prev_guess = angles
            print(f"  Waypoint {i+1}: {[round(x*1000) for x in xyz]} mm  "
                  f"→  {np.round(angles, 1)} deg")
        else:
            print(f"  Waypoint {i+1}: IK failed — {ik_result['message']}")

    if len(waypoints_deg) >= 2:
        print("\n        Displaying trajectory ghost overlay ...")
        print("        (close window to continue)")
        viz.show_trajectory(
            waypoints_deg,
            title="Trajectory — ghost overlay of all waypoints"
        )

        # ── Visualization 3: smooth animation ─────────────────────────────────
        # Same trajectory, now animated. Watch for any unexpected swings,
        # elbow flips, or joint-limit violations before touching hardware.
        print("\n[VIZ 3] Animating trajectory (loops until window closed) ...")
        viz.animate_trajectory(
            waypoints_deg,
            fps=30,
            interpolation_steps=25,
            title="Animated trajectory — verify before sending to ESP32"
        )

    print("\n[DONE] All tests and visualizations complete.")
    print("       When the motion looks correct in the visualizer:")
    print("         1. Set esp32_ip in RobotConfig")
    print("         2. Call arm.controller.connect()")
    print("         3. Replay move_to_position(..., send=True) for each waypoint")
    print("=" * 60)