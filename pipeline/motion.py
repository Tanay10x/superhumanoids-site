"""Kinematic motion scripts for each robot. Pure mujoco + numpy (no bpy).

Each robot returns (model, list_of_qpos, extras) where extras may carry
per-frame transforms for props (e.g. a cube the arm picks up).
"""
import math
import numpy as np
import mujoco

ROOT = "/home/tanay/sh-build/models"


def smooth(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * t * (t * (6 * t - 15) + 10)


def jadr(m, name):
    j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
    assert j >= 0, name
    return m.jnt_qposadr[j]


def humanoid(n):
    m = mujoco.MjModel.from_xml_path(f"{ROOT}/humanoid/humanoid.xml")
    base = m.key_qpos[0].copy()
    # turn the robot slightly toward camera-left so the raised arm reads well
    frames = []
    for i in range(n):
        t = i / (n - 1)
        q = base.copy()
        a = smooth((t - 0.15) / 0.45)          # arm raise
        w = smooth((t - 0.55) / 0.35)          # hand open + wrist turn
        q[jadr(m, "right_shoulder_pitch_joint")] = 0.2 - 0.75 * a
        q[jadr(m, "right_shoulder_roll_joint")] = -0.2 - 1.15 * a
        q[jadr(m, "right_shoulder_yaw_joint")] = 0.0 + 0.9 * a
        q[jadr(m, "right_elbow_joint")] = 1.28 - 0.15 * a
        q[jadr(m, "right_wrist_roll_joint")] = -0.6 * w
        q[jadr(m, "waist_yaw_joint")] = 0.18 * smooth(t / 0.8)
        q[jadr(m, "left_shoulder_pitch_joint")] = 0.2 + 0.08 * math.sin(t * math.pi)
        for jn, v in [("right_hand_thumb_0_joint", 0.0), ("right_hand_middle_0_joint", 1.05),
                      ("right_hand_middle_1_joint", 1.2), ("right_hand_index_0_joint", 1.05),
                      ("right_hand_index_1_joint", 1.2)]:
            q[jadr(m, jn)] = v * (1 - w)
        frames.append(q)
    # breathing-like knee flex keeps feet planted: compensate base height
    return m, frames, {}


def quadruped(n):
    m = mujoco.MjModel.from_xml_path(f"{ROOT}/quadruped_clean/quadruped.xml")  # vendor lettering erased
    base = m.key_qpos[0].copy()
    legs = ["FL", "FR", "RL", "RR"]
    phase = {"FL": 0.0, "RR": 0.0, "FR": 0.5, "RL": 0.5}
    frames = []
    cycles = 2.0
    for i in range(n):
        t = i / (n - 1)
        env = smooth(t / 0.15) * (1 - smooth((t - 0.85) / 0.15))
        q = base.copy()
        for L in legs:
            p = (t * cycles + phase[L]) % 1.0
            lift = math.sin(p * 2 * math.pi) if p < 0.5 else 0.0
            lift *= env
            q[jadr(m, f"{L}_thigh_joint")] = 0.9 - 0.25 * lift
            q[jadr(m, f"{L}_calf_joint")] = -1.8 - 0.55 * lift
        q[2] = 0.27 + 0.006 * math.sin(t * cycles * 4 * math.pi) * env
        frames.append(q)
    return m, frames, {}


def arm(n):
    m = mujoco.MjModel.from_xml_path(f"{ROOT}/arm/arm.xml")
    d = mujoco.MjData(m)
    home = m.key_qpos[0].copy()
    site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "hand")
    cube = np.array([0.55, -0.12, 0.025])
    hand_off = 0.105  # hand body to fingertip center

    def ik(q0, target, iters=200):
        q = q0.copy()
        R_goal = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], float)
        for _ in range(iters):
            d.qpos[:] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            p = d.xpos[site] + d.xmat[site].reshape(3, 3) @ np.array([0, 0, hand_off])
            R = d.xmat[site].reshape(3, 3)
            ep = target - p
            er = 0.5 * (np.cross(R[:, 0], R_goal[:, 0]) + np.cross(R[:, 1], R_goal[:, 1]) + np.cross(R[:, 2], R_goal[:, 2]))
            jp = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv))
            mujoco.mj_jac(m, d, jp, jr, p, site)
            J = np.vstack([jp, jr])[:, :7]
            e = np.concatenate([ep, er * 0.6])
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-3 * np.eye(6), e)
            q[:7] += dq
            if np.linalg.norm(e) < 1e-5:
                break
        return q

    above = ik(home, cube + [0, 0, 0.22])
    grasp = ik(above, cube + [0, 0, 0.0])
    lift = ik(grasp, cube + [0.05, 0.14, 0.3])
    keys = [(0.0, home, 0.04), (0.3, above, 0.04), (0.5, grasp, 0.04), (0.6, grasp, 0.025), (0.95, lift, 0.025)]
    frames, cube_pose = [], []
    for i in range(n):
        t = i / (n - 1)
        for k in range(len(keys) - 1):
            if keys[k][0] <= t <= keys[k + 1][0]:
                a = smooth((t - keys[k][0]) / (keys[k + 1][0] - keys[k][0]))
                q = keys[k][1] * (1 - a) + keys[k + 1][1] * a
                g = keys[k][2] * (1 - a) + keys[k + 1][2] * a
                break
        else:
            q, g = keys[-1][1], keys[-1][2]
        q = q.copy(); q[7] = q[8] = g
        frames.append(q)
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        if t >= 0.6:
            p = d.xpos[site] + d.xmat[site].reshape(3, 3) @ np.array([0, 0, hand_off])
            cube_pose.append((p.tolist(), d.xquat[site].tolist()))
        else:
            cube_pose.append((cube.tolist(), [1, 0, 0, 0]))
    return m, frames, {"cube": cube_pose}


ROBOTS = {"humanoid": humanoid, "quadruped": quadruped, "arm": arm}
