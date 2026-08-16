"""Write a motion_command capture as the reference-trajectory NPZ the sim task loads.

The deployment CSV (`export_csv.py`) is not enough for simulation. `ReferenceTrajectory`
in shape14 requires body-level state as well:

    fps, joint_pos, joint_vel, body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w

We only ever produce joint angles and a root pose, so the body arrays are recovered by
forward kinematics: set the robot's qpos in MuJoCo, `mj_forward`, and read the body frames
out. That is the same model and the same mapping-by-name the live display already uses.

Details taken from their loader and from an existing clip, not assumed:

* **`quat_order` is xyzw**, while MuJoCo hands back wxyz — the same flip the CSV needed.
* **Body order follows the file's own `body_names`**, which the loader resolves against the
  runtime's tracked bodies. Our MJCF carries extra proxy bodies, so we select and order by
  the reference clip's names rather than dumping every body MuJoCo knows.
* **Velocities are finite differences at fps**, matching how `joint_vel` is derived
  elsewhere in this pipeline; the last frame repeats the previous difference so the arrays
  stay the same length as the positions.
* **fps must equal the policy rate** or the loader raises — ours is 50 Hz by construction.

    .venv-ik/bin/python export_npz.py outputs/feas_mc.npz outputs/clips/live_take1
"""

import argparse
from pathlib import Path

import mujoco
import numpy as np

from export_csv import write_csv
from ik_server import MJCF

# the 24 bodies a K1_rev1 reference clip carries, in file order
REF_CLIP = ("/home/robotis-ai/Projects/shape14/cyclo_lab_private/source/cyclo_lab/"
            "data/motions/K1_rev1/aiming1/aiming1.npz")


def body_states(robot, joint_pos, root, body_names):
    """FK every frame and read out the requested bodies, in the requested order."""
    model = mujoco.MjModel.from_xml_path(MJCF[robot])
    data = mujoco.MjData(model)
    bids = []
    for b in body_names:
        i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b)
        if i < 0:
            raise KeyError(f"{b} is not a body in {MJCF[robot]}")
        bids.append(i)
    return model, data, bids


def export(npz_in, out_dir, robot="k1", name=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = name or out_dir.name

    src = np.load(npz_in, allow_pickle=True)
    q = np.asarray(src["joint_pos"], np.float32)
    joint_names = [str(n) for n in src["joint_names"]]
    fps = float(src["rate"]) if "rate" in src.files else 50.0
    n = len(q)
    root = (np.asarray(src["root"], np.float64) if "root" in src.files and len(src["root"]) == n
            else np.tile([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], (n, 1)))

    body_names = [str(b) for b in np.load(REF_CLIP, allow_pickle=True)["body_names"]]
    model, data, bids = body_states(robot, q, root, body_names)

    # map our joint vector onto qpos by name, never by order
    qadr = []
    for jn in joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        if jid < 0:
            raise KeyError(f"{jn} is not a joint in {MJCF[robot]}")
        qadr.append(model.jnt_qposadr[jid])
    free = model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE

    body_pos = np.zeros((n, len(bids), 3), np.float32)
    body_quat = np.zeros((n, len(bids), 4), np.float32)
    for i in range(n):
        if free:
            data.qpos[0:3] = root[i, 0:3]
            data.qpos[3:7] = root[i, [6, 3, 4, 5]]  # file xyzw -> MuJoCo wxyz
        for a, v in zip(qadr, q[i]):
            data.qpos[a] = v
        mujoco.mj_forward(model, data)
        body_pos[i] = data.xpos[bids]
        body_quat[i] = data.xquat[bids][:, [1, 2, 3, 0]]  # MuJoCo wxyz -> file xyzw

    def d_dt(a):
        v = np.diff(a, axis=0) * fps
        return np.concatenate([v, v[-1:]], axis=0).astype(np.float32) if len(v) else np.zeros_like(a)

    out = {
        "fps": np.array([fps], np.int64),
        "joint_names": np.array(joint_names),
        "body_names": np.array(body_names),
        "quat_order": np.array("xyzw"),
        "joint_pos": q,
        "joint_vel": d_dt(q.astype(np.float64)),
        "body_pos_w": body_pos,
        "body_quat_w": body_quat,
        "body_lin_vel_w": d_dt(body_pos.astype(np.float64)),
        # angular velocity from quaternions is not a difference; the finite-difference
        # stand-in below is fine for playback but should be revisited if a reward reads it
        "body_ang_vel_w": _ang_vel(body_quat, fps),
    }
    np.savez(out_dir / f"{name}.npz", **out)
    write_csv(npz_in, out_dir / f"{name}.csv")  # the runtime path wants the CSV too
    return out_dir / f"{name}.npz", n, len(body_names)


def _ang_vel(quat_xyzw, fps):
    """Angular velocity from consecutive orientations: 2 * (dq * conj(q)) vector part."""
    q = quat_xyzw[:, :, [3, 0, 1, 2]].astype(np.float64)  # wxyz for the algebra
    q0, q1 = q[:-1], q[1:]
    # flip sign where the quaternion crossed the double cover, or the difference explodes
    q1 = np.where((q0 * q1).sum(-1, keepdims=True) < 0, -q1, q1)
    w0, v0 = q0[..., :1], q0[..., 1:]
    w1, v1 = q1[..., :1], q1[..., 1:]
    # q1 * conj(q0)
    vec = w1 * (-v0) + w0 * v1 - np.cross(v1, -v0)
    omega = 2.0 * vec * fps
    return np.concatenate([omega, omega[-1:]], axis=0).astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("npz", help="a motion_command capture")
    p.add_argument("out_dir", help="clip directory to create, named after the clip")
    p.add_argument("--robot", default="k1", choices=["g1", "k1"])
    a = p.parse_args()
    path, n, nb = export(a.npz, a.out_dir, a.robot)
    print(f"{n} frames, {nb} bodies -> {path}")


if __name__ == "__main__":
    main()
