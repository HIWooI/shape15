"""Publish reference frames over UDP, so a simulator can be driven live.

The clip path writes a whole take to an NPZ and the simulator loads it. That is the right
shape for validating a reference and the wrong shape for a demo, where the robot should
follow the person now. Everything the policy reads comes from `reference.<array>[frame_ids]`
in shape14's command term, so a live source only has to produce the same per-frame rows.

One datagram per 50 Hz step, little-endian float32, in this order:

    seq(1) joint_pos(J) joint_vel(J) body_pos_w(B*3) body_quat_w(B*4)
    body_lin_vel_w(B*3) body_ang_vel_w(B*3)

The body arrays are recovered by forward kinematics exactly as `export_npz.py` does, and
the velocities are finite differences against the previous published frame. UDP because a
late reference frame is worthless: the receiver wants the newest one, and dropping is the
correct response to falling behind, not queueing.

    # test the receiver with no camera: stream a saved clip in real time
    .venv-ik/bin/python ref_stream.py --replay outputs/clips/take5b/take5b.npz
"""

import socket
import struct
import time

import mujoco
import numpy as np

from export_npz import REF_CLIP
from ik_server import MJCF

DEFAULT_ADDR = ("127.0.0.1", 9411)


class Publisher:
    """Turns joint angles plus a root pose into the rows the simulator's command reads."""

    def __init__(self, robot="k1", joint_names=None, addr=DEFAULT_ADDR, fps=50.0):
        self.model = mujoco.MjModel.from_xml_path(MJCF[robot])
        self.data = mujoco.MjData(self.model)
        self.body_names = [str(b) for b in np.load(REF_CLIP, allow_pickle=True)["body_names"]]
        self.bids = []
        for b in self.body_names:
            i = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, b)
            if i < 0:
                raise KeyError(f"{b} is not a body in {MJCF[robot]}")
            self.bids.append(i)
        self.joint_names = list(joint_names) if joint_names is not None else None
        self.qadr = None
        if self.joint_names:
            self._bind(self.joint_names)
        self.free = self.model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE
        self.dt = 1.0 / fps
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.addr = addr
        self.seq = 0
        self.prev = None  # (joint_pos, body_pos, body_quat) for the differences

    def _bind(self, joint_names):
        self.qadr = []
        for jn in joint_names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jn)
            if jid < 0:
                raise KeyError(f"{jn} is not a joint")
            self.qadr.append(self.model.jnt_qposadr[jid])

    def send(self, joint_pos, root):
        """joint_pos in our URDF order; root as x,y,z,qx,qy,qz,qw."""
        q = np.asarray(joint_pos, np.float64)
        root = np.asarray(root, np.float64)
        if self.free:
            self.data.qpos[0:3] = root[0:3]
            self.data.qpos[3:7] = root[[6, 3, 4, 5]]  # file xyzw -> MuJoCo wxyz
        for a, v in zip(self.qadr, q):
            self.data.qpos[a] = v
        mujoco.mj_forward(self.model, self.data)
        bpos = self.data.xpos[self.bids].copy()
        bquat = self.data.xquat[self.bids][:, [1, 2, 3, 0]].copy()  # wxyz -> xyzw

        if self.prev is None:
            jvel = np.zeros_like(q)
            lin = np.zeros_like(bpos)
            ang = np.zeros_like(bpos)
        else:
            pq, pp, pr = self.prev
            jvel = (q - pq) / self.dt
            lin = (bpos - pp) / self.dt
            ang = _ang_vel(pr, bquat, self.dt)
        self.prev = (q, bpos, bquat)

        payload = np.concatenate([
            [self.seq], q, jvel, bpos.ravel(), bquat.ravel(), lin.ravel(), ang.ravel(),
        ]).astype("<f4")
        self.sock.sendto(payload.tobytes(), self.addr)
        self.seq += 1
        return len(payload)


def _ang_vel(q0_xyzw, q1_xyzw, dt):
    """2 * vec(q1 * conj(q0)) / dt, with the double cover handled."""
    q0 = q0_xyzw[:, [3, 0, 1, 2]]
    q1 = q1_xyzw[:, [3, 0, 1, 2]]
    q1 = np.where((q0 * q1).sum(-1, keepdims=True) < 0, -q1, q1)
    w0, v0 = q0[:, :1], q0[:, 1:]
    w1, v1 = q1[:, :1], q1[:, 1:]
    return 2.0 * (w1 * (-v0) + w0 * v1 - np.cross(v1, -v0)) / dt


def frame_size(n_joints, n_bodies):
    return 1 + 2 * n_joints + n_bodies * (3 + 4 + 3 + 3)


def replay(clip_npz, addr=DEFAULT_ADDR, robot="k1", realtime=True):
    """Stream a saved clip at its own rate — the receiver cannot tell it from a camera."""
    d = np.load(clip_npz, allow_pickle=True)
    names = [str(n) for n in d["joint_names"]]
    q = np.asarray(d["joint_pos"])
    fps = float(np.asarray(d["fps"]).ravel()[0]) if "fps" in d.files else 50.0
    bp, bq = np.asarray(d["body_pos_w"]), np.asarray(d["body_quat_w"])

    pub = Publisher(robot, names, addr, fps)
    dt = 1.0 / fps
    t0 = time.time()
    for i in range(len(q)):
        root = np.concatenate([bp[i, 0], bq[i, 0]])
        pub.send(q[i], root)
        if realtime:
            nxt = t0 + (i + 1) * dt
            slack = nxt - time.time()
            if slack > 0:
                time.sleep(slack)
    return len(q), len(names), len(pub.body_names)


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--replay", metavar="NPZ", required=True,
                   help="stream a saved reference clip instead of a live take")
    p.add_argument("--robot", default="k1", choices=["g1", "k1"])
    p.add_argument("--host", default=DEFAULT_ADDR[0])
    p.add_argument("--port", type=int, default=DEFAULT_ADDR[1])
    p.add_argument("--fast", action="store_true", help="do not pace to real time")
    a = p.parse_args()
    n, nj, nb = replay(a.replay, (a.host, a.port), a.robot, realtime=not a.fast)
    print(f"sent {n} frames, {nj} joints x {nb} bodies "
          f"({frame_size(nj, nb) * 4} B/frame) -> {a.host}:{a.port}")


if __name__ == "__main__":
    main()
