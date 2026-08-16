"""Render a saved policy reference in MuJoCo, so a motion_command file can be eyeballed.

    .venv-ik/bin/python render_reference.py ref.npz out.mp4 --robot k1
"""

import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio
import mujoco
import numpy as np

from ik_server import MJCF


def main():
    ref_path, out_path = sys.argv[1], sys.argv[2]
    robot = sys.argv[sys.argv.index("--robot") + 1] if "--robot" in sys.argv else "k1"
    d = np.load(ref_path, allow_pickle=True)
    q, names = d["joint_pos"], [str(n) for n in d["joint_names"]]

    # a floor, so the eye can judge whether the reference stands on the ground; the
    # retargeting MJCF has none, and foot penetration is the defect this is used to see
    spec = mujoco.MjSpec.from_file(MJCF[robot])
    g = spec.worldbody.add_geom()
    g.name, g.type = "floor", mujoco.mjtGeom.mjGEOM_PLANE
    g.size, g.rgba = [0.0, 0.0, 0.05], [0.30, 0.34, 0.40, 1.0]
    g.contype = g.conaffinity = 0  # display only; nothing is stepped here
    model = spec.compile()
    data = mujoco.MjData(model)
    qadr = []
    for n in names:  # map by name, never by order
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
        qadr.append(model.jnt_qposadr[jid] if jid >= 0 else -1)
    missing = [n for n, a in zip(names, qadr) if a < 0]
    if missing:
        print(f"not in MJCF, left at rest: {missing}")

    renderer = mujoco.Renderer(model, 480, 640)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.distance, cam.elevation, cam.azimuth = 3.0, -10, 135
    cam.lookat[:] = [0, 0, 0.8]
    cam.distance = 3.5

    # an exported clip carries the root the sim was checked against; a raw capture may
    # not, and then the robot is stood upright at a plausible height instead
    if "root" in d.files:
        root = np.asarray(d["root"], np.float64)
    elif "body_pos_w" in d.files:
        root = np.concatenate([np.asarray(d["body_pos_w"])[:, 0],
                               np.asarray(d["body_quat_w"])[:, 0]], axis=1)
    else:
        root = np.tile([0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 1.0], (len(q), 1))

    frames = []
    for i, row in enumerate(q):
        if model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE:
            data.qpos[0:3] = root[i, 0:3]
            data.qpos[3:7] = root[i, [6, 3, 4, 5]]  # file xyzw -> MuJoCo wxyz
        for a, v in zip(qadr, row):
            if a >= 0:
                data.qpos[a] = v
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, cam)
        frames.append(renderer.render())
    imageio.mimsave(out_path, frames, fps=float(d["rate"] if "rate" in d.files else np.asarray(d["fps"]).ravel()[0]), macro_block_size=1)
    print(f"wrote {len(frames)} frames to {out_path}")


if __name__ == "__main__":
    main()
