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

    model = mujoco.MjModel.from_xml_path(MJCF[robot])
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

    frames = []
    for row in q:
        if model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE:
            data.qpos[0:3] = [0.0, 0.0, 0.9]  # the reference carries no root position
            data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        for a, v in zip(qadr, row):
            if a >= 0:
                data.qpos[a] = v
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, cam)
        frames.append(renderer.render())
    imageio.mimsave(out_path, frames, fps=float(d["rate"]), macro_block_size=1)
    print(f"wrote {len(frames)} frames to {out_path}")


if __name__ == "__main__":
    main()
