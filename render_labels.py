"""Look at what the student was trained on: SOMA skeleton beside the teacher's robot pose.

Numbers about label quality mislead — self-consistency is not correctness, and that
mistake cost this project a day. Render the pair and look at it.

    GEM-X/.venv/bin/python render_labels.py data/big4_k1.npz out.mp4 [--clips 5,16]
    GEM-X/.venv/bin/python render_labels.py data/synth_interp_k1.npz out.mp4 \
        --perception data/synth_interp.perception.npz

The labels npz carries `body_pose`; SOMA FK turns that back into a 77-joint skeleton, so
synthetic clips (which have no video) render exactly like real ones.
"""

import argparse
import os
import subprocess
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, "/home/robotis-ai/Projects/shape15")

import cv2
import mujoco
import numpy as np
import torch

import demo_webcam as D  # chdirs into GEM-X
from gem.utils.kp2d_utils import PARENTS_77
from ik_server import MJCF

BONES = [(p, c) for c, p in enumerate(PARENTS_77) if p >= 0]


def soma_image(j3d, size=480, label=""):
    """Front view. Camera frame is x-right / y-down, so x,y project directly."""
    img = np.full((size, size, 3), 24, np.uint8)
    xy = j3d[:, :2] - j3d[:, :2].mean(0)
    sc = size * 0.40 / (np.abs(xy).max() + 1e-6)
    px = (xy * sc + size / 2).astype(int)
    d = j3d[:, 2]
    near, far = d.min(), d.max() + 1e-6
    for p, c in BONES:
        t = 1 - (d[[p, c]].mean() - near) / (far - near)   # nearer = brighter
        cv2.line(img, tuple(px[p]), tuple(px[c]),
                 (int(90 + 130 * t), int(150 + 90 * t), int(90 + 60 * t)), 2, cv2.LINE_AA)
    for k in range(77):
        cv2.circle(img, tuple(px[k]), 3, (235, 235, 235), -1, cv2.LINE_AA)
    return img


def main():
    p = argparse.ArgumentParser()
    p.add_argument("labels")
    p.add_argument("out")
    p.add_argument("--perception", help="defaults to <labels>.perception.npz's usual name")
    p.add_argument("--clips", help="comma-separated clip ids (default: a spread of them)")
    p.add_argument("--per_clip", type=int, default=180, help="frames per clip")
    p.add_argument("--fps", type=float, default=20.0)
    args = p.parse_args()

    lab = np.load(args.labels, allow_pickle=True)
    Y, clip, names = lab["teacher_q"], lab["clip"], [str(x) for x in lab["joint_names"]]
    rb = str(lab["robot"])
    bp = lab["body_pose"].reshape(len(Y), 76, 3)

    # identity/scale live in the perception file; without it the skeleton is a default body
    per = None
    for cand in ([args.perception] if args.perception else
                 [args.labels + ".perception.npz",
                  "data/big4_g1.npz.perception.npz"]):
        if cand and os.path.exists(cand):
            per = np.load(cand)
            break
    if per is None or len(per["p_ic_body_pose"]) != len(Y):
        sys.exit("need a matching perception npz for identity/scale (--perception)")

    cl = np.unique(clip)
    want = [int(c) for c in args.clips.split(",")] if args.clips else list(cl[:: max(1, len(cl) // 4)])[:4]
    sel = np.concatenate([np.where(clip == c)[0][: args.per_clip] for c in want])
    tags = sum([[int(c)] * min(args.per_clip, int((clip == c).sum())) for c in want], [])
    print(f"{args.labels}: clips {want}, {len(sel)} frames", flush=True)

    model = D.build_model()
    soma = model.body_model.soma if hasattr(model.body_model, "soma") else model.body_model
    J = np.empty((len(sel), 77, 3), np.float32)
    with torch.no_grad():
        for s0 in range(0, len(sel), 128):
            s = sel[s0:s0 + 128]
            poses = torch.cat([torch.from_numpy(per["p_ic_global_orient"][s]).cuda()[:, None],
                               torch.from_numpy(bp[s]).cuda()], 1)
            soma._identity_frozen = False
            J[s0:s0 + 128] = model.body_model.static_forward(
                poses,
                torch.from_numpy(per["p_ic_identity_coeffs"][s]).cuda(),
                torch.from_numpy(per["p_ic_scale_params"][s]).cuda(),
                torch.from_numpy(per["p_ic_transl"][s]).cuda(),
                return_joints_only=True)["joints"].cpu().numpy()

    mj = mujoco.MjModel.from_xml_path(MJCF[rb])
    data = mujoco.MjData(mj)
    qadr = [mj.jnt_qposadr[mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in names]
    rend = mujoco.Renderer(mj, 480, 480)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.distance, cam.elevation, cam.azimuth = 3.2, -8, 150
    cam.lookat[:] = [0, 0, 0.8]

    LEG = [i for i, n in enumerate(names) if any(k in n for k in ("hip", "knee", "ankle"))]
    ARM = [i for i, n in enumerate(names) if any(k in n for k in ("shoulder", "elbow", "wrist"))]

    def robot(q):
        data.qpos[0:3] = [0, 0, 0.9]
        data.qpos[3:7] = [1, 0, 0, 0]
        for a, v in zip(qadr, q):
            data.qpos[a] = v
        mujoco.mj_forward(mj, data)
        rend.update_scene(data, cam)
        return rend.render()[..., ::-1]

    frames = []
    for i, fi in enumerate(sel):
        both = np.hstack([soma_image(J[i]), robot(Y[fi])])
        cv2.putText(both, f"SOMA (human)  |  teacher {rb.upper()}", (12, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (235, 235, 235), 2, cv2.LINE_AA)
        # per-frame joint magnitudes make arm/leg coupling readable while watching
        cv2.putText(both, f"clip {tags[i]}   arm {np.degrees(np.abs(Y[fi][ARM])).mean():5.1f}deg"
                          f"   leg {np.degrees(np.abs(Y[fi][LEG])).mean():5.1f}deg", (12, 466),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        frames.append(both)

    # mp4v then ffmpeg: this OpenCV has no avc1 encoder and writes an empty file silently
    tmp = args.out.replace(".mp4", "_raw.mp4")
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
    for f in frames:
        vw.write(f)
    vw.release()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart",
                    args.out], check=True)
    os.remove(tmp)
    print(f"wrote {len(frames)} frames -> {args.out}")


if __name__ == "__main__":
    main()
