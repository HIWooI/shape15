"""Look at what the student was trained on: SOMA skeleton beside the teacher's robot pose.

Numbers about label quality mislead — self-consistency is not correctness, and that
mistake cost this project a day. Render the pair and look at it.

    GEM-X/.venv/bin/python render_labels.py data/big4_k1.npz out.mp4 [--clips 5,16]
    GEM-X/.venv/bin/python render_labels.py data/synth_interp_k1.npz out.mp4 \
        --perception data/synth_interp.perception.npz
    GEM-X/.venv/bin/python render_labels.py data/legstatic_test_k1.npz out.mp4 \
        --models models/k1_retarget_parts.pt models/k1_retarget_tight.pt

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

_CWD = os.getcwd()
import demo_webcam as D  # chdirs into GEM-X, so path args must be resolved against _CWD
from gem.utils.kp2d_utils import PARENTS_77

def _p(path):
    return path if path is None or os.path.isabs(path) else os.path.join(_CWD, path)


MJCF = {   # inlined: importing ik_server here would pull in jaxlie, which the GEM venv lacks
    "g1": "/home/robotis-ai/.cache/newton/newton-assets_unitree_g1_308a72cd/"
          "unitree_g1/mjcf/g1_29dof_rev_1_0.xml",
    "k1": "/home/robotis-ai/Projects/shape15/GEM-X/third_party/soma-retargeter/"
          "soma_retargeter/configs/ai_sapiens/ai_sapiens_retarget.xml",
}

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
    p.add_argument("--models", nargs="*", metavar="CK",
                   help="append a pane per checkpoint, driven by the same body_pose")
    p.add_argument("--per_clip", type=int, default=180, help="frames per clip")
    p.add_argument("--fps", type=float, default=20.0)
    args = p.parse_args()

    args.labels, args.out = _p(args.labels), _p(args.out)
    lab = np.load(args.labels, allow_pickle=True)
    Y, clip, names = lab["teacher_q"], lab["clip"], [str(x) for x in lab["joint_names"]]
    rb = str(lab["robot"])
    bp = lab["body_pose"].reshape(len(Y), 76, 3)

    # identity/scale live in the perception file; without it the skeleton is a default body
    per = None
    for cand in ([_p(args.perception)] if args.perception else
                 [args.labels + ".perception.npz",
                  # newer sets name them <stem>_<robot>.npz / <stem>.perception.npz
                  args.labels.replace(f"_{rb}.npz", ".perception.npz"),
                  _p("data/big4_g1.npz.perception.npz")]):
        if cand and os.path.exists(cand):
            per = np.load(cand)
            break
    if per is None or len(per["p_ic_body_pose"]) != len(Y):
        sys.exit("need a matching perception npz for identity/scale (--perception)")

    # synthetic perception files only carry the p_* names; real ones also have p_ic_*
    def pk(k):
        return per[f"p_ic_{k}"] if f"p_ic_{k}" in per.files else per[f"p_{k}"]

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
            poses = torch.cat([torch.from_numpy(pk("global_orient")[s]).cuda()[:, None],
                               torch.from_numpy(bp[s]).cuda()], 1)
            soma._identity_frozen = False
            J[s0:s0 + 128] = model.body_model.static_forward(
                poses,
                torch.from_numpy(pk("identity_coeffs")[s]).cuda(),
                torch.from_numpy(pk("scale_params")[s]).cuda(),
                torch.from_numpy(pk("transl")[s]).cuda(),
                return_joints_only=True)["joints"].cpu().numpy()

    # students run on the same body_pose the teacher saw, so the panes are comparable
    student = []
    for ck_path in (args.models or []):
        ck = torch.load(_p(ck_path), map_location="cpu", weights_only=False)
        net = D.build_student(ck, len(names))
        if "state" in ck:
            net.load_state_dict(D._remap_block_keys(ck["state"]))
        net.eval()
        raw = str(ck.get("arch", "plain")) == "parts"   # per-part nets normalise internally
        mu = torch.tensor(np.asarray(ck["mu"])) if not raw else None
        sd = torch.tensor(np.asarray(ck["sd"])) if not raw else None
        with torch.no_grad():
            q = np.stack([net(t if raw else (t - mu) / sd).numpy()[:len(names)]
                          for t in torch.tensor(lab["body_pose"][sel], dtype=torch.float32)])
        student.append((os.path.basename(ck_path).replace(".pt", ""), q))

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

    titles = ["SOMA (human)", f"teacher {rb.upper()}"] + [n for n, _ in student]
    frames = []
    for i, fi in enumerate(sel):
        qs = [Y[fi]] + [q[i] for _, q in student]
        im = np.hstack([soma_image(J[i])] + [robot(q) for q in qs])
        for k, t in enumerate(titles):
            cv2.putText(im, t, (12 + 480 * k, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (235, 235, 235), 2, cv2.LINE_AA)
        # per-frame joint magnitudes make arm/leg coupling readable while watching
        for k, q in enumerate(qs):
            cv2.putText(im, f"arm {np.degrees(np.abs(q[ARM])).mean():5.1f}  "
                            f"leg {np.degrees(np.abs(q[LEG])).mean():5.1f} deg",
                        (12 + 480 * (k + 1), 466), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(im, f"clip {tags[i]}", (12, 466), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (200, 200, 200), 1, cv2.LINE_AA)
        frames.append(im)

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
