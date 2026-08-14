"""Generate (student input, teacher output) pairs for distilling soma-retargeter.

The teacher is slow but offline, so its cost does not matter — 29 ms/frame for G1 against
1179 ms for K1, which is why G1 is the robot to validate the idea on. What it produces is
the thing our IK is scored against, so these pairs are worth having even if no network is
ever trained on them.

Inputs are collected the *streaming* way — sliding window, last frame only, image features
from whatever token has arrived — because that is the distribution the student would see
live. Labels come from the teacher run over the same frames. Both therefore sit on the
same underlying pose estimate, so a student learns the retargeting map and not the gap
between offline and streaming perception.

    GEM-X/.venv/bin/python make_labels.py a.mp4 b.mp4 --out labels.npz --robot g1

Runs the teacher in a second process on purpose: soma-retargeter captures a CUDA graph and
torch on the legacy stream meanwhile raises `cudaErrorStreamCaptureImplicit`, and the
SAM-3D-Body worker thread is exactly that trigger. A fresh process has no such thread.
"""

import argparse
import subprocess
import sys
from collections import deque
from pathlib import Path

import numpy as np

_CWD = Path.cwd()
_HERE = Path(__file__).parent.resolve()

CSV_CONFIG = {"g1": "UnitreeG129DOF_CSVConfig", "k1": "AISapiens23DOF_CSVConfig"}


def perception(args):
    """Stream the clip through the live loop; save student inputs and the teacher's input."""
    import cv2
    import torch

    import demo_webcam as D
    import gem.pipeline.gem_pipeline as gp
    from gem.utils.vitpose_extractor import VitPoseExtractor

    # the teacher needs world-frame body params, which the live demo stubs out for speed
    gp.get_body_params_w_Rt_v2 = gp.get_body_params_w_Rt_v2_full

    vitpose = VitPoseExtractor(device="cuda:0", pose_type="soma", tqdm_leave=False)
    vitpose.flip_test = False
    model = D.build_model()
    worker = None
    if not args.no_imgfeat:
        from gem.utils.sam3db_extractor import SAM3DBExtractor

        worker = D.TokenWorker(SAM3DBExtractor(device="cuda:0"))

    joints, confs, params, src_idx, clip_idx, kp2ds = [], [], [], [], [], []
    fps = None
    for ci, video in enumerate(args.video):
        cap = cv2.VideoCapture(video)
        fps = fps or (cap.get(cv2.CAP_PROP_FPS) or 30.0)
        frames = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
        cap.release()
        if not frames:
            print(f"  skip (unreadable): {video}", flush=True)
            continue
        if args.limit:
            frames = frames[: args.limit]
        H, W = frames[0].shape[:2]
        K = D.estimate_K(W, H)
        n0 = len(joints)
        _one_clip(args, D, torch, vitpose, model, worker, frames, K, W, H,
                  joints, confs, params, src_idx, kp2ds)
        clip_idx += [ci] * (len(joints) - n0)
        print(f"  [{ci + 1}/{len(args.video)}] {len(joints) - n0}/{len(frames)} frames "
              f"from {Path(video).name}", flush=True)

    if not joints:
        sys.exit("no frames survived the warm-up window — is anyone in these clips?")
    out = {"joints": np.stack(joints), "conf": np.stack(confs),
           "kp2d": np.stack(kp2ds),
           "src": np.array(src_idx), "clip": np.array(clip_idx), "fps": np.float32(fps)}
    for k in params[0]:
        out[f"p_{k}"] = np.concatenate([p[k].numpy() for p in params])
    np.savez(args.mid, **out)
    print(f"perception: {len(joints)} frames over {len(args.video)} clips -> {args.mid}",
          flush=True)


def _one_clip(args, D, torch, vitpose, model, worker, frames, K, W, H,
              joints, confs, params, src_idx, kp2ds):
    """One clip through the streaming loop; appends to the shared lists."""
    kp2d_win, bbx_win, tok_win = (deque(maxlen=args.window) for _ in range(3))
    bbx = D.bootstrap_bbox(vitpose, frames[0], W, H)
    if worker is not None:
        worker.reset()
    for src, frame in enumerate(frames):
        with torch.autocast("cuda", dtype=torch.float16):
            kp2d = vitpose.extract(frame[None], bbx[None], path_type="np", batch_size=1)[0]
        kp2d_win.append(kp2d.cuda())
        bbx_win.append(bbx.cuda())
        if worker is not None:
            token = worker.submit(frame, bbx)
            if token is not None:
                tok_win.append(token)
        nb = D.bbox_from_kps(kp2d, W, H)
        bbx = D.bootstrap_bbox(vitpose, frame, W, H) if nb is None else nb
        if nb is None:  # track lost: the window is meaningless, start it over
            kp2d_win.clear()
            bbx_win.clear()
            tok_win.clear()
            if worker is not None:
                worker.reset()
            continue
        if len(kp2d_win) < args.min_window:
            continue

        f_img = None
        if tok_win:
            f_img = torch.stack([tok_win[0]] * (len(kp2d_win) - len(tok_win)) + list(tok_win))
        pred = model.predict(D.make_data(kp2d_win, bbx_win, K, f_img),
                             static_cam=True, postproc=False)
        _, joints3d, _ = D.project_last_frame(model, pred, K)
        joints.append(joints3d[D.SOMA_IDX].astype(np.float32))
        confs.append(kp2d[D.SOMA_IDX, 2].numpy().astype(np.float32))
        kp2ds.append(kp2d.numpy().astype(np.float32))  # all 77, for stage-cut students
        # global params feed the teacher; incam params are what the live path computes
        # its joint rotations from (the global ones go through GEM's world rollout, which
        # re-optimises the body — 40-66 mm and 5-18 deg away from incam, measured)
        params.append({**{k: v[-1:].cpu() for k, v in pred["body_params_global"].items()},
                       **{"ic_" + k: v[-1:].cpu()
                          for k, v in pred["body_params_incam"].items()}})
        src_idx.append(src)


def teacher(args):
    """Run soma-retargeter over the saved params and pull its joint angles out by name."""
    import os

    # `robot_target` reaches for `scripts.demo`, which lives in the GEM-X tree; the other
    # entry points get this for free by importing demo_webcam, which we deliberately do not.
    sys.path.insert(0, str(_HERE / "GEM-X"))
    os.chdir(_HERE / "GEM-X")

    import torch

    from robot_target import Retargeter

    import soma_retargeter.assets.csv as SC

    d = np.load(args.mid)
    keys = [k[2:] for k in d.files if k.startswith("p_") and not k.startswith("p_ic_")]
    n = len(d["joints"])
    clip = d["clip"] if "clip" in d.files else np.zeros(n, int)
    cfg = getattr(SC, CSV_CONFIG[args.robot])()
    names = [h[:-4] for h in cfg.csv_header if h.endswith("_dof")]
    cols = [cfg.csv_header.index(f"{nm}_dof") for nm in names]

    if args.teacher_clip >= 0:
        # one clip in this process, then exit — see below for why
        ci = args.teacher_clip
        sel = np.where(clip == ci)[0]
        params = [{k: torch.from_numpy(d[f"p_{k}"][i:i + 1]) for k in keys} for i in sel]
        # Per clip, never concatenated: the teacher retargets a *motion*, with a feet
        # stabiliser and a smoothing filter that run along it. Splicing two clips together
        # would have it smooth across the cut and invent frames that belong to neither.
        rt = Retargeter(args.robot, fps=float(d["fps"]))
        buf = rt.run(params)
        rows = np.array([cfg.to_csv_row(i, buf.get_data(i)) for i in range(buf.num_frames)],
                        np.float32)
        q = rows[:, cols]
        q = np.radians(q) if np.abs(q).max() > 7 else q
        # Row i of the buffer must be frame i of the input. It is — the pipeline prepends
        # its own 10 initialisation and 5 stabilisation frames (the progress bar counts
        # n+15) but does not emit them. Assert rather than truncate: quietly keeping the
        # first n rows of a misaligned buffer pairs every input with the wrong label and
        # still trains perfectly happily.
        if len(q) != len(sel):
            sys.exit(f"clip {ci}: teacher returned {len(q)} rows for {len(sel)} frames — "
                     f"labels would be misaligned; do not train on this")
        np.save(f"{args.mid}.teacher_{args.robot}{ci}.npy", q.astype(np.float32))
        print(f"  clip {ci}: {len(q)} pairs", flush=True)
        return

    # One subprocess per clip. The retarget pipeline does not free its buffers when the
    # Retargeter is dropped, so building 18 of them in one process accumulated 48 GB of
    # RSS and the kernel OOM-killed the run at the last clip. A process exit frees
    # everything, and the per-clip .npy doubles as a resume point.
    qs = []
    for ci in np.unique(clip):
        # Robot-specific resume points: G1 and K1 label the same perception file, and a
        # robot-less name would make the K1 run silently adopt the G1 partials.
        part = Path(f"{args.mid}.teacher_{args.robot}{ci}.npy")
        if not part.exists():
            subprocess.run([sys.executable, str(_HERE / "make_labels.py"), "x",
                            "--out", args.out, "--robot", args.robot, "--mid", args.mid,
                            "--teacher", "--teacher_clip", str(ci)],
                           cwd=_HERE / "GEM-X", check=True)
        qs.append(np.load(part))

    # Carry the SOMA articulation through as well. It is what the teacher actually
    # retargets, and a student given only our 14 points is guessing at what this states
    # outright — 17.0 deg against 9.5 deg on the same split. The *incam* one: it is what
    # the live loop computes every frame, and it trains identically to the global one
    # (9.48 deg either way).
    bp_key = "p_ic_body_pose" if "p_ic_body_pose" in d.files else "p_body_pose"
    np.savez(args.out, joints=d["joints"], conf=d["conf"], src=d["src"], clip=clip,
             kp2d=d["kp2d"] if "kp2d" in d.files else np.zeros((n, 0), np.float32),
             body_pose=d[bp_key].reshape(n, -1).astype(np.float32),
             teacher_q=np.concatenate(qs).astype(np.float32),
             joint_names=np.array(names, dtype=object), robot=args.robot, fps=d["fps"])
    print(f"wrote {n} pairs to {args.out}  ({len(names)} DOF, {args.robot})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("video", nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--robot", choices=["g1", "k1"], default="g1")
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--min_window", type=int, default=16)
    p.add_argument("--limit", type=int, help="stop after this many source frames")
    p.add_argument("--no_imgfeat", action="store_true")
    p.add_argument("--mid", help="intermediate file (default: <out>.perception.npz)")
    p.add_argument("--teacher", action="store_true", help="internal: second phase")
    p.add_argument("--teacher_clip", type=int, default=-1,
                   help="internal: retarget only this clip index and exit")
    args = p.parse_args()
    args.video = [str(_CWD / v) for v in args.video]
    args.out = str(_CWD / args.out)
    args.mid = args.mid or f"{args.out}.perception.npz"

    if args.teacher:
        teacher(args)
        return
    perception(args)
    # second process: see the module docstring for why a thread will not do
    subprocess.run([sys.executable, str(_HERE / "make_labels.py"), *args.video,
                    "--out", args.out, "--robot", args.robot, "--mid", args.mid, "--teacher"],
                   cwd=_HERE, check=True)


if __name__ == "__main__":
    main()
