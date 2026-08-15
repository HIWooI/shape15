"""Replay a video through the live demo loop and render input | skeleton side by side.

The right panel shows what the real-time demo would have on screen at that instant:
frames the loop was too busy to grab are skipped and the last skeleton is held, so the
gap between the panels is the actual end-to-end delay.

    GEM-X/.venv/bin/python replay_delay.py demo.mov out.mp4
"""

import argparse
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

_CWD = Path.cwd()  # demo_webcam chdirs into GEM-X on import, so resolve paths first
import demo_webcam as D  # noqa: E402


from robot_target import Retargeter  # noqa: E402

# the 14 ik_map joints, as indices into SOMA's 77
SOMA_IDX = [i - 1 for i in (1, 4, 13, 14, 15, 41, 42, 43, 68, 69, 70, 73, 74, 75)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("video")
    p.add_argument("out")
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--min_window", type=int, default=16)
    p.add_argument("--height", type=int, default=540, help="output panel height")
    p.add_argument("--no_imgfeat", action="store_true", help="skip SAM-3D-Body conditioning")
    p.add_argument(
        "--retarget",
        choices=["g1", "k1"],
        help="also retarget to a robot, in chunks, and report the robot's own delay",
    )
    p.add_argument("--chunk", type=int, default=30, help="frames per retarget call (min 30)")
    p.add_argument("--ik", choices=["g1", "k1"], help="per-frame PyRoki IK in a worker process")
    p.add_argument("--ik_async", action="store_true",
                   help="talk to the worker through demo_webcam's IKLink (the live path): "
                        "never wait for a reply, drop the frame if it is behind")
    p.add_argument("--ik_render", type=int, metavar="PORT",
                   help="let the worker render MuJoCo and stream it, as it does live -- "
                        "that render is what stalls the reply")
    p.add_argument("--mlp", metavar="PT",
                   help="retarget with this distilled checkpoint instead of the IK solve, "
                        "as demo_webcam --mlp does")
    p.add_argument("--worker_profile", action="store_true",
                   help="have the worker print its own per-stage medians on exit")
    p.add_argument("--profile", action="store_true", help="break the loop down by stage")
    p.add_argument("--motion_command", help="write the 50 Hz policy reference to this .npz")
    p.add_argument("--soma_rot", action="store_true",
                   help="use SOMA's joint rotations as orientation targets (unverified: it "
                        "moves the twist joints a lot but has not been shown to be better)")
    args = p.parse_args()
    args.video, args.out = str(_CWD / args.video), str(_CWD / args.out)

    from gem.utils.vitpose_extractor import VitPoseExtractor

    vitpose = VitPoseExtractor(device="cuda:0", pose_type="soma", tqdm_leave=False)
    vitpose.flip_test = False
    model = D.build_model()
    worker = None
    if not args.no_imgfeat:
        from gem.utils.sam3db_extractor import SAM3DBExtractor

        worker = D.TokenWorker(SAM3DBExtractor(device="cuda:0"))

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    H, W = frames[0].shape[:2]
    K = D.estimate_K(W, H)
    full_bbx = D.get_bbx_xys_from_xyxy(torch.tensor([[0.0, 0.0, W - 1.0, H - 1.0]]))[0].float()

    retargeter = None
    if args.retarget:
        # The retarget needs global body params, which the demo stubs out for speed.
        import gem.pipeline.gem_pipeline as gp

        gp.get_body_params_w_Rt_v2 = gp.get_body_params_w_Rt_v2_full
        retargeter = Retargeter(args.retarget, fps=fps)

    mlp = None
    if args.mlp:
        ck = torch.load(str(_CWD / args.mlp), map_location="cpu", weights_only=False)
        mlp_net = D.build_student(ck, len(ck["names"]))
        mlp_net.load_state_dict(ck["state"])
        mlp_net.eval()
        mmu = torch.tensor(np.asarray(ck["mu"]), dtype=torch.float32)
        msd = torch.tensor(np.asarray(ck["sd"]), dtype=torch.float32)
        mlp = lambda bp: mlp_net((bp - mmu) / msd).numpy().astype("<f4")

    ik, ik_ndof, ik_dropped = None, 0, 0
    if args.ik:
        import subprocess

        ik_ndof = {"g1": 29, "k1": 23}[args.ik]
        cmd = ([".venv-ik/bin/python", "ik_server.py", "--robot", args.ik]
               + (["--motion_command", str(_CWD / args.motion_command)]
                  if args.motion_command else [])
               + (["--stream", str(args.ik_render)] if args.ik_render else [])
               + (["--mlp"] if args.mlp else [])
               + (["--profile"] if args.worker_profile else []))
        if args.ik_async:
            ik = D.IKLink(cmd, _CWD, ik_ndof)  # reads the ready marker itself
        else:
            ik = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, cwd=_CWD)
            ik.stdout.read(4)  # wait for jit warm-up before the clock starts

    # --- simulate the live loop against a wall clock -------------------------
    kp2d_win, bbx_win = deque(maxlen=args.window), deque(maxlen=args.window)
    tok_win = deque(maxlen=args.window)
    bbx, t = D.bootstrap_bbox(vitpose, frames[0], W, H), 0.0
    shots = []  # (time the result hits the screen, source frame index, drawable)
    ik_ms = []
    prof = {k: [] for k in ("vitpose", "bbox", "make_data", "predict", "denoiser_net",
                            "decode", "fk", "mlp", "ik", "rest")}
    # sub-timers inside model.predict: the denoiser network proper and the token decode.
    # (cuda-synced, so they cost a little; only the --profile numbers pay it)
    den, dec = model.pipeline.denoiser3d, model.pipeline.endecoder
    den_fwd, dec_fn = den.forward, dec.decode

    def timed_den(*a, **kw):
        torch.cuda.synchronize(); t = time.time()
        out = den_fwd(*a, **kw)
        torch.cuda.synchronize(); prof["denoiser_net"].append(time.time() - t)
        return out

    def timed_dec(*a, **kw):
        torch.cuda.synchronize(); t = time.time()
        out = dec_fn(*a, **kw)
        torch.cuda.synchronize(); prof["decode"].append(time.time() - t)
        return out
    den.forward, dec.decode = timed_den, timed_dec
    robot_shots = []  # same, for the retargeted robot pose
    pending = []  # global params awaiting a retarget chunk
    while True:
        src = int(t * fps)
        if src >= len(frames):
            break
        t0 = time.time()
        with torch.autocast("cuda", dtype=torch.float16):
            kp2d = vitpose.extract(frames[src][None], bbx[None], path_type="np", batch_size=1)[0]
        torch.cuda.synchronize(); ts = time.time(); prof["vitpose"].append(ts - t0)
        kp2d_win.append(kp2d.cuda())
        bbx_win.append(bbx.cuda())
        if worker is not None:
            token = worker.submit(frames[src], bbx)
            if token is not None:
                tok_win.append(token)
        nb = D.bbox_from_kps(kp2d, W, H)
        bbx = D.bootstrap_bbox(vitpose, frames[src], W, H) if nb is None else nb
        torch.cuda.synchronize(); prof["bbox"].append(time.time() - ts); ts = time.time()
        if nb is None:
            kp2d_win.clear()
            bbx_win.clear()
            tok_win.clear()
            if worker is not None:
                worker.reset()

        if len(kp2d_win) >= args.min_window:
            f_img = None
            if tok_win:
                f_img = torch.stack(
                    [tok_win[0]] * (len(kp2d_win) - len(tok_win)) + list(tok_win)
                )
            data = D.make_data(kp2d_win, bbx_win, K, f_img)
            torch.cuda.synchronize(); prof["make_data"].append(time.time() - ts); ts = time.time()
            pred = model.predict(data, static_cam=True, postproc=False)
            torch.cuda.synchronize(); prof["predict"].append(time.time() - ts); ts = time.time()
            xy, joints3d, jrot = D.project_last_frame(model, pred, K)
            torch.cuda.synchronize(); prof["fk"].append(time.time() - ts); ts = time.time()
            draw = (xy, None)
            if ik is not None:
                # in-camera joints are enough for a display; this skips the 35 ms
                # global-trajectory rollout entirely.
                t14 = joints3d[SOMA_IDX]  # camera frame; the IK worker converts
                conf = kp2d[SOMA_IDX, 2].numpy().astype("<f4")
                rot = (jrot if (args.soma_rot and jrot is not None)
                       else np.zeros((14, 3, 3), np.float32))
                payload = (t14.astype("<f4").tobytes() + conf.tobytes()
                           + rot.astype("<f4").tobytes())
                if mlp is not None:
                    tm = time.time()
                    with torch.no_grad():
                        bp = pred["body_params_incam"]["body_pose"][-1].reshape(-1).float().cpu()
                        payload += mlp(bp).tobytes()
                    prof["mlp"].append(time.time() - tm)
                if args.ik_async:
                    ik_dropped += not ik.submit(payload)
                    if ik.latest is not None:
                        ik_ms.append(np.frombuffer(ik.latest, "<f4")[0])
                else:
                    ik.stdin.write(payload)
                    ik.stdin.flush()
                    ik_ms.append(np.frombuffer(ik.stdout.read((1 + ik_ndof) * 4), "<f4")[0])
                prof["ik"].append(time.time() - ts); ts = time.time()
            if retargeter is not None:
                pending.append((src, {k: v[-1:] for k, v in pred["body_params_global"].items()}))
        else:  # warm-up: the live demo shows the raw 2D keypoints
            draw = (kp2d[:, :2].numpy(), kp2d[:, 2].numpy())

        # A chunk has piled up: retarget it. This blocks the loop, as it would live --
        # a thread cannot help here, see robot_target.py.
        chunk_ms = None
        if retargeter is not None and len(pending) >= args.chunk:
            newest_src, tc = pending[-1][0], time.time()
            retargeter.run([p for _, p in pending])
            pending.clear()
            chunk_ms = (time.time() - tc) * 1000

        torch.cuda.synchronize()
        prof["rest"].append(time.time() - ts)
        t += time.time() - t0
        shots.append((t, src, draw))
        if chunk_ms is not None:
            robot_shots.append((t, newest_src, chunk_ms))

    # --- render: left = live input, right = what the screen would show -------
    scale = args.height / H
    panel = (int(W * scale), args.height)
    writer = cv2.VideoWriter(
        args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (panel[0] * 2, panel[1])
    )
    lags, k = [], 0
    for i, frame in enumerate(frames):
        now = i / fps
        while k + 1 < len(shots) and shots[k + 1][0] <= now:
            k += 1
        right = frame.copy()
        if shots[k][0] <= now:
            xy, conf = shots[k][2]
            right = D.draw_skeleton(right, xy, conf)
            lag = now - shots[k][1] / fps
            lags.append(lag)
            cv2.putText(
                right, f"+{lag * 1000:.0f} ms", (20, 60), cv2.FONT_HERSHEY_SIMPLEX,
                1.6, (0, 255, 0), 3,
            )
        both = np.hstack([cv2.resize(frame, panel), cv2.resize(right, panel)])
        writer.write(both)
    writer.release()
    print(f"{len(shots)} inferences over {len(frames)} frames "
          f"({len(shots) / (len(frames) / fps):.1f} FPS effective)")
    p50, p90 = np.percentile(lags, [50, 90]) * 1000
    print(f"skeleton delay: median {p50:.0f} ms, p90 {p90:.0f} ms, "
          f"max {np.max(lags) * 1000:.0f} ms at {np.argmax(lags) / fps:.1f}s")
    # How long the loop went without producing a frame. The delay above is what a viewer
    # sees; this is what freezes the panel, and it is what a blocking IK reply inflates.
    gaps = np.diff([s[0] for s in shots]) * 1000
    print(f"frame gap: median {np.median(gaps):.0f} ms, p99 {np.percentile(gaps, 99):.0f} ms, "
          f"max {gaps.max():.0f} ms at {shots[int(np.argmax(gaps)) + 1][0]:.1f}s"
          + (f" | dropped {ik_dropped}/{len(shots)} IK frames (worker behind)"
             if args.ik_async else ""))

    if retargeter is not None and not robot_shots:
        print(f"no retarget finished (chunk is {args.chunk} frames)")
    if robot_shots:
        # a robot pose from source frame s becomes visible at time t -> delay t - s/fps
        rl = np.array([t_done - s / fps for t_done, s, _ in robot_shots])
        # the first chunk pays warp's per-shape kernel specialisation; a live system
        # would pay that at startup, so report it apart from the steady state
        ct = np.array([c for _, _, c in robot_shots])
        print(f"{args.retarget} robot delay, {len(rl)} chunks of {args.chunk}: "
              f"first {rl[0] * 1000:.0f} ms | "
              + (f"steady median {np.median(rl[1:]) * 1000:.0f} ms, "
                 f"min {rl[1:].min() * 1000:.0f}, max {rl[1:].max() * 1000:.0f}"
                 if len(rl) > 1 else "no steady-state chunk"))
        print(f"  retarget compute per chunk: first {ct[0]:.0f} ms, "
              f"then median {np.median(ct[1:]):.0f} ms" if len(ct) > 1 else "")


    if args.profile:
        print("stage breakdown (median ms over the loop):")
        tot = 0.0
        for k, v in prof.items():
            if v:
                m = np.median(v[3:]) * 1000
                tot += m
                print(f"  {k:10s} {m:6.1f} ms  (n={len(v)})")
        print(f"  {'TOTAL':10s} {tot:6.1f} ms  -> {1000 / tot:.1f} FPS if nothing else intervened")
    if ik_ms:
        m = np.array(ik_ms[1:])
        print(f"{args.ik} IK in-loop: median {np.median(m):.1f} ms, p90 "
              f"{np.percentile(m, 90):.1f} ms -> robot target delay = skeleton + this")
    if ik is not None:
        if args.ik_async:
            ik.close()
        else:
            ik.stdin.close(); ik.wait()


if __name__ == "__main__":
    main()
