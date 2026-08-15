"""Synthesize SOMA pose trajectories and label them with the teacher — no video needed.

The teacher is a deterministic function of SOMA params, so training pairs do not have to
come from perception: sample the pose space directly and ask the teacher. This sidesteps
the diversity wall (13 real clips) that the distilled MLP hit — its error is almost
entirely generalization gap, and coverage is exactly what synthesis buys.

Not a grid: sweeping ~20 joints at 5 deg is 37^20 combinations. Instead, random keyframes
inside each joint's plausible range, cosine-interpolated into smooth ~30 fps trajectories
— the same coverage, linear cost, and the teacher (a motion pipeline with smoothing and a
feet stabiliser) gets motions, which is what it is built for.

Ranges come from the real dataset: per-axis empirical min/max of the observed body_pose,
widened by MARGIN. Joints that matter move a lot in real footage and get wide ranges;
fingers and the like stay near rest automatically.

    GEM-X/.venv/bin/python make_synth.py real_perception.npz --out synth --clips 24
    GEM-X/.venv/bin/python make_labels.py x --out synth_g1.npz --robot g1 \
        --mid synth.perception.npz --teacher
"""

import argparse

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("real", help="a make_labels perception npz, for ranges and identities")
    p.add_argument("--out", required=True, help="output prefix")
    p.add_argument("--clips", type=int, default=24)
    p.add_argument("--frames", type=int, default=600, help="per synthetic clip")
    p.add_argument("--keyframe_s", type=float, default=0.7,
                   help="seconds between pose keyframes (speed of the synthetic motion)")
    p.add_argument("--margin", type=float, default=1.5,
                   help="box mode: widen the observed per-axis range by this factor")
    p.add_argument("--mode", choices=["box", "interp"], default="box",
                   help="box: keyframes uniform in a per-axis range. interp: keyframes are "
                        "real poses (optionally blended with a neighbour), which stays on "
                        "the human manifold — box at margin 1.5 made 41%% of K1 labels "
                        "pathological because the teacher pushes impossible poses into "
                        "joint limits")
    p.add_argument("--blend", type=float, default=0.35,
                   help="interp mode: how far a keyframe may drift toward another real pose")
    p.add_argument("--bias", choices=["uniform", "extreme"], default="uniform",
                   help="interp mode: which real frames to draw keyframes from. extreme "
                        "oversamples the poses the student is worst at — hands above the "
                        "shoulders (5.3 deg -> 21.4), feet wide apart (5.1 -> 20.6), arms "
                        "fully extended (6.2 -> 19.7). Blending stays on the manifold, so "
                        "this reaches the hard region without inventing impossible poses")
    p.add_argument("--bias_power", type=float, default=3.0,
                   help="how hard to skew toward extreme frames (1 = mild, 3 = strong)")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    d = np.load(args.real)
    bp = d["p_ic_body_pose"] if "p_ic_body_pose" in d.files else d["p_body_pose"]
    bp = bp.reshape(len(bp), -1)  # (N, 228)
    go = d["p_global_orient"]
    rng = np.random.default_rng(args.seed)

    # plausible box per axis, widened around the observed centre
    lo, hi = bp.min(0), bp.max(0)
    mid, half = (lo + hi) / 2, (hi - lo) / 2
    lo, hi = mid - half * args.margin, mid + half * args.margin

    # identities: reuse real ones per clip, so the teacher's skeleton stays human-shaped
    ident_pool = np.unique(d["p_identity_coeffs"], axis=0)
    scale_pool = np.unique(d["p_scale_params"], axis=0)

    # per-frame sampling weight over the real poses
    pick = None
    if args.mode == "interp" and args.bias == "extreme":
        if "joints" not in d.files:
            raise SystemExit("--bias extreme needs the perception npz's `joints`")
        j = d["joints"]                     # (N,14,3) camera frame: x right, y down, z fwd
        sh = (j[:, 2] + j[:, 5]) / 2        # shoulders
        hand_up = (sh[:, 1] - (j[:, 4, 1] + j[:, 7, 1]) / 2)      # hands above shoulders
        foot_gap = np.linalg.norm(j[:, 10] - j[:, 13], axis=1)     # stance width
        arm_ext = (np.linalg.norm(j[:, 4] - j[:, 2], axis=1)
                   + np.linalg.norm(j[:, 7] - j[:, 5], axis=1)) / 2
        z = lambda v: (v - v.mean()) / (v.std() + 1e-9)
        score = np.maximum(z(hand_up), 0) + np.maximum(z(foot_gap), 0) + np.maximum(z(arm_ext), 0)
        w = (score + 0.05) ** args.bias_power
        pick = w / w.sum()
        top = np.argsort(-pick)[: len(pick) // 10]
        print(f"extreme bias: top 10% of frames carry {pick[top].sum() * 100:.0f}% of the weight")

    kf_gap = max(2, int(args.keyframe_s * args.fps))
    n = args.frames
    outs = {k: [] for k in ("p_body_pose", "p_global_orient", "p_transl",
                            "p_identity_coeffs", "p_scale_params")}
    clip_idx = []
    for ci in range(args.clips):
        nkf = n // kf_gap + 2
        if args.mode == "box":
            kf = rng.uniform(lo, hi, (nkf, bp.shape[1]))
        else:
            # keyframes are real poses nudged toward other real poses: novel combinations,
            # but every one is a blend of things a body actually did
            draw = (lambda k: rng.integers(len(bp), size=k)) if pick is None else \
                   (lambda k: rng.choice(len(bp), size=k, p=pick))
            a = bp[draw(nkf)]
            b = bp[draw(nkf)]
            w = rng.uniform(0.0, args.blend, (nkf, 1))
            kf = a * (1 - w) + b * w
        # cosine interpolation between keyframes: C1-smooth, no overshoot
        t = np.arange(n) / kf_gap
        i0 = t.astype(int)
        w = (1 - np.cos(np.pi * (t - i0))) / 2
        poses = kf[i0] * (1 - w[:, None]) + kf[i0 + 1] * w[:, None]
        # slow global yaw wander around the real clips' mean orientation
        base = go[rng.integers(len(go))]
        yaw = np.cumsum(rng.normal(0, 0.02, n))
        gori = np.repeat(base[None], n, 0).copy()
        gori[:, 1] += yaw  # y is up-ish in SOMA's world; close enough for coverage
        ident = ident_pool[rng.integers(len(ident_pool))]
        scale = scale_pool[rng.integers(len(scale_pool))]
        outs["p_body_pose"].append(poses.astype(np.float32))
        outs["p_global_orient"].append(gori.astype(np.float32))
        outs["p_transl"].append(np.zeros((n, 3), np.float32))
        outs["p_identity_coeffs"].append(np.repeat(ident[None], n, 0).astype(np.float32))
        outs["p_scale_params"].append(np.repeat(scale[None], n, 0).astype(np.float32))
        clip_idx += [ci] * n

    total = args.clips * n
    np.savez(f"{args.out}.perception.npz",
             **{k: np.concatenate(v) for k, v in outs.items()},
             # the incam copy the student trains on is the same articulation
             p_ic_body_pose=np.concatenate(outs["p_body_pose"]),
             joints=np.zeros((total, 14, 3), np.float32),   # teacher ignores these;
             conf=np.ones((total, 14), np.float32),          # kept for format compatibility
             kp2d=np.zeros((total, 0), np.float32),
             src=np.arange(total), clip=np.array(clip_idx), fps=np.float32(args.fps))
    print(f"wrote {total} synthetic frames over {args.clips} clips -> {args.out}.perception.npz")


if __name__ == "__main__":
    main()
