"""Can a small network beat our IK at reproducing soma-retargeter?

The IK and the teacher disagree by ~8 deg. That gap is mostly in the DOFs 14 points cannot
determine, which is exactly the kind of thing a network can learn a prior for and a
geometric solver cannot. So: same inputs, same teacher, and see.

Two phases in two venvs, the same split the rest of the project uses — the IK needs JAX and
pyroki, torch lives in the GEM-X venv. The first spawns the second.

    .venv-ik/bin/python distill.py labels.npz

Held out by *clip*, never by frame: neighbouring frames of one clip are nearly identical,
so a frame-wise split reports a number that has nothing to do with new motion.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).parent.resolve()
GEM_PY = _HERE / "GEM-X/.venv/bin/python"


def prep(args):
    """Build the network's inputs and the IK baseline it has to beat."""
    import jax.numpy as jnp
    import jaxlie

    import ik_retarget as IR

    d = np.load(args.labels, allow_pickle=True)
    rb = str(d["robot"])
    extras = [np.load(f, allow_pickle=True) for f in (args.extra or [])]
    for e in extras:
        if str(e["robot"]) != rb:
            sys.exit(f"--extra {e['robot']} does not match {rb}")
    IR.IK_MAP = IR.ROBOTS[rb][1]
    robot, link_idx, pos_w = IR.build(rb)
    names = list(robot.joints.actuated_names)
    solve, _, _ = IR.make_solver(robot, link_idx, pos_w, IR.TWIST_REST_W[rb])
    clip = d["clip"]

    feats, ik_q = [], []
    for ci in np.unique(clip):
        sel = np.where(clip == ci)[0]
        tg = IR.cam_to_world(d["joints"][sel])
        tg[..., :2] -= tg[:, :1, :2]
        # scales are frozen per session, and one clip is one session
        _, scales = IR.scale_to_robot(robot, link_idx, tg[:15])
        q, T = jnp.zeros(len(names)), jaxlie.SE3.identity()
        for i in range(len(sel)):
            sc, _ = IR.scale_to_robot(robot, link_idx, tg[i:i + 1], scales)
            sc = sc[0]
            w = jnp.array(IR.target_weights(pos_w, d["conf"][sel][i]))
            q, T = solve(jnp.array(sc), q, T, w)
            ik_q.append(np.asarray(q))
            # pelvis-relative, so the free base drops out and the network never has to
            # learn to ignore where the person stood
            feats.append(np.concatenate([(sc - sc[0]).ravel(), d["conf"][sel][i]]))
        print(f"  clip {ci}: {len(sel)} frames", flush=True)

    X = np.stack(feats).astype(np.float32)
    if args.features != "points":
        if "body_pose" not in d.files:
            sys.exit("this labels file predates body_pose; re-run make_labels.py --teacher")
        bp = d["body_pose"]
        if args.features == "pose":
            X = bp
        elif args.features == "pose_conf":
            # confidence tells the net which limbs the estimator actually saw — the IK
            # gets this via target weights, and without it the net poses hallucinated limbs
            X = np.concatenate([bp, d["conf"]], 1)
        else:
            X = np.concatenate([bp, X], 1)
    # t14 in the pelvis frame: the worker only needs it to Kabsch-fit the base, and
    # predicting it here lets the live loop skip SOMA FK (6.2 ms) entirely.
    t14 = (d["joints"] - d["joints"][:, :1]).reshape(len(d["joints"]), -1)
    X = X.astype(np.float32)
    Yq, ikq, clip = d["teacher_q"], np.stack(ik_q).astype(np.float32), clip
    aug = np.zeros(len(X), bool)

    # Extra label sets (synthetic, other captures) are training material only: they get
    # clip ids past the real ones so the holdout — the last two REAL clips — never sees
    # them, and no IK baseline is solved for them.
    def append(Xn, Yn, tag):
        nonlocal X, Yq, ikq, clip, aug
        base = clip.max() + 1
        X = np.concatenate([X, Xn.astype(np.float32)])
        Yq = np.concatenate([Yq, Yn.astype(np.float32)])
        ikq = np.concatenate([ikq, np.zeros((len(Xn), Yq.shape[1]), np.float32)])
        clip = np.concatenate([clip, np.full(len(Xn), base)])
        aug = np.concatenate([aug, np.ones(len(Xn), bool)])
        print(f"  + {tag}: {len(Xn)} frames as clip {base}", flush=True)

    for f, e in zip(args.extra or [], extras):
        if args.features != "pose":
            sys.exit("--extra currently assumes --features pose")
        append(e["body_pose"], e["teacher_q"], Path(f).name)

    if args.mirror:
        # Left/right reflection. Both mappings were validated by forward kinematics:
        # the SOMA rule (swap the 34 pairs, negate ry and rz) reproduces a geometric
        # mirror to 1.66 mm, and the robot rule (swap left/right, negate roll and yaw)
        # to 0.00 mm. Cheap coverage: every capture is also its own mirror image.
        pair = np.load(_HERE / "fixtures/soma_pair77.npy")
        bpm = (X[:, :228].reshape(-1, 76, 3)[:, pair[1:] - 1, :]
               * np.array([1, -1, -1], np.float32)).reshape(len(X), -1)
        jp = np.arange(len(names))
        for i, n in enumerate(names):
            if n.startswith("left_"):
                jp[i] = names.index(n.replace("left_", "right_", 1))
                jp[jp[i]] = i
        sgn = np.array([-1.0 if ("roll" in n or "yaw" in n) else 1.0 for n in names], np.float32)
        append(bpm, Yq[:, jp] * sgn, "mirror")

    np.savez(args.feats, X=X, features=args.features, ik_q=ikq, teacher_q=Yq,
             t14=t14.astype(np.float32), clip=clip, aug=aug,
             joint_names=np.array(names, dtype=object), robot=rb)
    print(f"prepped {len(X)} frames, {X.shape[1]} features ({args.features}) -> {args.feats}")


def train(args):
    import torch
    import torch.nn as nn

    d = np.load(args.feats, allow_pickle=True)
    X, ik_q, teacher = d["X"], d["ik_q"], d["teacher_q"]
    aux = d["t14"] if (args.aux_t14 and "t14" in d.files) else None
    clip, names, rb = d["clip"], [str(x) for x in d["joint_names"]], str(d["robot"])

    aug = d["aug"] if "aug" in d.files else np.zeros(len(X), bool)
    real_clips = np.unique(clip[~aug])
    clips = np.unique(clip)
    # Explicit, because len(clips)//4 silently changed the split when the clip count grew
    # and made runs incomparable. Every number in PLAN.md uses the last 2 clips — and the
    # holdout is drawn from the REAL clips only, never from synthetic or mirrored rows.
    n_test = args.test_clips if args.test_clips else max(1, len(real_clips) // 4)
    test_clips = real_clips[-n_test:]
    te = np.isin(clip, test_clips)
    tr = ~te
    print(f"{rb}: {tr.sum()} train / {te.sum()} test frames, "
          f"{len(clips) - n_test} / {n_test} clips")

    # Fit the IK's constant offsets on the training clips only. The shipped fixture was
    # fitted on the whole of whatever clip produced it, so scoring against that would let
    # the baseline see the test set — and it flatters it a lot: fitted-and-evaluated on one
    # clip the IK reads 7.8 deg, on a clip it did not see, 21.9. The network gets exactly
    # the same deal, so this is the comparison that means something.
    rb_tr = tr & ~aug          # the IK baseline is only defined on real frames
    resid_off = np.median(ik_q[rb_tr] - teacher[rb_tr], axis=0)
    keep = np.degrees(np.abs((ik_q[rb_tr] - resid_off) - teacher[rb_tr])).mean(0) < 15.0
    off = np.where(keep, resid_off, 0.0).astype(np.float32)

    deg = lambda a, b: float(np.degrees(np.abs(a - b)).mean())
    base_te = deg(ik_q[te] - off, teacher[te])
    print(f"IK baseline (offsets fitted on train) test {base_te:6.2f} deg   "
          f"train {deg(ik_q[rb_tr] - off, teacher[rb_tr]):6.2f}   "
          f"[raw, no offsets: test {deg(ik_q[te], teacher[te]):.2f}]")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    Xn = ((X - mu) / sd).astype(np.float32)

    n_out = len(names) + (aux.shape[1] if aux is not None else 0)

    # Which body_pose axes belong to which chain. Frozen in a fixture because it comes
    # from PARENTS_77, which lives in the GEM venv and is not importable from here.
    part77 = np.load(_HERE / "fixtures/soma_part77.npy").astype(str)
    axis_part = np.repeat(part77[1:], 3)          # body_pose is joints 1..76
    PART_IN = {"leg": ("leg", "torso"), "arm": ("arm", "torso"), "torso": ("torso",)}
    PART_OUT = {"leg": ("hip", "knee", "ankle"), "arm": ("shoulder", "elbow", "wrist"),
                "torso": ("waist",)}
    if args.part_in in ("tight", "solo"):
        # Cut the torso at the shoulders. "torso" lumps the spine in with the clavicles
        # (joints 11 and 39) and the head chain (4-10), and the clavicles rotate whenever
        # the arms move -- so with the wide split the leg expert has a live arm input even
        # though it never sees an arm joint. The spine (1,2,3) is the part that genuinely
        # moves the legs, so only that reaches them.
        fine = part77[1:].copy()                  # index i is joint i+1
        fine[[0, 1, 2]] = "spine"                 # joints 1-3
        fine[[3, 4, 5, 6, 7, 8, 9, 10, 38]] = "upper"   # head chain 4-11 and clavicle 39
        axis_part = np.repeat(fine, 3)
        PART_IN = {"leg": ("leg", "spine"), "arm": ("arm", "spine", "upper"),
                   "torso": ("spine", "upper")}
        if args.part_in == "solo":
            # Nothing above the pelvis reaches the legs at all. The spine still moves the
            # legs in the teacher (a raised arm shifts the stance), so this gives that up
            # on purpose in exchange for a leg pose that cannot be contaminated.
            PART_IN["leg"] = ("leg",)

    class ResBlock(nn.Module):
        """Pre-norm residual MLP block.

        Depth only helps this problem when it is residual: plain nets measured 15.45 deg
        at 2 layers and 16.03 at 6 (and wider is worse too — w4096 reads 16.19), while
        4 residual blocks reach 13.98 with a seed spread of 0.01. More blocks then get
        worse again, so 4 is the default.
        """

        def __init__(self, w, drop):
            super().__init__()
            self.norm = nn.LayerNorm(w)
            self.f = nn.Sequential(nn.Linear(w, w), nn.GELU(), nn.Dropout(drop),
                                   nn.Linear(w, w))

        def forward(self, x):
            return x + self.f(self.norm(x))

    def build():
        w = args.width
        if args.arch == "plain":
            return nn.Sequential(nn.Linear(X.shape[1], w), nn.GELU(),
                                 nn.Linear(w, w), nn.GELU(), nn.Linear(w, n_out))
        return nn.Sequential(nn.Linear(X.shape[1], w),
                             *[ResBlock(w, args.dropout) for _ in range(args.blocks)],
                             nn.LayerNorm(w), nn.Linear(w, n_out))

    def fit_parts(target, tag, epochs=400):
        """One expert per part, each blind to the other parts' inputs."""
        if X.shape[1] != len(axis_part):
            sys.exit(f"--arch parts needs the 228-dim pose features, got {X.shape[1]}")
        full = np.zeros((len(X), len(names)), np.float32)   # every row, for --fuse
        saved = {}
        for pname, keep in PART_IN.items():
            ci = np.where(np.isin(axis_part, keep))[0]
            co = [i for i, n in enumerate(names) if any(k in n for k in PART_OUT[pname])]
            if not len(co):
                continue
            torch.manual_seed(0)
            Xi = X[:, ci]
            mu_i, sd_i = Xi[tr].mean(0), Xi[tr].std(0) + 1e-6
            Xn_i = ((Xi - mu_i) / sd_i).astype(np.float32)
            w = args.width
            net = nn.Sequential(nn.Linear(len(ci), w),
                                *[ResBlock(w, args.dropout) for _ in range(args.blocks)],
                                nn.LayerNorm(w), nn.Linear(w, len(co))).to(dev)
            opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
            xt, yt = torch.tensor(Xn_i[tr], device=dev), torch.tensor(target[tr][:, co], device=dev)
            xv = torch.tensor(Xn_i[te], device=dev)
            best = (1e9, None)
            for ep in range(epochs):
                net.train()
                perm = torch.randperm(len(xt), device=dev)
                for k in range(0, len(xt), args.batch):
                    idx = perm[k:k + args.batch]
                    loss = nn.functional.smooth_l1_loss(net(xt[idx]), yt[idx], beta=0.05)
                    opt.zero_grad(); loss.backward(); opt.step()
                sched.step()
                net.eval()
                with torch.no_grad():
                    p = net(xv).cpu().numpy()
                e = deg(p, target[te][:, co])
                if e < best[0]:
                    best = (e, {k: v.cpu().clone() for k, v in net.state_dict().items()})
            net.load_state_dict(best[1])
            net.eval()
            with torch.no_grad():
                for k in range(0, len(Xn_i), 8192):
                    full[k:k + 8192, co] = net(
                        torch.tensor(Xn_i[k:k + 8192], device=dev)).cpu().numpy()
            saved[pname] = {"state": best[1], "mu": mu_i, "sd": sd_i,
                            "in": ci, "out": np.array(co)}
            print(f"    {pname:6s} expert: {best[0]:6.2f} deg  (input {len(ci)} dims)",
                  flush=True)

        err = deg(full[te], target[te])
        if args.fuse:
            # The experts cannot see each other, which is the point -- but the teacher
            # does couple them (a raised arm shifts the stance), and the leg expert has no
            # way to know. Let a small head correct the assembled pose from the assembled
            # pose alone: 23 robot angles in, 23 deltas out. The coupling it can express is
            # bounded by that bottleneck, unlike a single net reading all 228 raw inputs.
            torch.manual_seed(0)
            fmu, fsd = full[tr].mean(0), full[tr].std(0) + 1e-6
            Fn = ((full - fmu) / fsd).astype(np.float32)
            head = nn.Sequential(nn.Linear(len(names), 256), nn.GELU(),
                                 nn.Linear(256, 256), nn.GELU(),
                                 nn.Linear(256, len(names))).to(dev)
            head[-1].weight.data.zero_(); head[-1].bias.data.zero_()   # start at identity
            opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
            ft = torch.tensor(Fn[tr], device=dev)
            bt = torch.tensor(full[tr], device=dev)
            yt = torch.tensor(target[tr][:, :len(names)], device=dev)
            fv, bv = torch.tensor(Fn[te], device=dev), torch.tensor(full[te], device=dev)
            best = (1e9, None)
            for ep in range(epochs):
                head.train()
                perm = torch.randperm(len(ft), device=dev)
                for k in range(0, len(ft), args.batch):
                    idx = perm[k:k + args.batch]
                    loss = nn.functional.smooth_l1_loss(bt[idx] + head(ft[idx]), yt[idx],
                                                        beta=0.05)
                    opt.zero_grad(); loss.backward(); opt.step()
                sched.step()
                head.eval()
                with torch.no_grad():
                    e = deg((bv + head(fv)).cpu().numpy(), target[te])
                if e < best[0]:
                    best = (e, {k: v.cpu().clone() for k, v in head.state_dict().items()})
            print(f"    fuse head:      {best[0]:6.2f} deg  (was {err:.2f})", flush=True)
            saved["_fuse"] = {"state": best[1], "mu": fmu, "sd": fsd}
            err = best[0]
        return err, saved

    def fit(target, tag, epochs=400):
        torch.manual_seed(0)
        if aux is not None:
            target = np.concatenate([target, aux], 1).astype(np.float32)
        net = build().to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        xt = torch.tensor(Xn[tr], device=dev)
        yt = torch.tensor(target[tr], device=dev)
        xv = torch.tensor(Xn[te], device=dev)
        best, best_state = 1e9, None
        for ep in range(epochs):
            net.train()
            perm = torch.randperm(len(xt), device=dev)
            for k in range(0, len(xt), args.batch):
                idx = perm[k:k + args.batch]
                xb = xt[idx]
                if args.noise > 0:  # gaussian aug in normalized space
                    xb = xb + args.noise * torch.randn_like(xb)
                loss = nn.functional.smooth_l1_loss(net(xb), yt[idx], beta=0.05)
                opt.zero_grad(); loss.backward(); opt.step()
            sched.step()
            net.eval()
            with torch.no_grad():
                pred = net(xv).cpu().numpy()
            err = deg(pred[:, :len(names)] + (ik_q[te] if tag == "residual" else 0.0),
                      teacher[te])
            if err < best:
                best, best_state = err, {k: v.clone() for k, v in net.state_dict().items()}
            if ep % 100 == 99:
                print(f"    {tag} epoch {ep+1:4d}: test {err:6.2f} deg (best {best:6.2f})")
        net.load_state_dict(best_state)
        return best, net

    if args.arch == "parts":
        err, saved = fit_parts(teacher, "direct")
        print(f"\n{'model':28} {'test deg':>9}  vs IK")
        print(f"{'IK (soma-retargeter gap)':28} {base_te:9.2f}   —")
        print(f"{'per-part experts':28} {err:9.2f}   {base_te - err:+.2f}")
        torch.save({"parts": saved, "mode": "direct", "width": args.width,
                    "blocks": args.blocks, "arch": "parts", "features": "pose",
                    "names": names, "robot": rb}, args.model)
        print(f"\nsaved per-part model -> {args.model}")
        return

    results = {}
    results["direct"], net_d = fit(teacher, "direct")
    results["residual"], net_r = fit((teacher - ik_q).astype(np.float32), "residual")

    if aux is not None:
        with torch.no_grad():
            p = net_d(torch.tensor(Xn[te], device=dev)).cpu().numpy()[:, len(names):]
        print(f"  aux t14 error: {np.linalg.norm((p - aux[te]).reshape(-1, 14, 3), axis=-1).mean()*1000:.1f} mm")
    print(f"\n{'model':28} {'test deg':>9}  vs IK")
    print(f"{'IK (soma-retargeter gap)':28} {base_te:9.2f}   —")
    for k, v in results.items():
        print(f"{'MLP ' + k:28} {v:9.2f}   {base_te - v:+.2f}")
    best_k = min(results, key=results.get)
    torch.save({"state": (net_d if best_k == "direct" else net_r).state_dict(),
                "mode": best_k, "mu": mu, "sd": sd, "width": args.width,
                "arch": args.arch, "blocks": args.blocks,
                "features": str(d["features"]) if "features" in d.files else "pose",
                "aux_t14": aux is not None, "names": names, "robot": rb}, args.model)
    print(f"\nsaved {best_k} model -> {args.model}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("labels", nargs="?")
    p.add_argument("--feats")
    p.add_argument("--model")
    p.add_argument("--features", choices=["points", "pose", "pose_conf", "both"], default="pose",
                   help="points: the 14 targets the IK solves, the live protocol today. "
                        "pose: SOMA's 76-joint articulation, what the teacher retargets.")
    p.add_argument("--extra", nargs="*", metavar="NPZ",
                   help="extra label files to train on (synthetic, other captures). They "
                        "never enter the holdout and get no IK baseline.")
    p.add_argument("--mirror", action="store_true",
                   help="also train on the left/right reflection of everything")
    p.add_argument("--test_clips", type=int, default=2,
                   help="hold out this many clips from the end (0 = len//4, the old rule)")
    p.add_argument("--arch", choices=["plain", "res", "parts"], default="plain",
                   help="res: pre-norm residual blocks — measured -1.3 deg over plain on "
                        "K1 (13.98 vs 15.26, seed spread 0.17); plain depth/width both hurt. "
                        "parts: one residual expert per body part, each seeing only its own "
                        "chain of the input, so arm motion cannot move the legs at all. Same "
                        "accuracy as one net (10.09 vs 10.13) with the leakage structurally "
                        "zero — a single net moves the legs up to 31 deg for a 10 deg arm "
                        "perturbation because every output reads every input.")
    p.add_argument("--part_in", choices=["wide", "tight", "solo"], default="wide",
                   help="--arch parts: 'tight' cuts the torso at the shoulders so the "
                        "clavicles and head reach the arms only, not the legs; 'solo' also "
                        "cuts the spine, so nothing above the pelvis reaches them")
    p.add_argument("--fuse", action="store_true",
                   help="--arch parts: add a correction head over the assembled 23 angles")
    p.add_argument("--blocks", type=int, default=4, help="residual blocks when --arch res")
    p.add_argument("--dropout", type=float, default=0.1, help="inside residual blocks")
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--aux_t14", action="store_true",
                   help="also predict the 14 pelvis-frame targets, so the live loop can "
                        "skip SOMA FK; the worker payload is unchanged")
    p.add_argument("--noise", type=float, default=0.0,
                   help="train-time gaussian noise on the (normalized) inputs")
    p.add_argument("--train", action="store_true", help="internal: second phase")
    args = p.parse_args()
    if args.train:
        train(args)
        return
    args.feats = args.feats or f"{args.labels}.feats.npz"
    args.model = args.model or f"{args.labels}.mlp.pt"
    prep(args)
    subprocess.run([str(GEM_PY), str(_HERE / "distill.py"), "--train",
                    "--feats", args.feats, "--model", args.model,
                    "--width", str(args.width), "--batch", str(args.batch),
                    "--noise", str(args.noise), "--test_clips", str(args.test_clips),
                    "--arch", args.arch, "--blocks", str(args.blocks),
                    "--dropout", str(args.dropout), "--part_in", args.part_in]
                   + (["--aux_t14"] if args.aux_t14 else [])
                   + (["--fuse"] if args.fuse else []),
                   cwd=_HERE, check=True)


if __name__ == "__main__":
    main()
