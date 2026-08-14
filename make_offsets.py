"""Fit the per-joint constant offsets from a `make_labels.py` file.

SOMA and the robot disagree about where each joint's zero sits, and a constant absorbs
that. A constant is only *allowed* where it actually explains the joint, though — where
the residual after removing the median is still large, an offset would move the error
around rather than remove it, so that joint gets none.

Derived from streaming inputs and the robot's own teacher, so it matches what the live
worker sees. Re-run it whenever the solver changes: the offsets are fitted to a particular
solve, and the twist nullspace changed which joints a constant can explain.

    .venv-ik/bin/python make_offsets.py labels.npz
"""

import sys

import jax.numpy as jnp
import jaxlie
import numpy as np

import ik_retarget as IR

MAX_RESIDUAL_DEG = 15.0


def main():
    path = sys.argv[1]
    d = np.load(path, allow_pickle=True)
    rb = str(d["robot"])
    IR.IK_MAP = IR.ROBOTS[rb][1]
    robot, link_idx, pos_w = IR.build(rb)
    names = list(robot.joints.actuated_names)
    if names != [str(x) for x in d["joint_names"]]:
        sys.exit("joint names in the labels do not match the URDF order")
    solve, _, _ = IR.make_solver(robot, link_idx, pos_w, IR.TWIST_REST_W[rb])

    tg = IR.cam_to_world(d["joints"])
    tg[..., :2] -= tg[:, :1, :2]
    _, scales = IR.scale_to_robot(robot, link_idx, tg[:15])
    q, T, out = jnp.zeros(len(names)), jaxlie.SE3.identity(), []
    for i in range(len(tg)):
        sc, _ = IR.scale_to_robot(robot, link_idx, tg[i:i + 1], scales)
        w = jnp.array(IR.target_weights(pos_w, d["conf"][i]))
        q, T = solve(jnp.array(sc[0]), q, T, w)
        out.append(np.asarray(q))
    ours, teacher = np.stack(out), d["teacher_q"]

    off = np.median(ours - teacher, axis=0)
    resid = np.degrees(np.abs((ours - off) - teacher)).mean(0)
    keep = resid < MAX_RESIDUAL_DEG
    off = np.where(keep, off, 0.0)
    before = np.degrees(np.abs(ours - teacher)).mean()
    after = np.degrees(np.abs((ours - off) - teacher)).mean()

    print(f"{rb}: {len(tg)} frames, twist_w {IR.TWIST_REST_W[rb]}")
    print(f"  {before:.1f} deg -> {after:.1f} deg with offsets on {int(keep.sum())}/{len(names)}")
    for i in np.where(~keep)[0]:
        print(f"  no offset: {names[i]:32s} residual {resid[i]:5.1f} deg")
    np.savez(f"fixtures/{rb}_joint_offsets.npz", offsets=off.astype(np.float32),
             names=np.array(names, dtype=object))
    print(f"  wrote fixtures/{rb}_joint_offsets.npz")


if __name__ == "__main__":
    main()
