"""Does the Calibrate button actually re-measure the operator?

Drives a real `ik_server.py` over the pipe — no camera, nobody in the room. The test: a
uniformly larger person, once calibrated, must come out as the *same robot pose*. Bone
scales are what absorb body size, so re-measuring them has to cancel the size change.
Without the request the worker keeps the first person's scales and the pose is far off,
which is the negative control that keeps this honest.

Not the symmetric rest pose. That one is the branch-degenerate case `test_apose.py`
reports 148 deg of asymmetry on, where a 1e-6 difference in the targets flips the solver
into a different branch and any result here is noise. A fixed asymmetric perturbation
makes the solve well-conditioned, which is what the live path actually sees.

    .venv-ik/bin/python test_calibrate.py [g1|k1]
"""
import subprocess
import sys

import numpy as np

ROBOT = sys.argv[1] if len(sys.argv) > 1 else "k1"
NDOF = {"g1": 29, "k1": 23}[ROBOT]
_REST = np.load("fixtures/soma_rest14.npy") * np.array([1.0, -1.0, 1.0], np.float32)
POSE = (_REST + np.random.default_rng(0).normal(0, 0.06, _REST.shape)).astype(np.float32)
SWITCH, TOTAL = 30, 120  # frames before the bigger person, and in total


def frame(pose, calibrate=False):
    conf = np.ones(14, "<f4")
    if calibrate:
        conf[0] = -conf[0]  # the sign bit is the request; demo_webcam.py sets it the same way
    return (pose.astype("<f4").tobytes() + conf.tobytes()
            + np.zeros((14, 3, 3), "<f4").tobytes())


def run(scale, calibrate):
    """Settle on a 1.0x person, then switch to `scale`x — with or without the request."""
    p = subprocess.Popen([".venv-ik/bin/python", "ik_server.py", "--robot", ROBOT],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p.stdout.read(4)  # ready marker
    for i in range(TOTAL):
        big = i >= SWITCH
        p.stdin.write(frame(POSE * (scale if big else 1.0), calibrate and i == SWITCH))
        p.stdin.flush()
        q = np.frombuffer(p.stdout.read((1 + NDOF) * 4), "<f4")[1:]
    p.stdin.close()
    resets = p.stderr.read().decode().count("re-measuring")
    p.wait()
    return q.copy(), resets


def link_positions(q):
    """Where the tracked links sit, in the pelvis frame.

    Joint angles are the wrong thing to compare: the twist DOFs are a nullspace, so two
    solves can put the robot in the same place through different angles, and G1 — with six
    more DOF than K1 — does exactly that (88 deg apart in `shoulder_yaw` and `waist_yaw`
    while standing identically). Task space is what calibration is answerable for. The
    pelvis frame drops the free base, which the worker does not report anyway.
    """
    import jax.numpy as jnp
    import jaxlie

    import ik_retarget as IR

    IR.IK_MAP = IR.ROBOTS[ROBOT][1]
    robot, link_idx, _ = IR.build(ROBOT)
    p = np.asarray(jaxlie.SE3(robot.forward_kinematics(jnp.array(q))[link_idx]).translation())
    return p - p[0]


def check_targets():
    """What calibration is actually answerable for: the targets it hands the solver.

    Exact and robot-independent, unlike anything downstream of the solve — bone scales
    absorb body size, so a re-measured 1.4x person must produce the same scaled skeleton
    as the 1.0x one, to float32. Where the solver then puts the robot is the solver's
    business, and on G1 it is not fully reproducible (see the bound below).
    """
    import ik_retarget as IR

    IR.IK_MAP = IR.ROBOTS[ROBOT][1]
    robot, link_idx, _ = IR.build(ROBOT)

    def scaled_for(s):
        pts = (POSE * s).reshape(1, 14, 3).astype(np.float32)
        tg = IR.cam_to_world(pts).copy()
        tg[..., :2] -= tg[:, :1, :2]
        stack = np.concatenate([tg] * 15)
        _, sc = IR.scale_to_robot(robot, link_idx, stack)
        fixed, _ = IR.scale_to_robot(robot, link_idx, stack, sc)
        out, _ = IR.scale_to_robot(robot, link_idx, tg, sc)
        out[..., 2] -= float(fixed[..., 2].min())
        return out[0]

    gap = float(np.abs(scaled_for(1.0) - scaled_for(1.4)).max())
    print(f"[{ROBOT}] re-measured targets vs the 1.0x reference: {gap:.2e} m")
    assert gap < 1e-5, f"re-measured scales do not reproduce the targets ({gap:.2e} m)"


if __name__ == "__main__":
    check_targets()
    base, _ = run(1.0, calibrate=False)
    recal, presses = run(1.4, calibrate=True)
    stale, unasked = run(1.4, calibrate=False)
    ref = link_positions(base)
    moved = lambda q: float(np.linalg.norm(link_positions(q) - ref, axis=-1).mean() * 1000)

    print(f"[{ROBOT}] 1.4x person vs the 1.0x reference, mean link displacement:")
    print(f"  after Calibrate  {moved(recal):7.1f} mm   ({presses} reset, requested)")
    print(f"  without it       {moved(stale):7.1f} mm   ({unasked} resets, none requested)")

    # K1 lands back on the reference exactly. G1 does not, and the bound below records
    # that as a known defect rather than a pass: with 29 DOF it has enough redundancy to
    # stay near wherever the size change pushed it, so re-measuring the scales recovers
    # the body size but not the exact pose. Tightening this is the open G1 item.
    LIMIT = {"k1": 15.0, "g1": 200.0}[ROBOT]  # g1: see the comment above, not a target
    assert presses == 1, f"one press must recalibrate exactly once, got {presses}"
    assert unasked == 0, f"nobody asked, but the worker recalibrated {unasked} times"
    assert moved(stale) > 40.0, f"negative control is dead: stale scales moved {moved(stale):.1f} mm"
    assert moved(recal) < LIMIT, f"recalibrated robot {moved(recal):.1f} mm off, limit {LIMIT}"
    print("ok" + ("  (g1 bound is a recorded defect, not a clean pass)" if ROBOT == "g1" else ""))
