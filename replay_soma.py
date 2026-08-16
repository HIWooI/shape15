"""Re-run the retarget stage on a saved take, without the camera or the GPU.

`--save_raw` keeps the SOMA skeleton the perception stage produced. That is exactly what
the worker consumes, so a take can be pushed back through `ik_server.py` byte for byte
and a change to the retarget can be judged against the same performance that motivated
it. Perception is not re-run and does not need to be: nothing upstream of the pipe moved.

    .venv-ik/bin/python replay_soma.py outputs/take4_raw/soma.npz outputs/take4b_mc.npz \
        --robot k1 --free_root

The frame timing comes from the take's own timestamps, so the 50 Hz resampling in
`MotionCommand` sees the gaps the live run saw, including the dropped frames. A
calibration request is injected on the first frame — the recording already begins where
the operator pressed Calibrate, and the worker needs the signal to measure bone scales.
"""

import argparse
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent.resolve()
NDOF = {"g1": 29, "k1": 23}


def replay(soma_npz, out_npz, robot="k1", extra=(), python=None, start=0.0, end=None):
    d = np.load(soma_npz, allow_pickle=True)
    j3d = np.asarray(d["joints3d"], "<f4")
    conf = np.asarray(d["conf"], "<f4")
    jrot = np.asarray(d["jrot"], "<f4")
    tq = np.asarray(d["target_q"], "<f4") if "target_q" in d.files else None
    t = np.asarray(d["t"], np.float64)

    # The raw log runs from process start, but a take begins where the operator pressed
    # Calibrate. Slicing here and calibrating on the first replayed frame reproduces the
    # live semantics, including measuring bone scales on the right person standing still.
    rel_all = t - t[0]
    keep = (rel_all >= start) & (rel_all <= (np.inf if end is None else end))
    j3d, conf, jrot, t = j3d[keep], conf[keep], jrot[keep], t[keep]
    if tq is not None:
        tq = tq[keep]
    n = len(j3d)
    if not n:
        raise ValueError(f"no frames in [{start}, {end}] of a {rel_all[-1]:.1f}s take")

    # relative seconds, not unix time: a float32 holds ~7 digits, which puts 1.7e9 at a
    # 128-second resolution and would collapse the whole take onto one instant
    rel = (t - t[0]).astype("<f4")

    cmd = [python or str(HERE / ".venv-ik/bin/python"), "ik_server.py",
           "--robot", robot, "--motion_command", str(out_npz), "--stamped", *extra]
    if tq is not None:
        cmd.append("--mlp")  # the take carries network angles; use the same path it did
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, cwd=HERE)
    proc.stdout.read(4)  # ready marker

    n_out = (1 + NDOF[robot]) * 4
    for i in range(n):
        c = conf[i].copy()
        if i == 0:
            c[0] = -abs(c[0])  # ask for calibration, the way the live loop does
        payload = j3d[i].tobytes() + c.tobytes() + jrot[i].tobytes()
        if tq is not None:
            payload += tq[i].tobytes()
        payload += rel[i].tobytes()
        proc.stdin.write(payload)
        proc.stdin.flush()
        # Read every reply rather than dropping like IKLink does: this is offline, so
        # there is no reason to throw away frames the live run only lost to latency.
        if len(proc.stdout.read(n_out)) < n_out:
            break
    proc.stdin.close()
    proc.wait(timeout=120)
    return n, float(t[-1] - t[0])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("soma", help="soma.npz from --save_raw")
    p.add_argument("out", help="motion_command npz to write")
    p.add_argument("--robot", default="k1", choices=["g1", "k1"])
    p.add_argument("--free_root", action="store_true",
                   help="keep the operator's horizontal motion (needed for a reference)")
    p.add_argument("--frames", action="store_true")
    p.add_argument("--start", type=float, default=0.0, metavar="SEC",
                   help="replay from here; calibration is injected on this frame")
    p.add_argument("--end", type=float, metavar="SEC")
    a = p.parse_args()
    extra = ([("--free_root")] if a.free_root else []) + (["--frames"] if a.frames else [])
    n, secs = replay(a.soma, a.out, a.robot, extra, start=a.start, end=a.end)
    print(f"{n} frames ({secs:.1f}s of take) -> {a.out}")


if __name__ == "__main__":
    main()
