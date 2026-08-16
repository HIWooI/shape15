"""Cut the dead ends off a motion_command capture.

Every live recording has them: the operator holds still after Calibrate, and walks
back to the keyboard before it stops. Neither is motion worth asking a policy to
track, and the walk-back in particular is a reference the robot has no reason to
follow. Slicing here rather than at export keeps the raw capture intact.

    .venv-ik/bin/python trim_clip.py outputs/take3_mc.npz outputs/take3_cut.npz --end 940
    .venv-ik/bin/python trim_clip.py outputs/take3_mc.npz outputs/take3_cut.npz --auto

`--auto` keeps the span between the first and last frame whose mean joint speed
crosses a threshold, padded a little so a gesture is not clipped mid-swing. Print
the speed profile first (`--show`) before trusting it: "still" and "tracking lost"
look identical in the speed alone.
"""

import argparse

import numpy as np


def profile(npz, bucket=50):
    """Mean joint speed per bucket of frames, for eyeballing where the motion is."""
    sp = np.abs(np.asarray(npz["joint_vel"])).mean(1)
    return [(s / 50.0, sp[s:s + bucket].mean()) for s in range(0, len(sp), bucket)]


def bounds(npz, thresh=0.15, pad=25):
    sp = np.abs(np.asarray(npz["joint_vel"])).mean(1)
    moving = np.flatnonzero(sp > thresh)
    if not len(moving):
        return 0, len(sp)
    return max(0, moving[0] - pad), min(len(sp), moving[-1] + pad + 1)


def trim(src, dst, start=None, end=None, auto=False):
    d = np.load(src, allow_pickle=True)
    n = len(d["joint_pos"])
    if auto:
        a, b = bounds(d)
        start = a if start is None else start
        end = b if end is None else end
    start, end = start or 0, n if end is None else end

    # slice anything that is per-frame; carry names, rate and the like through as-is
    out = {k: (v[start:end] if getattr(v := d[k], "shape", ()) and v.shape[0] == n else v)
           for k in d.files}
    np.savez(dst, **out)
    return start, end, n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("src")
    p.add_argument("dst", nargs="?")
    p.add_argument("--start", type=int)
    p.add_argument("--end", type=int)
    p.add_argument("--auto", action="store_true", help="keep the moving span")
    p.add_argument("--show", action="store_true", help="print the speed profile and stop")
    a = p.parse_args()

    if a.show:
        d = np.load(a.src, allow_pickle=True)
        for t, v in profile(d):
            print(f"{t:6.1f}s  {v:5.2f} rad/s  " + "#" * int(v * 25))
        return
    if not a.dst:
        p.error("dst is required unless --show")
    s, e, n = trim(a.src, a.dst, a.start, a.end, a.auto)
    print(f"{n} -> {e - s} frames [{s}:{e}] = {(e - s) / 50:.1f}s -> {a.dst}")


def _self_check():
    import tempfile, os
    n, ndof = 100, 3
    v = np.zeros((n, ndof))
    v[30:70] = 1.0  # the only motion
    with tempfile.TemporaryDirectory() as td:
        src, dst = os.path.join(td, "a.npz"), os.path.join(td, "b.npz")
        np.savez(src, joint_pos=np.zeros((n, ndof)), joint_vel=v,
                 joint_names=np.array(["a", "b", "c"]), rate=np.array([50.0]))
        s, e, _ = trim(src, dst, auto=True)
        d = np.load(dst, allow_pickle=True)
        assert (s, e) == (5, 95), (s, e)          # 30-25 .. 69+25+1
        assert len(d["joint_pos"]) == 90
        assert list(d["joint_names"]) == ["a", "b", "c"]  # not sliced
        print(f"ok: kept [{s}:{e}], names carried through")


if __name__ == "__main__":
    import sys
    if "--test" in sys.argv:
        _self_check()
    else:
        main()
