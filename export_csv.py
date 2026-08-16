"""Write a motion_command .npz as the CSV the shape14 policy runtime reads.

Their format (`transition/motion_clip.py`):

    x, y, z, qx, qy, qz, qw, <joint columns...>      50 fps

Three details that are easy to get wrong, all taken from their loader rather than guessed:

* **The quaternion is xyzw in the file** and wxyz internally (`quat = data[:, [6,3,4,5]]`),
  so the column order here is not the one jaxlie or MuJoCo hand you.
* **A header naming the joint columns is parsed**, not skipped — `joint_order` comes from
  `first.split(",")[7:]`. Writing one removes any dependence on our column order matching
  their `robot_joint_order`, which is the same class of bug as the MuJoCo qpos mapping.
* **Velocity is theirs to compute**, as a forward difference times fps. Our `joint_vel` is
  deliberately not a column: sending it would be ignored, and the clamp we apply upstream
  still shapes what they derive because it constrains the positions.

    .venv-ik/bin/python export_csv.py outputs/feas_mc.npz outputs/ref.csv
"""

import argparse

import numpy as np


def write_csv(npz_path, csv_path, header=True, root=None, q=None):
    d = np.load(npz_path, allow_pickle=True)
    # the sim export reshapes the clip (lead-in, ground correction); when it passes its
    # own arrays the runtime gets the exact trajectory the simulation was checked against
    q = np.asarray(d["joint_pos"] if q is None else q, np.float64)
    names = [str(n) for n in d["joint_names"]]
    n = len(q)

    if root is not None:
        # the sim export corrects the root for ground contact; the runtime should read
        # the same trajectory the simulation was checked against, not the raw capture
        root = np.asarray(root, np.float64)
    elif "root" in d.files and len(d["root"]) == n:
        root = np.asarray(d["root"], np.float64)  # already x,y,z,qx,qy,qz,qw
    else:
        # older captures dropped the root because the policy's observation never reads a
        # position; the file format still needs the columns, so stand it upright at origin
        root = np.tile(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]), (n, 1))

    rows = np.hstack([root, q])
    head = ",".join(["x", "y", "z", "qx", "qy", "qz", "qw"] + names) if header else ""
    np.savetxt(csv_path, rows, delimiter=",", fmt="%.6f", header=head, comments="")
    return n, len(names), float(d["rate"]) if "rate" in d.files else 50.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("npz")
    p.add_argument("csv")
    p.add_argument("--no_header", action="store_true",
                   help="write positionally instead; then the column order must match "
                        "the runtime's robot_joint_order exactly")
    a = p.parse_args()
    n, ndof, rate = write_csv(a.npz, a.csv, header=not a.no_header)
    print(f"{n} frames x {ndof} joints @ {rate:.0f} Hz -> {a.csv}")


if __name__ == "__main__":
    main()
