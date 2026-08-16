"""Turn the retargeting MJCF into something physics can actually run.

The K1 model we have is built for retargeting, not simulation: `nu = 0`, no ground plane,
and every geom has `contype = conaffinity = 0`, so nothing collides. Loading it and
stepping just drops the robot through the floor. shape14 uses Isaac with a USD robot for
this reason; this rebuilds the missing pieces on top of the same MJCF so a closed loop can
run here on CPU.

Added, and nothing else:

* a ground plane,
* collision on the feet only — those meshes are visual quality, and enabling every one of
  them makes contact both slow and unstable, while foot-ground is what decides whether the
  robot stays up,
* one position actuator per policy joint, with the gains from `sim2real.yaml` so the PD
  matches what the policy was trained against.

The result is a rough stand-in, not shape14's simulation. It answers "can a robot follow
this reference at all", not "how well does the deployed system perform".
"""

import mujoco
import numpy as np

FOOT_HINT = ("ankle_roll", "ankle_pitch")


def build(mjcf_path, joint_names, kp, kd, foot_hint=FOOT_HINT):
    spec = mujoco.MjSpec.from_file(str(mjcf_path))

    floor = spec.worldbody.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [0.0, 0.0, 0.05]
    floor.rgba = [0.3, 0.35, 0.4, 1.0]
    floor.contype, floor.conaffinity = 1, 1

    # Feet only. The proxy bodies the retarget config injects are targets, not geometry,
    # so they are skipped by name as well as by their lack of geoms.
    n_foot = 0
    for g in spec.geoms:
        body = g.parent.name if g.parent is not None else ""
        if any(h in body for h in foot_hint) and "proxy" not in body:
            g.contype, g.conaffinity = 1, 1
            g.condim = 3
            n_foot += 1

    have = {j.name for j in spec.joints}
    for n, p, d in zip(joint_names, kp, kd):
        if n not in have:
            raise KeyError(f"{n} is not a joint in {mjcf_path}")
        a = spec.add_actuator()
        a.name = n
        a.target = n
        a.trntype = mujoco.mjtTrn.mjTRN_JOINT
        a.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        a.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        a.gainprm[0] = p          # position gain
        a.biasprm[1] = -p         # -kp * q
        a.biasprm[2] = -d         # -kd * qdot
    model = spec.compile()
    return model, n_foot


def report(model, n_foot):
    return (f"actuators {model.nu} | foot collision geoms {n_foot} | "
            f"floor {'yes' if (model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE).any() else 'no'}")


if __name__ == "__main__":
    import yaml
    from pathlib import Path
    from ik_server import MJCF

    cfg = yaml.safe_load(Path(
        "/home/robotis-ai/Projects/shape14/outputs/student_eval/student_asset/"
        "params/sim2real.yaml").read_text())
    names = list(cfg["policy_joints"])
    jp = cfg["joint_properties"]
    m, nf = build(MJCF["k1"], names,
                  [jp[n]["stiffness"] for n in names], [jp[n]["damping"] for n in names])
    print(report(m, nf))
    d = mujoco.MjData(m)
    d.qpos[2] = 0.8
    for _ in range(200):
        mujoco.mj_step(m, d)
    print(f"200 steps of free settle -> pelvis z {d.qpos[2]:.3f} m "
          f"({'lands on the floor' if d.qpos[2] > 0.1 else 'still falling through'})")
