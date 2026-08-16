"""Drive the shape14 student policy in MuJoCo from one of our reference clips.

This closes roadmap step 6 without Isaac: the deployed student is an ONNX file plus
`sim2real.yaml`, and MuJoCo is already here for the display, so the loop can run on CPU.

    .venv-ik/bin/python sim_policy.py outputs/clips/live_take1/live_take1.npz \
        --video outputs/sim_take1.mp4

What the policy sees each step, in this order (124 = the student's observation):

    motion_command(46)  = reference joint positions 23 + reference joint velocities 23
    motion_anchor_ori_b(6) = reference torso orientation expressed in the robot's frame
    base_ang_vel(3), joint_pos_rel(23), joint_vel_rel(23), last_action(23)

Everything except the first two comes from the simulated robot, which is the point of the
exercise: it tells us whether a *robot* can follow the reference we generate, not whether
the numbers look plausible on paper.

Two conventions worth stating, both taken from `sim2real.yaml` rather than assumed:

* `joint_pos_rel` is measured against `default_position`, not zero, and the action is
  `scale * a + offset` in the same joint order as `policy_joints` — which is **not** our
  URDF order, so every vector is permuted by name.
* The anchor is a *relative* orientation: the reference torso frame seen from the robot's
  own base, so a robot facing elsewhere still gets a meaningful error signal.
"""

import argparse
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort
import yaml

import sim_scene
from ik_server import MJCF

ASSET = Path("/home/robotis-ai/Projects/shape14/outputs/student_eval/student_asset")


def quat_to_mat(q_wxyz):
    w, x, y, z = q_wxyz
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _load_adapter(asset):
    """shape14's own ExpertAdapter, loaded by path.

    Importing `cyclo_lab` pulls in Isaac; the adapter itself only needs numpy and yaml, so
    the asset's own `check_contract.py` loads it this way too. Using it rather than
    reimplementing the contract means the observation order, the history handling and the
    action decode all come from the side that trained the policy.
    """
    import importlib.util
    import sys

    src = ("/home/robotis-ai/Projects/shape14/cyclo_lab_private/source/cyclo_lab/"
           "cyclo_lab/transition/expert_adapter.py")
    spec = importlib.util.spec_from_file_location("expert_adapter", src)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["expert_adapter"] = mod  # its dataclasses look themselves up here
    spec.loader.exec_module(mod)
    k1 = Path("/home/robotis-ai/Projects/shape14/ai_sapiens_private/ai_sapiens_sim2real/"
              "config/k1_config.yaml")
    return mod.ExpertAdapter.from_asset_dir(asset, k1_config_path=k1 if k1.is_file() else None)


class Student:
    """The exported policy, driven through shape14's ExpertAdapter."""

    def __init__(self, asset=ASSET):
        cfg = yaml.safe_load((asset / "params/sim2real.yaml").read_text())
        self.names = list(cfg["policy_joints"])
        self.dt = float(cfg["step_dt"])
        jp = cfg["joint_properties"]
        self.default = np.array([jp[n]["default_position"] for n in self.names], np.float64)
        self.kp = np.array([jp[n]["stiffness"] for n in self.names], np.float64)
        self.kd = np.array([jp[n]["damping"] for n in self.names], np.float64)
        self.adapter = _load_adapter(asset)
        self.obs_dim = self.adapter.observation_size
        self.last_action = np.zeros(len(self.names))

    def act(self, terms):
        """terms: the observation by name; the adapter puts them in the trained order."""
        self.adapter.push_observation(terms)
        raw = self.adapter.act()
        self.last_action = raw
        return self.adapter.decode_action(raw, canonical=False)


def run(clip_npz, robot="k1", video=None, seconds=None, asset=ASSET):
    ref = np.load(clip_npz, allow_pickle=True)
    ref_names = [str(n) for n in ref["joint_names"]]
    ref_q, ref_dq = np.asarray(ref["joint_pos"]), np.asarray(ref["joint_vel"])
    body_names = [str(b) for b in ref["body_names"]]
    torso_i = body_names.index("torso_link")
    ref_torso_quat = np.asarray(ref["body_quat_w"])[:, torso_i]  # xyzw

    pol = Student(asset)
    # reference joints arrive in our URDF order; the policy has its own
    perm = [ref_names.index(n) for n in pol.names]

    # the retargeting MJCF has no actuators, no floor and no collision; sim_scene adds
    # exactly those, with the policy's own PD gains
    model, n_foot = sim_scene.build(MJCF[robot], pol.names, pol.kp, pol.kd)
    print(sim_scene.report(model, n_foot))
    model.opt.timestep = 0.002
    data = mujoco.MjData(model)
    n_sub = max(1, int(round(pol.dt / model.opt.timestep)))

    qadr, vadr = [], []
    for n in pol.names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
        qadr.append(model.jnt_qposadr[jid])
        vadr.append(model.jnt_dofadr[jid])
    qadr, vadr = np.array(qadr), np.array(vadr)

    # start standing at the policy's default pose, pelvis at the reference's first height
    mujoco.mj_resetData(model, data)
    data.qpos[qadr] = pol.default
    if model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE:
        data.qpos[0:3] = [0.0, 0.0, float(np.asarray(ref["body_pos_w"])[0, 0, 2])]
        data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)

    renderer = cam = None
    if video:
        renderer = mujoco.Renderer(model, 480, 640)
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(cam)
        cam.distance, cam.elevation, cam.azimuth = 3.0, -10, 135
        cam.lookat[:] = [0, 0, 0.8]

    n_steps = len(ref_q) if seconds is None else min(len(ref_q), int(seconds / pol.dt))
    frames, err, fell = [], [], None
    for k in range(n_steps):
        rq, rdq = ref_q[k][perm], ref_dq[k][perm]

        base_quat = data.qpos[3:7]                      # MuJoCo wxyz
        R_wb = quat_to_mat(base_quat)
        R_wt = quat_to_mat(ref_torso_quat[k][[3, 0, 1, 2]])
        anchor = (R_wb.T @ R_wt)[:, :2].T.reshape(-1)   # first two columns, base frame

        target = pol.act({
            "motion_command": np.concatenate([rq, rdq]),
            "motion_anchor_ori_b": anchor,
            "base_ang_vel": data.qvel[3:6].copy(),
            "joint_pos_rel": data.qpos[qadr] - pol.default,
            "joint_vel_rel": data.qvel[vadr].copy(),
            "last_action": pol.last_action,
        })
        data.ctrl[:] = target      # position servos carry the PD, as sim2real specifies
        for _ in range(n_sub):
            mujoco.mj_step(model, data)

        err.append(np.abs(data.qpos[qadr] - rq))
        if fell is None and data.qpos[2] < 0.35:
            fell = k * pol.dt
        if renderer is not None:
            renderer.update_scene(data, cam)
            frames.append(renderer.render())

    e = np.degrees(np.stack(err))
    print(f"{n_steps} steps ({n_steps * pol.dt:.1f} s) | tracking error "
          f"mean {e.mean():.1f}° p90 {np.percentile(e, 90):.1f}°")
    print(f"pelvis height end {data.qpos[2]:.3f} m | "
          + (f"fell at {fell:.1f} s" if fell is not None else "stayed up"))
    worst = np.argsort(-e.mean(0))[:4]
    print("worst joints: " + ", ".join(f"{pol.names[i]} {e[:, i].mean():.0f}°" for i in worst))
    if frames:
        import imageio
        imageio.mimsave(video, frames, fps=1.0 / pol.dt, macro_block_size=1)
        print(f"wrote {video}")
    return e


def main():
    p = argparse.ArgumentParser()
    p.add_argument("clip", help="a reference clip npz from export_npz.py")
    p.add_argument("--robot", default="k1", choices=["g1", "k1"])
    p.add_argument("--video")
    p.add_argument("--seconds", type=float)
    p.add_argument("--asset", default=str(ASSET))
    a = p.parse_args()
    run(a.clip, a.robot, a.video, a.seconds, Path(a.asset))


if __name__ == "__main__":
    main()
