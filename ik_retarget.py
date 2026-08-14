"""Per-frame SOMA -> robot inverse kinematics with PyRoki.

Replaces soma-retargeter's whole-motion API (30-frame chunks, ~870 ms of latency) with a
warm-started solve per frame. Correspondences and weights are lifted straight from
soma-retargeter's `ik_map`; adding a robot means one entry in ROBOTS.

    .venv-ik/bin/python ik_retarget.py soma14.npy --robot g1
"""

import sys
import time

import jax
import jax.numpy as jnp
import jaxlie
import jaxls
import numpy as np
import pyroki
import yourdfpy

# (SOMA joint, robot link, position weight) — from each robot's ik_map, in the order the
# exporter writes them. K1 differs from G1 only at the wrists: 23 DOF, no wrist yaw.
def _ik_map(hand_l, hand_r):
    return [
        ("Hips", "pelvis", 30.0),
        ("Chest", "torso_link", 0.7),
        ("LeftArm", "left_shoulder_roll_link", 1.5),
        ("LeftForeArm", "left_elbow_link", 1.0),
        ("LeftHand", hand_l, 2.0),
        ("RightArm", "right_shoulder_roll_link", 1.5),
        ("RightForeArm", "right_elbow_link", 1.0),
        ("RightHand", hand_r, 2.0),
        ("LeftLeg", "left_hip_roll_link", 1.5),
        ("LeftShin", "left_knee_link", 1.0),
        ("LeftFoot", "left_ankle_roll_link", 30.0),
        ("RightLeg", "right_hip_roll_link", 1.5),
        ("RightShin", "right_knee_link", 1.0),
        ("RightFoot", "right_ankle_roll_link", 30.0),
    ]


ROBOTS = {
    "g1": (
        "/home/robotis-ai/.cache/newton/newton-assets_unitree_g1_308a72cd/"
        "unitree_g1/urdf/g1_29dof_rev_1_0.urdf",
        _ik_map("left_wrist_yaw_link", "right_wrist_yaw_link"),
    ),
    "k1": (
        "/home/robotis-ai/Projects/shape15/GEM-X/third_party/soma-retargeter/"
        "third_party/ai_sapiens/ai_sapiens_description/urdf/k1_rev1/k1.urdf",
        _ik_map("left_wrist_roll_rubber_hand", "right_wrist_roll_rubber_hand"),
    ),
}
IK_MAP = ROBOTS["g1"][1]  # rebound in main() once the robot is known


def build(name):
    urdf_path, _ = ROBOTS[name]
    robot = pyroki.Robot.from_urdf(yourdfpy.URDF.load(urdf_path, load_meshes=False))
    names = list(robot.links.names)
    link_idx = jnp.array([names.index(link) for _, link, _ in IK_MAP], dtype=jnp.int32)
    pos_w = jnp.array([[w] for _, _, w in IK_MAP], dtype=jnp.float32)  # (14,1) to broadcast over xyz
    return robot, link_idx, pos_w


REST_W = 0.01  # every joint is weakly pulled toward its zero

# How hard to pull the twist DOFs. Per robot because the measured optimum differs — K1 has
# no wrist yaw, so its remaining twist DOFs carry more and take a firmer hand. Measured on
# the dance clip against each robot's own soma-retargeter output (`make_labels.py`):
#   K1  0.01 -> 13.1 deg | 0.05 -> 8.7 | 0.10 -> 8.0 | 0.30 -> 9.3
#   G1  0.01 -> 10.6 deg | 0.05 -> 7.9 | 0.10 -> 9.7 | 0.30 -> 11.1
TWIST_REST_W = {"g1": 0.05, "k1": 0.1}


def twist_joints(robot):
    """The DOFs 14 points cannot determine: rotation about a limb's own axis.

    Name-based, so it transfers to any robot the way the rest of this file does — G1 and
    K1 both name them `*_yaw_joint` and `*_wrist_roll_joint`.
    """
    return [i for i, n in enumerate(robot.joints.actuated_names)
            if "yaw" in n or "wrist_roll" in n]


def make_solver(robot, link_idx, pos_w, twist_w):
    joint_var, base_var = robot.joint_var_cls(0), jaxls.SE3Var(0)
    rest = jnp.zeros(robot.joints.num_actuated_joints)

    # Hold the unobservable twist DOFs near zero instead of letting the solver put them
    # anywhere that fits — the cheap half of what soma-retargeter's wrist-roll nullspace
    # and limb-plane objectives do. It only pays where the twist actually wanders: on the
    # dance clip it takes K1 from 13.1 to 8.0 deg and G1 from 10.6 to 7.9, but on a
    # low-motion webcam capture, where the twist joints already behave, it is worth
    # nothing and costs a little target accuracy. Pass twist_w=REST_W to disable.
    w_rest = np.full(robot.joints.num_actuated_joints, REST_W, np.float32)
    w_rest[twist_joints(robot)] = twist_w
    w_rest = jnp.array(w_rest)

    @jax.jit
    def solve(targets, q0, T0, w=None, quat=None, ow=0.0):
        # Points alone leave rotations free — the torso can lean while the hip target is
        # still met. `quat` supplies target orientations (wxyz) where we can derive them.
        rot = jaxlie.SO3.identity((len(IK_MAP),)) if quat is None else jaxlie.SO3(quat)
        pose = jaxlie.SE3.from_rotation_and_translation(rot, targets)
        costs = [
            pyroki.costs.pose_cost_with_base(
                robot, joint_var, base_var, pose, link_idx,
                pos_w if w is None else w, ow,
            ),
            pyroki.costs.limit_cost(robot, joint_var, 100.0),
            pyroki.costs.rest_cost(joint_var, rest, w_rest),
        ]
        sol = (
            jaxls.LeastSquaresProblem(costs, [joint_var, base_var])
            .analyze()
            .solve(
                initial_vals=jaxls.VarValues.make(
                    [joint_var.with_value(q0), base_var.with_value(T0)]
                ),
                verbose=False,
            )
        )
        return sol[joint_var], sol[base_var]

    return solve, joint_var, base_var


VIS_THR = 0.6  # below this a joint is the model's guess, not an observation


def target_weights(base_w, conf):
    """Scale each target by how much MediaPipe actually saw it.

    Landmarks outside the frame are still emitted — they are the model's guess at where a
    limb probably is. The ik_map weights feet at 30, the highest of all, so on a waist-up
    shot the solver chases invented feet and folds the robot onto the floor. Weighting by
    visibility lets the rest cost hold the unseen joints in a neutral pose instead.
    """
    scale = np.clip((conf - VIS_THR) / (1.0 - VIS_THR), 0.0, 1.0)
    return (np.asarray(base_w).reshape(-1) * scale).reshape(-1, 1).astype(np.float32)


def torso_frame(t14):
    """Orientation for the pelvis, built from the hips and chest we can see.

    A hip position target is a point and constrains no rotation at all, so the solver is
    free to lean the whole robot while still hitting it. Two observed directions are enough
    to pin it: up along hips->chest, left along right-hip->left-hip.
    """
    hips, chest, hip_l, hip_r = t14[0, 0], t14[0, 1], t14[0, 8], t14[0, 11]
    up = chest - hips
    up /= max(np.linalg.norm(up), 1e-6)
    left = hip_l - hip_r
    fwd = np.cross(left, up)
    fwd /= max(np.linalg.norm(fwd), 1e-6)
    left = np.cross(up, fwd)
    R = np.stack([fwd, left, up], axis=1)  # world x fwd, y left, z up
    q = np.zeros((14, 4), np.float32)
    q[:, 0] = 1.0  # identity for everything else
    q[0] = np.asarray(jaxlie.SO3.from_matrix(jnp.array(R)).wxyz)
    return jnp.array(q)


# rotation weights from soma-retargeter's ik_map r_weight — the half we never used
R_WEIGHTS = {
    "Hips": 2.0, "Chest": 0.7,
    "LeftArm": 0.15, "LeftForeArm": 1.0, "LeftHand": 1.2,
    "RightArm": 0.15, "RightForeArm": 1.0, "RightHand": 1.2,
    "LeftLeg": 0.15, "LeftShin": 1.0, "LeftFoot": 2.0,
    "RightLeg": 0.15, "RightShin": 1.0, "RightFoot": 2.0,
}


def soma_frames(robot, link_idx, delta_cam, weights):
    """Target orientations from SOMA's own joint frames, not from point directions.

    SOMA and the robot use unrelated link conventions, so absolute rotations are not
    interchangeable. What transfers is the *change* — how far SOMA has rotated a joint from
    its A-pose, which the perception side already differences out. Applying that to the
    robot link's rest orientation carries the twist about each limb, precisely what
    positions cannot express and what leaves an arm free to fold into the wrong branch.
    """
    rest_T = jaxlie.SE3(robot.forward_kinematics(
        jnp.zeros(robot.joints.num_actuated_joints))[link_idx])
    rest_R = np.asarray(rest_T.rotation().as_matrix())
    names = [soma for soma, _, _ in IK_MAP]
    R_cam2world = np.array([[0, 0, 1.0], [-1.0, 0, 0], [0, -1.0, 0]])  # same as cam_to_world
    # The delta acts on SOMA-canonical directions (y-up, facing +z); A carries those into
    # the robot world (z-up, facing +x). The first version conjugated by R_cam2world
    # instead, which maps SOMA's up to world *down* — checked against the position-derived
    # torso_frame on real frames, this A agrees to 4.4 deg where the old constant was
    # 180 deg out. (Correct frames still only move the teacher gap 17.1 -> 16.4 on held-out
    # clips: the remaining obstacle is A-pose vs rest mismatch on the arms, not the frame.)
    A = np.array([[0, 0, 1.0], [1.0, 0, 0], [0, 1.0, 0]])
    quat = np.zeros((len(names), 4), np.float32)
    quat[:, 0] = 1.0
    w = np.zeros((len(names), 1), np.float32)
    for i, name in enumerate(names):
        weight = weights.get(name, 0.0)
        if weight <= 0.0:
            continue
        delta = R_cam2world @ delta_cam[i] @ A.T
        quat[i] = np.asarray(jaxlie.SO3.from_matrix(jnp.array(delta @ rest_R[i])).wxyz)
        w[i] = weight
    return jnp.array(quat), jnp.array(w)


def bone_frames(robot, link_idx, scaled, which):
    """Target orientations that point each named link's bone the way the data does.

    Positions alone leave the twist about a limb free, so an arm can hit shoulder, elbow
    and wrist targets while rolled into an anatomically wrong branch. Rather than guess
    which local axis of a link runs along its bone — that is robot-specific — take the
    rotation that carries the link's *rest* bone direction onto the observed one and apply
    it to the link's rest orientation. That pins the two observable degrees of freedom and
    leaves the unobservable twist alone.
    """
    rest_T = jaxlie.SE3(robot.forward_kinematics(
        jnp.zeros(robot.joints.num_actuated_joints))[link_idx])
    rest_p = np.asarray(rest_T.translation())
    rest_R = np.asarray(rest_T.rotation().as_matrix())
    names = [soma for soma, _, _ in IK_MAP]
    quat = np.zeros((len(names), 4), np.float32)
    quat[:, 0] = 1.0
    w = np.zeros((len(names), 1), np.float32)
    for name, weight in which.items():
        i, p = names.index(name), names.index(PARENTS[name])
        a = rest_p[i] - rest_p[p]
        b = scaled[0, i] - scaled[0, p]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-6 or nb < 1e-6:
            continue
        a, b = a / na, b / nb
        v, c = np.cross(a, b), float(np.dot(a, b))
        if c < -0.999:  # antiparallel: any perpendicular axis will do
            v = np.cross(a, [1.0, 0.0, 0.0])
            if np.linalg.norm(v) < 1e-6:
                v = np.cross(a, [0.0, 1.0, 0.0])
            R_align = _rodrigues(v / np.linalg.norm(v), np.pi)
        else:
            R_align = np.eye(3) + _skew(v) + _skew(v) @ _skew(v) / (1.0 + c)
        quat[i] = np.asarray(jaxlie.SO3.from_matrix(jnp.array(R_align @ rest_R[i])).wxyz)
        w[i] = weight
    return jnp.array(quat), jnp.array(w)


def _skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], np.float64)


def _rodrigues(axis, ang):
    K = _skew(axis)
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)


def cam_to_world(j):
    """Camera frame (x right, y down, z forward) -> robot world (x forward, y left, z up).

    An axis permutation, not a sign flip: getting this wrong lays the robot on its side
    several metres below the floor, which renders as an empty scene.
    """
    return np.stack([j[..., 2], -j[..., 0], -j[..., 1]], axis=-1)


PARENTS = {  # chain over the 14 ik_map joints, from soma_to_g1_scaler_config.json
    "Chest": "Hips",
    "LeftArm": "Chest", "LeftForeArm": "LeftArm", "LeftHand": "LeftForeArm",
    "RightArm": "Chest", "RightForeArm": "RightArm", "RightHand": "RightForeArm",
    "LeftLeg": "Hips", "LeftShin": "LeftLeg", "LeftFoot": "LeftShin",
    "RightLeg": "Hips", "RightShin": "RightLeg", "RightFoot": "RightShin",
}


def scale_to_robot(robot, link_idx, targets, scales=None):
    """Rebuild the target skeleton with the robot's own bone lengths.

    G1 is shorter than an adult, so raw targets are unreachable and the solver settles on
    a large residual. A single ratio cannot match legs and torso at once (that left chest
    and shoulders ~140 mm out), and soma-retargeter's `joint_scales` are absolute ratios
    tuned for a 1.8 m human and meant to be applied with its `joint_offsets`. Measuring
    each bone off the robot's rest pose needs no config and is self-calibrating: every
    scaled segment is exactly as long as the robot's.
    """
    rest = np.asarray(jaxlie.SE3(robot.forward_kinematics(
        jnp.zeros(robot.joints.num_actuated_joints))[link_idx]).translation())
    names = [soma for soma, _, _ in IK_MAP]
    out = np.empty_like(targets)
    root = names.index("Hips")

    if scales is None:  # live, the first frame fixes them; offline, the whole clip does
        scales = {}
        for i, name in enumerate(names):
            if name == "Hips":
                continue
            p = names.index(PARENTS[name])
            robot_bone = np.linalg.norm(rest[i] - rest[p])
            human_bone = float(np.linalg.norm(targets[:, i] - targets[:, p], axis=-1).mean())
            scales[name] = robot_bone / max(human_bone, 1e-6)

    # stand the root at the robot's own hip height, keeping the horizontal trajectory
    leg = scales["LeftLeg"] * scales["LeftShin"] * scales["LeftFoot"]
    out[:, root] = targets[:, root] * np.array([1.0, leg ** (1 / 3), 1.0], np.float32)
    for i, name in enumerate(names):
        if name == "Hips":
            continue
        p = names.index(PARENTS[name])
        out[:, i] = out[:, p] + (targets[:, i] - targets[:, p]) * scales[name]
    return out, scales


def main():
    global IK_MAP
    name = sys.argv[sys.argv.index("--robot") + 1] if "--robot" in sys.argv else "g1"
    IK_MAP = ROBOTS[name][1]
    targets = jnp.array(np.load(sys.argv[1]))  # (L, 14, 3), metres, world frame
    robot, link_idx, pos_w = build(name)
    print(f"robot {name}: {robot.joints.num_actuated_joints} actuated joints")
    scaled, scales = scale_to_robot(robot, link_idx, np.asarray(targets))
    targets = jnp.array(scaled)
    print("measured bone scales: " + ", ".join(
        f"{k} {v:.2f}" for k, v in list(scales.items())[:5]))
    solve, _, _ = make_solver(robot, link_idx, pos_w, TWIST_REST_W[name])

    q = jnp.zeros(robot.joints.num_actuated_joints)
    T = jaxlie.SE3.identity()
    t_compile = time.time()
    q, T = solve(targets[0], q, T)
    q.block_until_ready()
    print(f"first solve (jit compile): {time.time() - t_compile:.1f} s")

    times, errs = [], []
    for i in range(len(targets)):
        t0 = time.time()
        q, T = solve(targets[i], q, T)  # warm start from the previous frame
        q.block_until_ready()
        times.append((time.time() - t0) * 1000)
        fk = jaxlie.SE3(robot.forward_kinematics(q)[link_idx])
        got = (T @ fk).translation()
        errs.append(np.asarray(jnp.linalg.norm(got - targets[i], axis=-1)))

    t = np.array(times[1:])
    print(f"IK per frame: median {np.median(t):.1f} ms, mean {t.mean():.1f} ms, "
          f"p90 {np.percentile(t, 90):.1f} ms over {len(t)} frames")
    print(f"sustained {1000 / t.mean():.0f} FPS (IK alone), "
          f"worst-frame {1000 / t.max():.0f} FPS")
    e = np.stack(errs) * 1000  # (L, 14) mm
    print(f"mean target error: {e.mean():.0f} mm")
    for (soma, _, w), v in zip(IK_MAP, e.mean(0)):
        print(f"  {soma:13s} w={w:4.1f}  {v:6.0f} mm")


if __name__ == "__main__":
    main()
