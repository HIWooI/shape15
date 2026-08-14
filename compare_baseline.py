"""Our per-frame IK vs soma-retargeter's own joint angles, on the same frames."""
import sys, csv, numpy as np, jax.numpy as jnp, jaxlie
sys.path.insert(0, "/home/robotis-ai/Projects/shape15")
import ik_retarget as IR
from ik_retarget import R_WEIGHTS

TMP = "/home/robotis-ai/.claude-accounts/hw/jobs/d8c0acfd/tmp"
IR.IK_MAP = IR.ROBOTS["k1"][1]
robot, link_idx, pos_w = IR.build("k1")
solve, _, _ = IR.make_solver(robot, link_idx, pos_w, IR.TWIST_REST_W["k1"])
names = list(robot.joints.actuated_names)

d = np.load(f"{TMP}/ours_in.npz")
joints, conf, rot = d["joints"], d["conf"], d["rot"]

rows = list(csv.DictReader(open(f"{TMP}/baseline_k1.csv")))
base = np.array([[float(r[f"{n}_dof"]) for n in names] for r in rows], np.float32)
if np.abs(base).max() > 7:  # the CSV is in degrees, our solver works in radians
    print(f"baseline in degrees (max {np.abs(base).max():.0f}), converting")
    base = np.radians(base)
print(f"baseline {base.shape} | range [{base.min():.2f}, {base.max():.2f}] rad")

BONES = {"LeftForeArm": 1.5, "LeftHand": 1.5, "RightForeArm": 1.5, "RightHand": 1.5,
         "LeftShin": 1.0, "LeftFoot": 1.0, "RightShin": 1.0, "RightFoot": 1.0}

RZ180 = np.array([[-1.0, 0, 0], [0, -1.0, 0], [0, 0, 1.0]])


def run(use_rot, yaw_flip=False):
    tg = IR.cam_to_world(joints)
    tg[..., :2] -= tg[:, :1, :2]
    _, scales = IR.scale_to_robot(robot, link_idx, tg[:15])
    q, T, out = jnp.zeros(len(names)), jaxlie.SE3.identity(), []
    for i in range(len(tg)):
        sc, _ = IR.scale_to_robot(robot, link_idx, tg[i:i + 1], scales)
        w = jnp.array(IR.target_weights(pos_w, conf[i]))
        if use_rot:
            quat, ow = IR.soma_frames(robot, link_idx, rot[i], R_WEIGHTS)
        else:
            quat, ow = IR.bone_frames(robot, link_idx, sc, BONES)
        tq = IR.torso_frame(sc)[0]
        if yaw_flip:
            R = np.asarray(jaxlie.SO3(jnp.array(tq)).as_matrix())
            tq = jnp.array(np.asarray(jaxlie.SO3.from_matrix(jnp.array(RZ180 @ R)).wxyz))
        quat = quat.at[0].set(tq).astype(jnp.float32)
        ow = (ow.at[0].set(3.0)) * (np.asarray(w) > 0)
        q, T = solve(jnp.array(sc[0]), q, T, w, quat, ow)
        out.append(np.asarray(q))
    return np.stack(out)

n = min(len(base), len(joints))
for use_rot, flip, label in ((False, False, "points only"), (True, False, "+ SOMA rotations"),
                            (False, True, "points + yaw flip"), (True, True, "rot + yaw flip")):
    ours = run(use_rot, flip)[:n]
    err = np.abs(ours - base[:n])
    print(f"{label:18s} mean {np.degrees(err.mean()):5.1f} deg | median "
          f"{np.degrees(np.median(err)):5.1f} | p90 {np.degrees(np.percentile(err,90)):6.1f}")
    worst = np.argsort(-err.mean(0))[:3]
    print("      worst: " + ", ".join(f"{names[i]} {np.degrees(err[:,i].mean()):.0f}deg" for i in worst))
