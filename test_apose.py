"""Feed the symmetric rest pose through the IK. Asymmetric output = a mapping bug."""
import sys, numpy as np, jax.numpy as jnp, jaxlie
sys.path.insert(0, "/home/robotis-ai/Projects/shape15")
import ik_retarget as IR
IR.IK_MAP = IR.ROBOTS["k1"][1]
robot, link_idx, pos_w = IR.build("k1")
solve, _, _ = IR.make_solver(robot, link_idx, pos_w, IR.TWIST_REST_W["k1"])
names = list(robot.joints.actuated_names)

a = np.load("fixtures/soma_rest14.npy")
a = a * np.array([1.0, -1.0, 1.0], np.float32)  # SOMA rest is y-up; cam_to_world wants y-down
tg = IR.cam_to_world(a); tg[..., :2] -= tg[:, :1, :2]
sc, scales = IR.scale_to_robot(robot, link_idx, tg)
BONES = {"LeftForeArm":1.5,"LeftHand":1.5,"RightForeArm":1.5,"RightHand":1.5,
         "LeftShin":1.0,"LeftFoot":1.0,"RightShin":1.0,"RightFoot":1.0}
# positions only, matching the shipped default — orientation targets measured worse
# (64 mm here vs 268 mm with both; bone_frames turned hip_yaw to -180 deg)
quat, ow = jnp.zeros((14, 4)).at[:, 0].set(1.0).astype(jnp.float32), jnp.zeros((14, 1))
if "--frames" in sys.argv:
    quat, ow = IR.bone_frames(robot, link_idx, sc, BONES)
    quat = quat.at[0].set(IR.torso_frame(sc)[0]).astype(jnp.float32)
    ow = ow.at[0].set(3.0)
q, T = jnp.zeros(len(names)), jaxlie.SE3.identity()
for _ in range(4):  # let it converge from rest
    q, T = solve(jnp.array(sc[0]), q, T, None, quat, ow)
q = np.asarray(q)

print("joint angles for a symmetric rest pose (deg):")
bad = []
for i, n in enumerate(names):
    if not n.startswith("left_"):
        continue
    j = names.index(n.replace("left_", "right_"))
    l, r = np.degrees(q[i]), np.degrees(q[j])
    # mirrored pose: pitch matches, roll/yaw negate
    exp = l if ("pitch" in n or "knee" in n or "elbow" in n) else -l
    dev = abs(exp - r)
    flag = "  <-- asymmetric" if dev > 5 else ""
    bad.append(dev)
    print(f"  {n.replace('left_',''):26s} L {l:7.1f}  R {r:7.1f}  dev {dev:5.1f}{flag}")
print(f"max asymmetry {max(bad):.1f} deg")
np.savez("outputs/apose_ref.npz", joint_pos=np.repeat(q[None], 60, 0).astype(np.float32),
         joint_vel=np.zeros((60, len(names)), np.float32),
         torso_ori6=np.zeros((60, 6), np.float32),
         joint_names=np.array(names, dtype=object), rate=50.0)
fk = jaxlie.SE3(robot.forward_kinematics(jnp.array(q))[link_idx])
got = np.asarray((T @ fk).translation())
print("target error: mean %.0f mm" % (np.linalg.norm(got - sc[0], axis=-1).mean() * 1000))
