"""IK worker process: 14 joint positions in, robot joint angles out, over stdin/stdout.

Lives in its own process because it needs `.venv-ik` (JAX) while perception needs the
GEM-X venv (torch). It runs on CPU, so it does not compete with perception for the GPU.

Protocol, little-endian float32, no framing beyond fixed sizes:
    in   14*3 floats  — target joint positions, metres, z-up
    out  1 + ndof floats — solve time in ms, then the joint angles

    .venv-ik/bin/python ik_server.py --robot g1
"""

import struct
import sys
import time as _t

import jaxlie
import jax.numpy as jnp
import numpy as np
import time

from motion_command import MotionCommand
from ik_retarget import (ROBOTS, REST_W, R_WEIGHTS, TWIST_REST_W, VIS_THR, soma_frames, bone_frames,
                         build, cam_to_world, make_solver, scale_to_robot, target_weights, torso_frame)

# Orientation targets are off by default. Measured on the symmetric rest pose
# (test_apose.py), each one makes the solve worse, not better:
#   positions only  64 mm   + torso frame  87 mm
#   + bone frames  211 mm   both          268 mm
# bone_frames drives hip_yaw to -180 deg — the legs turn backwards, which is the crossing
# seen live — and torso_frame splays hip_roll from 9 to 24 deg. Enable with --frames to
# experiment; the arm-fold and torso-lean they were meant to fix need another answer.
BONES = {"LeftForeArm": 1.5, "LeftHand": 1.5, "RightForeArm": 1.5, "RightHand": 1.5,
         "LeftShin": 1.0, "LeftFoot": 1.0, "RightShin": 1.0, "RightFoot": 1.0}


MJCF = {
    "g1": "/home/robotis-ai/.cache/newton/newton-assets_unitree_g1_308a72cd/"
          "unitree_g1/mjcf/g1_29dof_rev_1_0.xml",
    "k1": "/home/robotis-ai/Projects/shape15/GEM-X/third_party/soma-retargeter/"
          "soma_retargeter/configs/ai_sapiens/ai_sapiens_retarget.xml",
}


class Sim:
    """MuJoCo display of the solved pose. Kinematic only — qpos is set, never stepped."""

    def __init__(self, name, actuated_names, stream_port=None):
        import os

        if stream_port:  # no local window: render offscreen and serve the frames
            os.environ.setdefault("MUJOCO_GL", "egl")
        import mujoco

        self.mj = mujoco
        self.stream = None
        self.model = mujoco.MjModel.from_xml_path(MJCF[name])
        self.data = mujoco.MjData(self.model)
        # our joint vector is in URDF order; map it onto MuJoCo's qpos by name
        self.qadr = []
        for jn in actuated_names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jn)
            self.qadr.append(self.model.jnt_qposadr[jid] if jid >= 0 else -1)
        missing = [n for n, a in zip(actuated_names, self.qadr) if a < 0]
        if missing:
            print(f"[sim] joints absent from MJCF, left at rest: {missing}", file=sys.stderr)
        self.free = self.model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE
        if stream_port:
            from mjpeg import Streamer

            self.renderer = mujoco.Renderer(self.model, 480, 640)
            self.cam = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(self.cam)
            # the operator faces the camera, which puts the robot's back to a default
            # view; look from the front so it faces the person watching
            self.cam.distance, self.cam.elevation, self.cam.azimuth = 3.0, -10, -45
            self.cam.lookat[:] = [0, 0, 0.8]
            self.stream = Streamer(stream_port)
            self.viewer = None
        else:
            import mujoco.viewer

            self.viewer = mujoco.viewer.launch_passive(
                self.model, self.data, show_left_ui=False, show_right_ui=False
            )

    def show(self, q, T, prof=None):
        t0 = time.time()
        if self.free:
            self.data.qpos[0:3] = np.asarray(T.translation())
            self.data.qpos[3:7] = np.asarray(T.rotation().wxyz)
        for a, v in zip(self.qadr, np.asarray(q)):
            if a >= 0:
                self.data.qpos[a] = v
        self.mj.mj_forward(self.model, self.data)
        t1 = time.time()
        if self.stream is not None:
            self.renderer.update_scene(self.data, self.cam)
            img = self.renderer.render()[..., ::-1]
            t2 = time.time()
            self.stream.put(img)  # includes the JPEG encode
        else:
            self.viewer.sync()
            t2 = t1
        if prof is not None:
            prof["mj_forward"].append(t1 - t0)
            prof["render"].append(t2 - t1)
            prof["jpeg+put"].append(time.time() - t2)


def main():
    name = sys.argv[sys.argv.index("--robot") + 1] if "--robot" in sys.argv else "g1"
    import ik_retarget

    ik_retarget.IK_MAP = ROBOTS[name][1]
    robot, link_idx, pos_w = build(name)
    # --no_twist_null restores the uniform rest weight used before the twist DOFs were
    # pinned. On the dance clip that is 13.1 deg against soma-retargeter for K1 where
    # pinning gets 8.0; on low-motion footage the two are within noise.
    solve, _, _ = make_solver(
        robot, link_idx, pos_w,
        REST_W if "--no_twist_null" in sys.argv else TWIST_REST_W[name])
    ndof = robot.joints.num_actuated_joints
    use_frames = "--frames" in sys.argv
    # Mirror mode: the robot faces the operator and must move the arm on the operator's
    # own side, which means driving its opposite limb — so mirror the targets, not the
    # view. Note this mirrors the joint angles sent downstream too; for real teleoperation
    # (as opposed to a mirror demo) leave it off.
    mirror = "--mirror" in sys.argv
    # --free_root: keep the operator's horizontal motion instead of pinning the pelvis to
    # the origin every frame. Required for a policy reference; optional for the display,
    # where a wandering robot is just harder to look at.
    free_root = "--free_root" in sys.argv
    root_origin = None
    # --mlp: perception appends network-predicted joint angles to each frame; use them
    # instead of solving, and only fit the free base so the display stands in place.
    use_mlp = "--mlp" in sys.argv
    profile = "--profile" in sys.argv
    prof = {k: [] for k in ("parse+scale", "solve", "show", "mj_forward", "render", "jpeg+put")}
    port = int(sys.argv[sys.argv.index("--stream") + 1]) if "--stream" in sys.argv else None
    sim = (Sim(name, list(robot.joints.actuated_names), port)
           if ("--view" in sys.argv or port) else None)
    mc_path = sys.argv[sys.argv.index("--motion_command") + 1] if "--motion_command" in sys.argv else None
    mc = (MotionCommand(ndof, 50.0, robot.joints.actuated_names,
                    np.asarray(robot.joints.velocity_limits)) if mc_path else None)
    # --ref_stream HOST:PORT publishes each 50 Hz step as it is produced, so a simulator
    # can follow the operator instead of a file. It rides on the same resampled, filtered,
    # velocity-clamped steps the recording writes: the live demo and the saved clip are
    # then the same signal, and a fault in one is a fault in both.
    pub = None
    if "--ref_stream" in sys.argv:
        import ref_stream
        host, _, port = sys.argv[sys.argv.index("--ref_stream") + 1].rpartition(":")
        pub = ref_stream.Publisher(name, list(robot.joints.actuated_names),
                                   (host or "127.0.0.1", int(port)))
        print(f"[ref_stream] publishing to {host or '127.0.0.1'}:{port}", file=sys.stderr)
        if mc is None:
            mc = MotionCommand(ndof, 50.0, robot.joints.actuated_names,
                               np.asarray(robot.joints.velocity_limits))
    torso_i = 1  # IK_MAP index of Chest -> the torso link

    # Per-joint constant offsets measured against soma-retargeter's own output. SOMA and
    # the robot disagree on where each joint's zero sits; a constant absorbs that. Only
    # joints whose residual drops below 15 deg carry one — where a constant does not
    # explain the error (hip_yaw, shoulder_yaw, wrist_roll) it would just mask it.
    offsets = np.zeros(ndof, np.float32)
    if "--no_calib" not in sys.argv:
        try:
            cal = np.load(f"fixtures/{name}_joint_offsets.npz", allow_pickle=True)
            by_name = dict(zip([str(x) for x in cal["names"]], cal["offsets"]))
            offsets = np.array([by_name.get(n, 0.0) for n in robot.joints.actuated_names],
                               np.float32)
            print(f"[calib] offsets on {int((offsets != 0).sum())} joints", file=sys.stderr)
        except FileNotFoundError:
            pass

    q, T = jnp.zeros(ndof), jaxlie.SE3.identity()
    # --stamped: the frame carries its own capture time as a trailing float. Offline
    # replay runs faster than the take was recorded, so the wall clock would compress a
    # two-minute performance into seconds and the 50 Hz resampling would keep almost
    # nothing of it. Live runs leave this off and the wall clock is the right answer.
    stamped = "--stamped" in sys.argv
    n_in = (14 * 3 + 14 + 14 * 9) * 4  # xyz, confidence, and joint rotation
    if use_mlp:
        n_in += ndof * 4  # plus the network's joint angles
    if stamped:
        n_in += 4
    scales, calib, floor, recalibrating = None, [], 0.0, False
    CALIB_FRAMES = 15  # ~1 s at 15 Hz

    # warm up the jit on a plausible pose so the first real frame is not the slow one
    q, T = solve(jnp.zeros((14, 3)), q, T)
    q.block_until_ready()
    sys.stdout.buffer.write(struct.pack("<f", 0.0))  # ready marker
    sys.stdout.buffer.flush()

    while True:
        buf = sys.stdin.buffer.read(n_in)
        if len(buf) < n_in:
            break
        t_in = time.time()
        raw = np.frombuffer(buf, dtype="<f4")
        t_frame = float(raw[-1]) if stamped else None
        if stamped:
            raw = raw[:-1]
        mlp_q = raw[-ndof:].copy() if use_mlp else None
        conf = raw[42:56].copy()
        if np.signbit(conf[0]):
            # Operator pressed Calibrate. Restart the measurement from this frame, always:
            # if one was already running it was collecting whatever happened before the
            # button, which is exactly what the button exists to discard. Perception sends
            # the request once, so this cannot re-trigger itself.
            conf[0] = abs(conf[0])
            print("[calib] re-measuring scales and floor", file=sys.stderr)
            # the current scales keep driving the robot until the new ones replace them
            calib, recalibrating = [], True
            root_origin = None  # re-anchor where the operator is standing now
            if mc is not None:
                mc.reset()  # the clip starts at Calibrate, not at process start
        jrot = raw[56:182].reshape(14, 3, 3).copy()  # SOMA joint world rotations, camera frame
        pts = raw[:42].reshape(1, 14, 3).copy()
        if mirror:
            pts[..., 0] *= -1.0  # camera-frame x is the operator's left-right
        targets = cam_to_world(pts).copy()
        if free_root:
            # Anchor once, at Calibrate, instead of every frame. Pinning the pelvis to the
            # origin each frame keeps the display tidy but deletes the operator's weight
            # shift, and a reference that lifts a foot 0.35 m while the pelvis never leaves
            # centre is one no robot can execute — measured as the left foot running 0.25 m
            # away from the policy's own foot until the run terminated.
            if root_origin is None:
                root_origin = targets[0, 0, :2].copy()
            targets[..., :2] -= root_origin
        else:
            targets[..., :2] -= targets[:, :1, :2]  # keep the robot near the origin
        if calib is not None:
            # Calibrate over a window, not one frame: bone lengths from a single pose
            # estimate are noisy, and they hold until the operator asks for new ones.
            # While measuring, scale off each frame's own bones and keep the previous
            # floor: the person is the right size on screen immediately and the robot does
            # not lurch. The two alternatives both leave the solver in a wrong branch that
            # it never leaves — clearing the floor lifts the robot ~0.7 m for the window,
            # and holding the previous scales puts a differently-sized person out of reach.
            calib.append(targets)
            scaled, _ = scale_to_robot(robot, link_idx, targets)
            if len(calib) >= CALIB_FRAMES:
                stack = np.concatenate(calib)
                _, scales = scale_to_robot(robot, link_idx, stack)
                fixed, _ = scale_to_robot(robot, link_idx, stack, scales)
                floor = float(fixed[..., 2].min())  # constant, so jumps still read as jumps
                calib = None
                if recalibrating:
                    # Re-converge from the rest pose, so the previous person's pose stops
                    # anchoring this one. Measured in task space: neutral on K1, and on
                    # G1 — six more DOF, so more places to get stuck — it stops the robot
                    # staying where the size change pushed it (397 -> 338 mm).
                    # Only on a *re*-calibration: the opening one has no stale pose to
                    # escape, and resetting there measured worse on G1 (281 -> 338 mm).
                    # An earlier version judged this by joint angles on the symmetric rest
                    # pose and concluded the opposite; joint angles are not unique.
                    q, T = jnp.zeros(ndof), jaxlie.SE3.identity()
                    recalibrating = False
        else:
            scaled, _ = scale_to_robot(robot, link_idx, targets, scales)
        scaled[..., 2] -= floor
        # a joint the pose estimator was unsure of should not drag the robot around;
        # occluded legs at weight 30 are what cross them
        w = jnp.array(target_weights(pos_w, conf))
        quat, ow = jnp.zeros((14, 4)).at[:, 0].set(1.0), jnp.zeros((14, 1))
        if use_frames:
            if np.abs(jrot).max() > 1e-6:
                quat, ow = soma_frames(robot, link_idx, jrot, R_WEIGHTS)
            else:
                quat, ow = bone_frames(robot, link_idx, scaled, BONES)
            quat = quat.at[0].set(torso_frame(scaled)[0]).astype(jnp.float32)
            ow = (ow.at[0].set(3.0)) * (np.asarray(w) > 0)
        quat = jnp.asarray(quat, jnp.float32)
        t0 = time.time()
        prof["parse+scale"].append(t0 - t_in)
        if use_mlp:
            # the network already matched the teacher's joint angles; what remains is
            # where to stand. Kabsch-fit the base from FK against the scaled targets.
            q = jnp.asarray(mlp_q)
            fk = np.asarray(jaxlie.SE3(robot.forward_kinematics(q)[link_idx]).translation())
            a = fk - fk.mean(0)
            b = scaled[0] - scaled[0].mean(0)
            U, _, Vt = np.linalg.svd(b.T @ a)
            S = np.diag([1.0, 1.0, float(np.sign(np.linalg.det(U @ Vt)))])
            R = U @ S @ Vt
            T = jaxlie.SE3.from_rotation_and_translation(
                jaxlie.SO3.from_matrix(jnp.array(R)),
                jnp.array(scaled[0].mean(0) - R @ fk.mean(0)))
            q_out = q  # trained straight onto the teacher; the offsets are the IK's, not its
        else:
            q, T = solve(jnp.array(scaled[0]), q, T, w, quat, ow)
            q.block_until_ready()
            q_out = jnp.asarray(np.asarray(q) - offsets)
        prof["solve"].append(time.time() - t0)
        t_show = time.time()
        if sim is not None:
            sim.show(q_out, T, prof if profile else None)
        prof["show"].append(time.time() - t_show)
        if mc is not None:
            # the policy wants the reference torso orientation, not a root position
            R = np.asarray((T @ jaxlie.SE3(
                robot.forward_kinematics(q)[link_idx[torso_i]])).rotation().as_matrix())
            # the CSV the policy runtime reads wants a root pose too (7 columns,
            # quaternion stored xyzw), so carry the IK's own base transform along
            root = np.concatenate([np.asarray(T.translation()),
                                   np.asarray(T.rotation().wxyz)[[1, 2, 3, 0]]])
            emitted = mc.push(_t.time() if t_frame is None else t_frame,
                              np.asarray(q_out), R, root.astype(np.float32))
            if pub is not None:
                for st in emitted:
                    pub.send(st[1], st[4] if st[4] is not None
                             else np.array([0, 0, 0, 0, 0, 0, 1], np.float32))
        out = np.concatenate([[(time.time() - t0) * 1000], np.asarray(q_out)]).astype("<f4")
        sys.stdout.buffer.write(out.tobytes())
        sys.stdout.buffer.flush()


    if profile:
        print("[worker profile] median ms:", file=sys.stderr)
        for k, v in prof.items():
            if v:
                print(f"  {k:12s} {np.median(v[3:]) * 1000:6.2f}  (n={len(v)})", file=sys.stderr)
    if mc is not None and mc_path:
        n = mc.save(mc_path)
        print(f"[motion_command] wrote {n} steps at {mc.rate:.0f} Hz to {mc_path}",
              file=sys.stderr)


if __name__ == "__main__":
    main()
