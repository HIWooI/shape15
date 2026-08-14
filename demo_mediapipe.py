"""Camera -> MediaPipe -> robot IK -> MuJoCo, all in one CPU process.

The GEM-X path needs a GPU, two venvs and ~58 ms of perception, and its accuracy collapses
when the body is cropped — which is the normal webcam framing. MediaPipe is trained for
exactly that framing and predicts landmarks it cannot fully see, so this trades some
absolute accuracy for robustness on half-bodies, and drops perception to ~10 ms.

    .venv-ik/bin/python demo_mediapipe.py --robot g1
"""

import argparse
import threading
import time

import cv2
import jax.numpy as jnp
import jaxlie
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
import mediapipe as mp

import ik_retarget as IR
from ik_retarget import target_weights, torso_frame, VIS_THR
from ik_server import Sim

# MediaPipe's 33 landmarks -> the 14 joints of the ik_map, in IK_MAP order.
# A tuple means "midpoint of these two". Left/right are the person's own, as in SOMA.
MP_MAP = [
    (23, 24),  # Hips
    (11, 12),  # Chest
    11, 13, 15,  # LeftArm, LeftForeArm, LeftHand
    12, 14, 16,  # RightArm, RightForeArm, RightHand
    23, 25, 27,  # LeftLeg, LeftShin, LeftFoot
    24, 26, 28,  # RightLeg, RightShin, RightFoot
]
CALIB_FRAMES = 15
# limbs whose bone direction is observable, so their branch can be pinned
ARM_FRAMES = {"LeftForeArm": 1.5, "LeftHand": 1.5, "RightForeArm": 1.5, "RightHand": 1.5,
              "LeftShin": 1.0, "LeftFoot": 1.0, "RightShin": 1.0, "RightFoot": 1.0}
BONES = [(11, 13), (13, 15), (12, 14), (14, 16), (11, 12), (23, 24), (11, 23), (12, 24),
         (23, 25), (25, 27), (24, 26), (26, 28)]


def to14(world, vis):
    """MediaPipe's 33 landmarks -> (1, 14, 3) camera frame, plus a 0..1 confidence each."""
    out = np.empty((1, 14, 3), np.float32)
    conf = np.empty(14, np.float32)
    for i, src in enumerate(MP_MAP):
        idx = list(src) if isinstance(src, tuple) else [src]
        out[0, i] = world[idx].mean(0)
        conf[i] = vis[idx].min()
    return out, conf


class Camera:
    def __init__(self, index):
        self.cap = cv2.VideoCapture(index)
        assert self.cap.isOpened(), f"cannot open camera {index}"
        self.frame, self.lock, self.stop = None, threading.Lock(), False
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while not self.stop:
            ok, f = self.cap.read()
            if not ok:
                self.stop = True
                break
            with self.lock:
                self.frame = f

    def read(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--robot", default="g1", choices=["g1", "k1"])
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--flip", action="store_true")
    p.add_argument("--model", default="models/pose_landmarker_full.task")
    args = p.parse_args()

    IR.IK_MAP = IR.ROBOTS[args.robot][1]
    robot, link_idx, pos_w = IR.build(args.robot)
    solve, _, _ = IR.make_solver(robot, link_idx, pos_w)
    sim = Sim(args.robot, list(robot.joints.actuated_names))
    rest = np.asarray(jaxlie.SE3(robot.forward_kinematics(
        jnp.zeros(robot.joints.num_actuated_joints))[link_idx]).translation())
    stand_h = float(rest[0, 2] - rest[10, 2])  # hip height above the foot, at rest

    landmarker = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=args.model),
            running_mode=vision.RunningMode.VIDEO,
        )
    )

    q, T = jnp.zeros(robot.joints.num_actuated_joints), jaxlie.SE3.identity()
    q, T = solve(jnp.zeros((14, 3)), q, T)  # warm the jit before the camera opens
    q.block_until_ready()

    cam = Camera(args.camera)
    while cam.read() is None:
        time.sleep(0.05)
    scales, calib, floor, fps, t_ms, n_frames = None, [], 0.0, 0.0, 0, 0
    try:
        while not cam.stop:
            frame = cam.read()
            t0 = time.time()
            t_ms += 33
            res = landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=frame[..., ::-1].copy()), t_ms
            )
            if res.pose_world_landmarks:
                lms = res.pose_world_landmarks[0]
                world = np.array([[l.x, l.y, l.z] for l in lms], np.float32)
                vis = np.array([l.visibility for l in lms], np.float32)
                t14, conf = to14(world, vis)
                w = jnp.array(target_weights(pos_w, conf))
                targets = IR.cam_to_world(t14)
                targets[..., :2] -= targets[:, :1, :2]
                if scales is None:
                    calib.append(targets)
                    scaled, _ = IR.scale_to_robot(robot, link_idx, targets)
                    if len(calib) >= CALIB_FRAMES:
                        _, scales = IR.scale_to_robot(robot, link_idx, np.concatenate(calib))
                        stack, _ = IR.scale_to_robot(
                            robot, link_idx, np.concatenate(calib), scales
                        )
                        floor = float(stack[..., 2].min())
                else:
                    scaled, _ = IR.scale_to_robot(robot, link_idx, targets, scales)
                if conf[[10, 13]].min() < VIS_THR:
                    # feet unseen: stand the robot at its own hip height instead of
                    # deriving the floor from landmarks that were never observed
                    scaled[..., 2] += stand_h - scaled[:, 0, 2]
                else:
                    scaled[..., 2] -= floor
                quat, ow = IR.bone_frames(robot, link_idx, scaled, ARM_FRAMES)
                quat = quat.at[0].set(torso_frame(scaled)[0]).astype(jnp.float32)
                ow = ow.at[0].set(3.0)  # pelvis: the one frame measured directly
                ow = ow * (np.asarray(w) > 0)  # never orient a joint we cannot see
                q, T = solve(jnp.array(scaled[0]), q, T, w, quat, ow)
                q.block_until_ready()
                sim.show(q, T)

                px = res.pose_landmarks[0]  # normalised image coords, for the overlay
                h, w = frame.shape[:2]
                pt = lambda i: (int(px[i].x * w), int(px[i].y * h))
                for a, b in BONES:
                    cv2.line(frame, pt(a), pt(b), (0, 220, 0), 3, cv2.LINE_AA)
                for i in {i for ab in BONES for i in ab}:
                    cv2.circle(frame, pt(i), 5, (0, 0, 255), -1, cv2.LINE_AA)

            fps = 0.9 * fps + 0.1 / max(time.time() - t0, 1e-6)
            cv2.putText(frame, f"{fps:5.1f} FPS", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)
            n_frames += 1
            if n_frames % 30 == 0:
                print(f"{fps:5.1f} FPS", flush=True)
            cv2.imshow("MediaPipe -> robot", frame[:, ::-1] if args.flip else frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cam.stop = True
        landmarker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
