"""Drive the mimic policy in Isaac from a live reference stream instead of a clip file.

Runs inside the eval container. Everything the policy reads comes from
`reference.<array>[frame_ids]`, so a live source does not need a new command term: build
the env from a seed clip to fix the shapes, then overwrite one slot each step and keep
`frame_ids` pointed at it. Clip transitions are disabled — there is no next clip, only
the next frame off the wire.

    ./third_party/IsaacLab/_isaac_sim/python.sh /tools/live_play.py \
        --checkpoint <pt> --seed_clip /motions/take5b/take5b.npz --stream 8100

The stream is `ref_stream.py` on the host (host networking, so plain localhost). Frames
are UDP: the newest one is the only one worth having, and dropping is the right response
to falling behind. When none has arrived the previous reference is held, which reads as
the person standing still rather than as a discontinuity.
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--seed_clip", required=True,
                    help="any valid clip: fixes joint/body names, fps and tensor shapes")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--port", type=int, default=9411)
parser.add_argument("--stream", type=int, default=None, metavar="PORT",
                    help="serve the viewport as MJPEG so the run can be watched live")
parser.add_argument("--seconds", type=float, default=0.0, help="0 = until interrupted")
parser.add_argument("--render_every", type=int, default=3,
                    help="render one frame in N. Rendering every step costs more than the "
                         "physics does and drops the loop below the policy rate, which "
                         "shows up as reference frames being skipped, not as a slow video.")
parser.add_argument("--idle_reset", type=float, default=0.0,
                    help="reset if no frame arrives for this long [s]; 0 disables")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
if args.stream:
    args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import socket
import time
from importlib.metadata import version as _pkg_version

import gymnasium as gym
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

import cyclo_lab.simulation_tasks  # noqa: F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg

TASK = "Cyclo-Mimic-K1-Rev1-Multi-Play"

agent_cfg = handle_deprecated_rsl_rl_cfg(
    load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point"), _pkg_version("rsl-rl-lib"))
env_cfg = parse_env_cfg(TASK, num_envs=args.num_envs)
env_cfg.commands.reference_trajectory.trajectory_files = [args.seed_clip]
env_cfg.commands.reference_trajectory.transition_hub_clips = None
env_cfg.commands.reference_trajectory.transition_target_weights = None
env_cfg.commands.reference_trajectory.transition_probability = 0.0
env_cfg.observations.policy.enable_corruption = False
env_cfg.events.push_robot = None
env_cfg.episode_length_s = 1.0e9
env_cfg.scene.num_envs = args.num_envs

raw_env = gym.make(TASK, cfg=env_cfg, render_mode="rgb_array" if args.stream else None)
env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
runner.load(args.checkpoint)
policy = runner.get_inference_policy(device=raw_env.unwrapped.device)

cmd = raw_env.unwrapped.command_manager.get_term("reference_trajectory")
robot = raw_env.unwrapped.scene["robot"]
dev = raw_env.unwrapped.device

# The loader reindexes the file's columns into runtime order, so a frame off the wire --
# which is in the file's order -- has to be put through the same permutation. Recomputing
# it from the same two name lists keeps the two sides from drifting apart silently.
seed = np.load(args.seed_clip, allow_pickle=True)
file_joints = [str(x) for x in seed["joint_names"]]
file_bodies = [str(x) for x in seed["body_names"]]
joint_ids = [file_joints.index(n) for n in robot.data.joint_names]
body_ids = [file_bodies.index(n) for n in cmd.cfg.body_names]
NJ, NB = len(file_joints), len(file_bodies)
FRAME = 1 + 2 * NJ + NB * (3 + 4 + 3 + 3)
print(f"[live] {NJ} joints, {NB} bodies, {FRAME * 4} B/frame, tracking "
      f"{len(body_ids)} bodies", flush=True)

# never hand off, never resample: there is one slot and it is always the newest frame
SLOT = 0
cmd._sample_start_frames = lambda env_ids: cmd.frame_ids.__setitem__(env_ids, SLOT)
cmd._schedule_transition = lambda env_ids: cmd.will_transition.__setitem__(env_ids, False)

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind(("0.0.0.0", args.port))
sock.setblocking(False)

stream = None
if args.stream:
    import sys
    sys.path.insert(0, "/tools")
    from mjpeg import Streamer

    stream = Streamer(args.stream)
    print(f"[live] viewport on :{args.stream}", flush=True)


def newest_frame():
    """Drain the socket and keep only the last datagram; None if nothing arrived."""
    last = None
    while True:
        try:
            buf = sock.recv(65535)
        except BlockingIOError:
            break
        if len(buf) == FRAME * 4:
            last = buf
    return None if last is None else np.frombuffer(last, "<f4")


def apply(f):
    o = 1
    jp = torch.as_tensor(f[o:o + NJ][joint_ids].copy(), device=dev); o += NJ
    jv = torch.as_tensor(f[o:o + NJ][joint_ids].copy(), device=dev); o += NJ
    bp = torch.as_tensor(f[o:o + NB * 3].reshape(NB, 3)[body_ids].copy(), device=dev); o += NB * 3
    bq = torch.as_tensor(f[o:o + NB * 4].reshape(NB, 4)[body_ids].copy(), device=dev); o += NB * 4
    bl = torch.as_tensor(f[o:o + NB * 3].reshape(NB, 3)[body_ids].copy(), device=dev); o += NB * 3
    ba = torch.as_tensor(f[o:o + NB * 3].reshape(NB, 3)[body_ids].copy(), device=dev)
    cmd.reference.joint_position[SLOT] = jp
    cmd.reference.joint_velocity[SLOT] = jv
    cmd.reference._body_position_w[SLOT] = bp
    cmd.reference._body_orientation_w[SLOT] = bq
    cmd.reference._body_linear_velocity_w[SLOT] = bl
    cmd.reference._body_angular_velocity_w[SLOT] = ba


obs, _ = env.reset()
cmd.frame_ids[:] = SLOT
raw_env.unwrapped.command_manager.compute(dt=raw_env.unwrapped.step_dt)
obs = env.get_observations()

# Pace to the policy rate. Left free the loop runs ~1.7x real time, and then the robot
# gets 1.7 seconds of gravity per second of the operator's motion — the reference arrives
# on a wall clock and the physics would not.
step_dt = raw_env.unwrapped.step_dt
t0 = time.time()
got = miss = resets = late = 0
last_rx = time.time()
try:
    with torch.inference_mode():
        while args.seconds <= 0 or time.time() - t0 < args.seconds:
            f = newest_frame()
            if f is not None:
                apply(f)
                got += 1
                last_rx = time.time()
            else:
                miss += 1
            cmd.frame_ids[:] = SLOT  # the command advances it; there is nowhere to advance to
            action = policy(obs)
            obs, _, dones, _ = env.step(action)
            if dones.any():
                resets += 1
                cmd.frame_ids[:] = SLOT
            if args.idle_reset and time.time() - last_rx > args.idle_reset:
                obs, _ = env.reset()
                cmd.frame_ids[:] = SLOT
                last_rx = time.time()
            if stream is not None and (got + miss) % args.render_every == 0:
                img = raw_env.unwrapped.render()
                if img is not None:
                    stream.put(np.asarray(img)[..., ::-1])
            slack = (t0 + (got + miss) * step_dt) - time.time()
            if slack > 0:
                time.sleep(slack)
            else:
                late += 1
except KeyboardInterrupt:
    pass

el = time.time() - t0
print(f"[live] {el:.1f}s | frames applied {got}, steps with no new frame {miss}, "
      f"resets {resets}, steps behind real time {late}", flush=True)
simulation_app.close()
