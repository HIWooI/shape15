# Real-time RGB → robot motion targets

## Goal

A person moves in front of an RGB camera; the system emits robot joint targets in real
time and a simulator displays the robot following them. **Delay is the objective** — this
is a demonstration, not a robot being driven, so ground contact, self-collision and
velocity limits are cosmetic here and stay off the critical path.

Robots: structure must be robot-agnostic. Both G1 and K1 work.

## End-to-end today

| | skeleton | + robot target |
|---|---|---|
| soma-retargeter, 30-frame chunks | 300 ms (p90 6027) | **970 ms** |
| PyRoki per frame, K1 | 167 ms (p90 200) | **175 ms** |
| PyRoki per frame, G1 | 167 ms (p90 253) | 200 ms |

Measured on the 1080×1920 dance clip; the 640×480 webcam path was 100 ms for the skeleton,
so a webcam demo should land near 110 ms with K1.

Two changes got there. Per-frame IK replaced the chunked retarget, and the IK consumes
**in-camera** joints rather than world-frame ones, which drops GEM's global-trajectory
rollout (35 ms) altogether — a display does not need a world trajectory. The IK worker runs
in its own process on CPU; the round trip costs 0.5–0.9 ms over stdin/stdout, so isolation
is effectively free.

Delay is now almost entirely perception. Profiling the loop stage by stage
(`replay_delay.py --profile`):

| stage | before | after freezing identity |
|---|---|---|
| VitPose | 29.3 ms | 25.9 ms |
| GEM denoiser (`predict`) | 21.0 ms | 23.1 ms |
| SOMA FK | **17.4 ms** | **9.2 ms** |
| IK round trip (incl. IPC) | 8.9 ms | 8.1 ms |
| bbox + make_data + rest | 3.5 ms | 3.2 ms |
| **loop total** | **80.1 ms** | **69.5 ms** |
| skeleton delay / robot target | 167 / 175 ms | **133 / 140 ms** |

`prepare_identity` was rebuilding the person's rest shape and skeleton transfer on every
single frame — 12 ms of a 20 ms FK — although SOMA's own docstring says to call it once per
identity and then just `pose()`. `freeze_identity()` computes it on the first frame:
FK 19.8 ms → 6.2 ms standalone.

Note the reported delay (133 ms) exceeds the loop total (69.5 ms) because it is measured as
a viewer sees it: the loop runs at 12.7 FPS against a 30 FPS display, so each result is
shown for two or three display frames and ages while it is up. Cutting compute is what
moves it.

Remaining levers, in order of size: VitPose 25.9 ms (needs a lighter model), the GEM
denoiser 23.1 ms (its network is only ~7 ms — the rest is preprocessing and decode), FK
9.2 ms (still runs `_warp_skinning` for vertices we discard).

## Where it stands

`demo_webcam.py` runs the perception half live: VitPose (77 SOMA keypoints) → GEM-X
denoiser over a 64-frame sliding window → in-camera SOMA skeleton drawn on the frame.
`replay_delay.py` replays a video through that same loop against a wall clock, so the
delay it reports is what a live viewer would actually see. `robot_target.py` wraps
soma-retargeter for G1/K1.

## Measured (RTX 5090, shared with other jobs, ±20%)

Per-frame compute:

| | stage | cost | blocks the loop |
|---|---|---|---|
| GEM-X | VitPose 2D (DINOv3 ViT-H, fp16, no flip test) | 26 ms | yes |
| | SAM-3D-Body pose token, single frame | 237 ms | no — background thread |
| | GEM denoiser + FK + projection | 30 ms | yes |
| | global trajectory rollout (only needed to retarget) | +35 ms | yes |
| soma-retargeter | G1 IK, 24 iterations | 29 ms | yes, in 30-frame chunks |
| | K1 IK, 180 iterations | 1179 ms | yes |

One-time: model loads 15–40 s, retarget pipeline construction 7.5 s, first chunk's warp
kernel specialisation 8.0 s.

End-to-end delay:

| configuration | skeleton | robot target |
|---|---|---|
| webcam, no retarget (640×480) | 100 ms (p90 133) | — |
| 1080 clip + global rollout + G1 retarget | 300 ms (p90 6027) | 970 ms |

The 970 ms is ~100 ms perception + 870 ms chunk compute. The p90 blow-up is the retarget
stalling the perception loop once per chunk.

Accuracy, against the offline pipeline on the same clip (joint error as % of bbox size):

| configuration | error |
|---|---|
| full window + image features (offline reference) | 0.71 % |
| streaming + image features, token ~7 frames stale | 4.82 % |
| streaming, no image features | 7.95 % |

Image features are the dominant accuracy factor and matter most when the body is cropped
(waist-up webcam framing), because 2D keypoints alone cannot fix scale and depth.

## Decision: replace the retargeter, keep the perception half

The IK computation was never the problem — 29 ms/frame for G1 is already real-time. The
870 ms is soma-retargeter's whole-motion API (`add_input_motions` + `execute`, minimum 30
frames), and K1's 1179 ms is its objective stack (180 iterations plus dozens of
AI-Sapiens-specific objectives). A general-purpose per-frame differential IK has neither
problem: it warm-starts from the previous frame and solves one frame at a time.

Plan A — swap in per-frame differential IK ([PyRoki](https://pyroki-toolkit.github.io/),
or [mink](https://kevinzakka.github.io/mink/) if we stay in MuJoCo). Projected budget:
56 ms perception + a few ms IK ≈ 60–100 ms end to end.

Considered and deferred:

- **B, learned retargeting.** [NMR](https://arxiv.org/pdf/2603.22201) already does this for
  G1: RL experts repair human motion onto the robot's feasible manifold, a CNN-Transformer
  learns the mapping. Go here if A's quality is not good enough.
- **C, skip retargeting.** [H2O / OmniH2O](https://github.com/LeCAR-Lab/human2humanoid) feed
  human keypoints straight into an RL policy. Rejected for now: the output is robot actions,
  not motion targets, which is not what we want to hand downstream.

Closest published system to what we are building:
[MIRROR](https://arxiv.org/abs/2603.23995) — visual skeleton estimation → GPU-parallel
continuation-based differential IK with control-barrier-function self-collision avoidance,
real-time upper-body teleoperation on real hardware.

## Known blockers

- **Warp and torch cannot share a process across threads.** The G1 retarget captures a CUDA
  graph (`newton_pipeline.py:2322`, `feet_stabilizer.py:241`); torch work on the legacy
  stream meanwhile raises `cudaErrorStreamCaptureImplicit`. The SAM-3D-Body worker thread is
  the trigger — with `--no_imgfeat` the retarget is stable. Warp exposes no global switch,
  so decoupling needs a separate process. (Moot if we leave warp behind with plan A.)
- **Constraint porting is the real work, not the solver.** soma-retargeter's K1 config holds
  joint limits, a feet stabiliser and limb-bend priors that someone tuned. A fresh IK setup
  has to re-establish those. Skip ground contact and the robot skates.
- **Window edge.** The newest frame of the sliding window has no future context and is the
  least accurate — and it is exactly the one a robot would follow. A 2–4 frame lookahead
  buffer (+100 ms) recovers most of the gap; for a trajectory target that is a fair trade.
- **SOMA compatibility.** The shape10 K1 toolchain is SOMA-77 throughout. Replacing the pose
  estimator would break that; replacing only the retargeter does not.

## The rest-pose test: where the 45° actually comes from

`test_apose.py` feeds SOMA's own rest pose (`fixtures/soma_rest14.npy`, symmetric to
1.2 mm) through the IK. A known, still input separates IK bugs from perception noise, and
a symmetric input must give a symmetric solve.

**The IK is not mirrored wrong.** Left/right deviation is ~0 on every joint except the
elbow (8.9°). No left/right mapping bug.

**But a person standing in a T-pose comes out contorted**, with a 268 mm mean target error:

```
hip_roll        L -23.4   R  23.5     legs splayed 23° each
ankle_roll      L  23.5   R -23.5     ankles cranked back to compensate
shoulder_yaw    L -88.9   R  87.6     shoulders twisted 90°
wrist_roll      L  82.7   R -82.4     wrists twisted 82°
```

Those are the same joints that carried the error against the soma-retargeter baseline, so
this is the same defect seen without perception noise in the way. Two causes:

- **The splayed legs are a scaling gap.** We match bone *lengths* to the robot but never
  hip *width*. SOMA's hips sit at ±0.120 m; if K1's hip spacing differs, the legs are the
  right length attached in the wrong place, and roll opens them to reach. The equal and
  opposite ankle roll is the solver paying that back.
- **The 90° shoulder and wrist twist is `bone_frames` overreaching.** Aligning the rest
  bone direction onto the observed one also pins the twist axis, which positions never
  determined. It picks a value, and the value is wrong.

Fix order: put hip width into `scale_to_robot` the same way bone lengths are measured, then
lower the `bone_frames` weights or leave the twist axis free. `test_apose.py` is the
regression check — target error and the shoulder twist should both fall.

## The accuracy baseline we finally measured

soma-retargeter run on the same 60 frames, its joint angles compared against ours by name
(its CSV is in degrees, ours in radians — that bit me once):

| configuration | mean | median | p90 |
|---|---|---|---|
| points only | **44.8°** | 31.6° | 142.7° |
| + SOMA rotations | 54.6° | 39.9° | 110.7° |
| points + 180° yaw flip | 51.4° | 31.0° | 167.3° |

45° mean error confirms the "accuracy is terrible" judgement with a number. Two things
that number tells us:

**The error lives in the twist DOFs.** Worst joints, every configuration: `waist_yaw`
(170°), `shoulder_yaw` (138–156°), `hip_yaw`, `wrist_roll`. Those are exactly the degrees
of freedom a position-only IK cannot determine — and exactly what soma-retargeter's
discarded objectives (wrist roll nullspace, elbow branch hints, limb plane normal) were
built to resolve. The pitch-like DOFs that positions do determine are not the problem.

**It is not one convention bug.** A 180° yaw flip was the obvious suspect — a person
facing the camera faces backwards in the robot's frame — but flipping it moved the error
from `waist_yaw` to `hip_yaw` and made the mean worse. The error is distributed, not a
single sign.

So the honest ranking of remedies: port the specific objectives that resolve twist, or
follow H2O and stop tracking joints whose twist we cannot observe (it uses 8 keypoints,
OmniH2O 3, against our 14). Which is right depends on whether the policy cares about twist
accuracy — worth asking before building either.

## SOMA joint rotations: wired, not confirmed

`soma.pose()` computes a world transform for every joint and returns only its translation
column (`joints = T_world[..., :3, 3]`), so the rotations — the one thing positions cannot
give — were being thrown away. `capture_joint_rotations()` keeps them, referenced to the
A-pose (a zero-pose FK, not the first frame, which would make the robot's rest pose
whatever the person happened to be doing at startup). `soma_frames()` maps the A-pose→now
delta onto each robot link's rest orientation, the same convention-free trick as
`bone_frames()` but carrying the twist.

**It changes the solve a great deal and has not been shown to improve it.** Same dance
clip, with vs without:

| | vel p99 | vel max | jerk p99 | IK |
|---|---|---|---|---|
| points only | 10.80 | 20.90 | 3.79 | 7.8 ms |
| + SOMA rotations | 10.70 | 20.90 | 3.99 | 5.3 ms |

Joint angles differ by 60° on average, concentrated exactly where predicted — 241° on
`right_shoulder_yaw`, then `left_hip_yaw`, `left_shoulder_roll`. Those are the twist DOFs
that positions leave free, so the mechanism works. But smoothness did not improve, and
rendering both references (`render_reference.py`) shows the rotation version is not
visibly better and arguably worse. A 241° swing also smells like a convention error
somewhere in the delta.

So it ships **off by default**, behind `--soma_rot` on both demos. Settling it needs the
comparison against soma-retargeter's own output that keeps being deferred — without a
baseline there is no way to say which of the two is closer to right.

## Policy reference output

The downstream whole-body policy (TWIST-derived, K1, 23 DOF) observes:

| field | dim | source |
|---|---|---|
| `motion_command` | 46 | 23 reference joint positions + 23 reference joint velocities |
| `motion_anchor_ori_b` | 6 | reference torso orientation vs the robot's, first two columns of R |
| `base_ang_vel`, `joint_pos_rel`, `joint_vel_rel`, `last_action` | — | the robot's own IMU/encoders, not ours |

Three consequences. **No root position is required**, so the floor calibration and hip
re-centring only matter for the display. **Velocity is ours to produce**, and differencing
divides target jitter by dt — a branch flip in the IK became 55 rad/s in the first
version. **Physical feasibility is the policy's job**, so ground contact, self-collision
and balance leave our critical path.

`motion_command.py` handles it: one-euro filter on the joint angles, resample to the policy
rate by interpolating (holding the last value instead would alternate zero velocity with
spikes), clamp each step to the URDF's own joint velocity limits, then difference. Emitted
via `replay_delay.py --motion_command out.npz`, and `ik_server.py --motion_command`. Run
`motion_command.py` directly for its self-check.

On the 17 s capture: 860 steps at 50 Hz, velocity median 0.04, p99 5.7, max 20.9 rad/s —
the max is now a joint's own limit rather than an artefact. The clamp bounds the damage
from IK branch flips, it does not remove them; the fix for that is orientation targets
from GEM-X's own joint frames.

## Options

Every path built so far stays selectable; nothing is replaced in place.

| what | how | notes |
|---|---|---|
| perception: GEM-X | `demo_webcam.py [--robot g1\|k1]` | 58 ms, GPU, SOMA 77 joints, temporally smooth; weak on cropped bodies |
| perception: MediaPipe | `demo_mediapipe.py --robot g1\|k1` | 22 ms, CPU, 33 landmarks, built for webcam framing |
| image features | `--no_imgfeat` to disable (GEM-X path) | on by default; 7.95 % → 4.82 % joint error |
| retarget: PyRoki | default in both demos | per-frame, 6–18 ms |
| retarget: soma-retargeter | `replay_delay.py --retarget g1\|k1` | the baseline: 30-frame chunks, 870 ms |
| robot | `--robot g1` or `--robot k1` | one entry in `ROBOTS` adds another |
| 3D skeleton view | `--view3d` (GEM-X path) | open3d window |
| offline delay measurement | `replay_delay.py --ik g1\|k1 [--profile]` | side-by-side video + per-stage breakdown |
| IK exchange: non-blocking | `demo_webcam.py` (live, always) / `replay_delay.py --ik_async` | drops a frame rather than stall perception |
| IK exchange: blocking | `replay_delay.py --ik` without `--ik_async` | the baseline the 13× is measured against |
| worker renders during a replay | `replay_delay.py --ik_render PORT` | reproduces the stall the live worker causes |
| re-calibrate mid-session | **Calibrate** button on the page, or `GET /calibrate` | T-pose, ~15 frames |
| twist nullspace | on by default; `ik_server.py --no_twist_null` to disable | K1 13.1°→8.0°, G1 10.6°→7.9° on dance; neutral on low-motion |
| distillation labels | `make_labels.py a.mp4 b.mp4 --out labels.npz --robot g1\|k1` | streaming inputs + teacher outputs |
| distillation | `distill.py labels.npz --features pose\|points` | MLP vs the IK, held out by clip |
| retarget: distilled MLP (G1) | `demo_webcam.py --mlp models/g1_retarget_mlp_v2.pt` | 11.5° vs IK 19.6°, ~17 Hz |
| retarget: distilled MLP (K1) | 재학습 대기 — 구 모델은 오염 라벨 기반이라 삭제됨 | — |
| retarget: kp2d student | `demo_webcam.py --kp2d_mlp models/g1_kp2d_student.pt` | 15.6°, ~30 Hz — no GEM model at all |
| refit joint offsets | `make_offsets.py labels.npz` | re-run after any solver change |

## Known limitations

**We solve positions only.** `make_solver` builds its targets with `SO3.identity()` and
passes `ori_weight=0`, so the solver sees 14 *points*. soma-retargeter's `ik_map` carries an
`r_weight` for every joint that we are not using. Consequences: elbow branch and limb roll
are undetermined, which is why an arm can fold inward while still hitting the wrist target.
GEM-X does produce joint orientations — that information is available and discarded.
MediaPipe does not produce them at all.

**K1 suffers most.** Beyond the above: 23 DOF with no wrist yaw, so hand orientation has
little to act through; its ik_map targets `*_g1_proxy` links that exist only in the MJCF
patch, and we substituted same-named real URDF links, which moves the targets; and K1's
soma-retargeter config carries dozens of robot-specific objectives (elbow branch hints,
limb bend angle, limb plane normal, wrist roll nullspace) over 180 iterations, all of which
we dropped. That stack exists because a naive position IK on K1 looks poor. Measured error:
K1 31 mm vs G1 20 mm.

## Crossed legs, diagnosed from a recording

`record_input.py` captures raw webcam frames so a failure seen live can be replayed. The
capture that showed crossed legs turned out to have two causes, neither of them the leg
tracking itself:

- **The subject was seated**, legs folded and occluded by a chair. VitPose's leg keypoints
  were unreliable, and the GEM-X path had no confidence weighting — so those guesses drove
  the robot at weight 30 and the legs crossed. The worker now takes a confidence per joint
  (VitPose's own per-keypoint score, `kp2d[SOMA_IDX, 2]`) and weights targets by it, the
  same mechanism the MediaPipe path already used.
- **Several people were in frame.** Tracking is a bbox grown from the previous frame's
  keypoints with no identity, so it latched onto a seated person in the middle rather than
  the nearest one. Unfixed — it needs a detector with tracking, and YOLOX is unavailable
  here because onnxruntime is broken.

## Cropped framing (waist-up webcam)

Two bugs, not one, and neither was upper-body tracking itself — the landmarks on the
visible half were fine all along.

**Invisible joints were still driven.** MediaPipe emits landmarks for limbs outside the
frame — the model's guess at where they probably are — and the ik_map weights feet at 30,
the highest of any target. The solver chased invented feet and folded the robot onto the
floor. `target_weights()` now scales each target by its landmark visibility, so unseen
joints fall out of the cost and the rest term holds them in a neutral pose. When the feet
are unseen the root height comes from the robot's own rest hip height rather than from a
floor derived from landmarks nobody observed.

**The torso leaned.** A hip position target is a point and constrains no rotation, so the
solver could tilt the whole robot while still meeting it. `torso_frame()` builds a pelvis
orientation from two directions we can actually measure — up along hips→chest, left along
right-hip→left-hip — and feeds it as the one orientation target. The solver takes target
rotations and per-joint orientation weights now, so more can be added where they can be
derived.

Result on the waist-up clip: the robot stands upright and raises both arms with the person
(`outputs/mp_g1_upperbody.mp4`). Cost: IK 8 → 25 ms, still ~22 FPS end to end.

**Limb twist was free, so arms folded inward.** Hitting shoulder, elbow and wrist targets
still leaves the rotation about the upper arm unconstrained, and the solver would settle
into an anatomically wrong branch. `bone_frames()` supplies an orientation per limb link
without needing to know which local axis runs along the bone: it takes the rotation
carrying the link's *rest* bone direction onto the observed one and applies it to the
link's rest orientation, which pins the two observable degrees of freedom and leaves the
unobservable twist free. Applied to forearms, hands, shins and feet, and only where the
landmark was actually seen. Full-body dance clip: `outputs/mp_g1_dance.mp4`, 27 FPS.

Depth would not have fixed this. Joints outside the frame are a field-of-view problem, and
an RGB-D camera cannot see them either. Depth does help elsewhere — metric scale (removing
the bone-length calibration) and the limb-toward-or-away ambiguity behind the elbow-branch
problem — but rotations stay unobservable from points regardless, so orientation has to
come from a parametric model (GEM-X has it and we discard it) or from derived frames like
`torso_frame()`.

## The demo

```
camera RGB ──► VitPose + GEM ──► SOMA 14 joints ──► PyRoki IK ──► MuJoCo
   ~0 ms          58 ms          (camera frame)       8 ms        display
└─────────── GEM-X venv, GPU ───────────┘   └──── .venv-ik, CPU ────┘
```

    GEM-X/.venv/bin/python demo_webcam.py --flip --robot g1

`demo_webcam.py --robot g1|k1` spawns `ik_server.py --robot X --view`, which holds both
the IK and the MuJoCo window; perception sends 14 camera-frame joint positions per frame
over a pipe and the worker does the frame conversion, scaling, solve and display. Live on
the webcam it holds 11–25 FPS; the replay harness, which has a person in frame throughout,
measures the loop at 69.5 ms (14.4 FPS) with a 140 ms robot-target delay.

Both robots work end to end: 29/29 joints mapped for G1, 23/23 for K1
(`outputs/sim_g1.mp4`, `outputs/sim_k1.mp4`).

Note the frame convention — camera is x-right/y-down/z-forward, the robot world is
x-forward/y-left/z-up. It is an axis permutation, not a sign flip; getting it wrong lays
the robot on its side below the floor and renders an empty scene.

## Plan A result: it holds, for both robots

`ik_retarget.py --robot g1|k1` (in `.venv-ik`, JAX **CPU**). Adding a robot is one entry in
`ROBOTS`: a URDF path and its ik_map. K1 differs from G1 only at the wrists — 23 DOF, no
wrist yaw, so `wrist_roll_rubber_hand` stands in. The `*_g1_proxy` link names in K1's
retargeter config exist only in its MJCF patch, not the URDF, but the URDF carries the
same `*_link` naming as G1, so the map transfers directly.

| | G1 (29 DOF) | K1 (23 DOF) |
|---|---|---|
| soma-retargeter | 29 ms/frame + 870 ms chunk | **1179 ms/frame** |
| PyRoki per frame | 8.3 ms median, p90 38 ms | **6.6 ms median, p90 14 ms** |
| sustained | 81 FPS | 121 FPS |
| mean target error | 20 mm | 31 mm |

K1 went from 1179 ms to 6.6 ms — **180×**. Its 180-iteration objective stack was never
inherent to the robot; a general solver on 23 DOF is simply cheap, and more consistent
than G1 (p90 14 ms vs 38 ms) because it has fewer joints to resolve.

Per-joint error: hips, chest and feet land at 0–1 mm on K1 (0–14 mm on G1), hands 20–28 mm
(4–8 mm on G1), shoulders and thighs 50–65 mm. K1's larger residual is the missing wrist
DOF and its shorter arms.

### Earlier attempt, kept as a warning

Porting soma-retargeter's `joint_scales` (torso 0.80, arms 0.85, legs 0.86) made accuracy
*worse* — 61 mm to 77 mm. Those are absolute ratios tuned for a 1.8 m human and are meant
to be applied together with the config's `joint_offsets`. Measuring each bone off the
robot's own rest-pose FK instead needs no config, self-calibrates to whoever is in front of
the camera, and took the error to 20 mm.

## Original plan A result (single uniform scale)

`ik_g1.py` (in `.venv-ik`, JAX **CPU**) solves SOMA → G1 one frame at a time, warm-started
from the previous solve. Correspondences and weights come from soma-retargeter's own
`ik_map`; a single leg-length ratio scales the human onto the robot.

| | soma-retargeter | PyRoki per-frame |
|---|---|---|
| compute | 29 ms/frame (GPU) | **18 ms/frame median**, p90 40 ms (CPU) |
| latency added | **870 ms** (30-frame chunk) | **18 ms** — no chunking |
| first call | 8.0 s warp kernel specialisation | 2.0 s jit compile |

Target error, 120 frames of dance:

| joint group | weight | error |
|---|---|---|
| Hips, both Feet | 30 | 1–3 mm |
| Hands | 2.0 | 14–18 mm |
| Forearms, shins, thighs | 1.0–1.5 | 48–83 mm |
| Chest, shoulders | 0.7–1.5 | 131–148 mm |

The solver nails everything it was told to weight heavily, and the end effectors that
matter for teleoperation are within 20 mm. The 130–150 mm at chest and shoulders is the
single uniform scale: one ratio cannot match leg length and torso proportions at once.
soma-retargeter solves this with a per-segment `human_robot_scaler_config` — porting that
is the next accuracy step, not a solver problem.

Running on CPU is a feature, not a limitation: the IK no longer contends with perception
for the GPU, and the warp/torch CUDA-graph conflict disappears with warp.

Projected end to end: 56 ms perception + 18 ms IK ≈ **75 ms**, against 970 ms today.

## Next steps

1. Port per-segment scaling from `human_robot_scaler_config` — should pull chest and
   shoulders in from ~140 mm.
2. Add orientation targets (the ik_map carries `r_weight`; we currently solve positions only).
3. Port the constraints that soma-retargeter earns its keep with: joint limits are in, but
   velocity limits, self-collision and ground contact are not. Without contact handling the
   robot will skate.
4. Wire it into the live loop as a separate process and re-measure end to end.
5. Then K1: same code path, only the URDF and ik_map change.

---

# Where this stands (end of session 2026-08-13)

## Done

**Latency: 970 ms → 140 ms** for the robot target.
- Per-frame PyRoki IK replaced soma-retargeter's 30-frame chunks. K1 went 1179 ms → 6.6 ms.
- The IK consumes in-camera joints, which drops GEM's global-trajectory rollout (35 ms).
- Removed 220 ms of pure waste found by profiling: identity rebuilt every frame (13 ms),
  the empty caption re-encoded every frame (30 ms), the unused global rollout (35 ms), and
  SAM-3D-Body running its decoder twice (142 ms).
- The IK runs on CPU in its own process: no GPU contention, and warp's CUDA-graph conflict
  with torch disappears with warp.

**Accuracy: 44.8° → 22.0° → 8.0° (K1) / 7.8° (G1)** mean joint error against
soma-retargeter, on the dance clip with streaming inputs.
- The orientation targets I added were turning the legs backwards (`bone_frames` drove
  hip_yaw to −180°). Off by default now: rest-pose error 268 mm → 64 mm.
- Per-joint constant offsets calibrate the remaining convention gap, applied only to the
  13 of 23 joints where a constant actually explains the error.

**Robustness.**
- Landmark/keypoint confidence weights every IK target, so limbs the estimator could not
  see stop dragging the robot around.
- Camera reconnects instead of ending the demo; re-detection costs one VitPose pass
  instead of three (it was a 1.7 s freeze on every track loss).
- The IK exchange is non-blocking, so a slow worker can no longer stall perception.
  **Verified** — see below.
- onnxruntime's CUDA lookup fixed, which unblocks YOLOX+ByteTrack (identity tracking, 62
  ms/frame) and the ONNX/TensorRT paths. ONNX measured *slower* than our PyTorch fp16, so
  that door is open but not worth walking through.

**Delivery.** Browser streaming (`--stream 8080`) puts both panels on one page with no X
server; MuJoCo renders offscreen. `--mirror` makes the robot behave like a reflection.

## Non-blocking IK: verified, 13× on the freeze

`replay_delay.py` could not have shown this before — it had its own blocking write/read and
never touched `IKLink`, and it launched the worker without a display, so the render that
does the stalling was not in the picture. It now takes `--ik_async` (route through
`demo_webcam.IKLink`, the live path) and `--ik_render PORT` (let the worker render MuJoCo
and stream, as it does live), and always reports the frame gap — the interval the loop went
without producing anything, which is what a viewer sees as a freeze, as opposed to the delay
figure, which is how *old* what they see is.

Same clip (`outputs/input_capture.mp4`, the 601-frame webcam capture behind
`baseline_profile.txt`), same worker rendering, back to back under the same Isaac contention:

| | blocking | non-blocking |
|---|---|---|
| **frame gap max** | **4017 ms** | **308 ms** |
| frame gap p99 | 295 ms | 100 ms |
| frame gap median | 92 ms | 62 ms |
| skeleton delay median / p90 | 167 / 2140 ms | **100 / 133 ms** |
| effective rate | 8.6 FPS | **16.1 FPS** |
| `ik` stage inside the loop | 15.9 ms | 0.1 ms |
| loop total | 88.1 ms | 55.3 ms |

The freeze is gone: max gap 4017 → 308 ms, and the whole tail moves with it (p99 295 → 100),
so it was never one unlucky spike. Note the blocking column is worse than the 1667 ms
previously seen live — a rendering worker stalls harder than a headless one, which is the
point of measuring with `--ik_render`.

The cost is 31 of 322 frames dropped (9.6%) because the worker was still busy, so the robot
updates at ~14.5 Hz against perception's 16.1. That is the trade the change makes on purpose:
the robot may lag a frame, the camera never stops. What is not a trade is the loop itself
getting cheaper — the `ik` stage falls from 15.9 ms to 0.1 ms because waiting for the reply
*was* the cost, and that alone is most of 88.1 → 55.3 ms.

## Calibration is a deliberate act now

A **Calibrate** button on the streamed page → `GET /calibrate` → the perception loop clears
the frozen SOMA identity and prompts the operator to hold a T-pose, and the worker
re-measures bone scales and the floor over its next 15 frames. Bone lengths measure worst
on bent limbs, so what the session opens on is the worst thing to be stuck with.

The request crosses a process boundary over a fixed-size binary pipe. It travels as the
**sign bit of the first confidence value** — the protocol is unchanged, the value itself
survives (`abs()` on the far side), and nothing has to be framed. Perception sends it once,
retrying until a submit actually lands, because `IKLink` drops frames when the worker is
behind. `test_calibrate.py` covers it: drive a real worker, switch to a 1.4× person, and the
robot pose must be unchanged.

| | worst joint vs the 1.0× reference |
|---|---|
| after Calibrate | **3.6°** |
| without it (stale scales) | 147° |

Three things this cost, all of them the same lesson — *a calibration must not disturb the
solver*, because this IK never leaves a branch it falls into:

- **Do not clear `floor` while re-measuring.** It lifts the robot ~0.7 m for the window and
  the solver never recovers. Keep the old floor until the new one is ready.
- **Do not hold the previous scales either.** A differently-sized person is then out of
  reach, which is worse. Scale off each frame's own bones during the window.
- **Do not reset the solver's warm start.** "A calibration starts a session, so start from
  the rest pose" sounds right and measures worse: 11.7° against 3.6° for leaving it alone.

`test_apose.py`'s symmetric rest pose is the wrong fixture for this. It is exactly the
branch-degenerate configuration it reports 148° of asymmetry on, where a 1e-6 difference in
the targets — float32 rounding between a 1.0× and a 1.4× person — flips the solve by 126°.
Everything above is measured on a fixed asymmetric perturbation instead.

Startup detection stays at three passes. The plan expected to drop it to one once
calibration was deliberate, but that saves ~50 ms of a 50 s startup — the 1.7 s figure was
the mid-stream re-detect, which is already at one pass.

## We finally looked at the teacher, and it paid for itself

The 22° was always measured *against* soma-retargeter, and nobody had checked whether
soma-retargeter is right. `outputs/teacher_vs_ours.mp4` renders both on the same 60 frames.
It is right, and the way it is right is specific and actionable — its **twist DOFs barely
move**:

| joint | soma-retargeter | ours (before) |
|---|---|---|
| `left_hip_yaw` | −14° … 25° | 52° … **165°** |
| `waist_yaw` | −4° … 11° | **−106°** … 11° |
| `left_shoulder_yaw` | −11° … 3° | **−141°** … 5° |
| `right_shoulder_yaw` | 3° … 14° | −117° … **72°** |
| `left_wrist_roll` | −23° … 13° | −65° … **83°** |

The teacher holds them in a 10–40° band. We were swinging ±150°. That is not a subtle
prior to be learned — it is "do not let the twist wander", which is what its wrist-roll
nullspace and limb-plane objectives were for, and it is a rest weight.

### Pinning the twist

`rest_cost` already pulled every joint toward zero at a uniform 0.01. Its `weight` argument
broadcasts elementwise, so the change is a per-joint vector: 0.01 everywhere, more on the
twist DOFs (`*_yaw_joint` and `*_wrist_roll_joint`, name-based).

**The weight is per robot, and the first version of this section overstated the win.** That
version measured K1 only, on the dance clip, against a baseline built from *offline
full-window* perception — not the streaming inputs the live path produces — and asserted
the name rule "transfers to G1" from naming alone, without measuring G1 at all. Redone with
`make_labels.py`, so both robots are scored against their own teacher on matched streaming
inputs (mean joint error after the constant offsets):

| twist weight | K1 dance | G1 dance | K1 webcam | G1 webcam |
|---|---|---|---|---|
| 0.01 (off) | 13.1° | 10.6° | 6.6° | 2.6° |
| 0.05 | 8.7° | **7.9°** | 6.3° | 2.6° |
| 0.10 | **8.0°** | 9.7° | 6.2° | 2.8° |
| 0.30 | 9.3° | 11.1° | 6.0° | 2.6° |

So `TWIST_REST_W = {"g1": 0.05, "k1": 0.1}`. Two things the single-clip version missed:

- **The optimum differs by robot.** G1 is past its best at 0.1, where K1 is at its best.
  G1's wrist yaw gives it somewhere else for twist to go.
- **The win is content-dependent, not universal.** It pays where the twist actually
  wanders — the dance clip, K1 13.1° → 8.0° and G1 10.6° → 7.9°. On the low-motion webcam
  capture the twist joints already behave and it buys nothing (6.6° → 6.2°, 2.6° → 2.8°)
  while costing a little target accuracy. The honest claim is "large on dynamic motion,
  neutral otherwise", not the flat 3× the first version implied.

Offsets are now fitted by `make_offsets.py` from the same labels, for both robots — **G1
had no offsets fixture at all before this**. K1 15.8° → 8.0° (22/23 joints), G1 13.3° →
7.8° (29/29). Re-run it whenever the solver changes; the offsets are fitted to a solve.
`--no_twist_null` restores the uniform weight.

The rest-pose test still improves a lot: 148.1° max asymmetry / 64 mm → **31.8° / 38 mm**,
`shoulder_yaw` deviation 148.1° → 2.5°. The worst asymmetry is now a leg one that was
previously buried — left knee 22.3° against a right knee at 0.0°.

### Calibration on G1: correct, but the solve does not fully recover

`test_calibrate.py` now takes a robot argument, and G1 failed it on first run. Two separate
things were wrong, and only one of them was the code:

**The test was asserting on joint angles, which are not unique.** The twist DOFs are a
nullspace, so two solves can stand the robot in the same place through angles 88° apart —
and G1, with six more DOF, does exactly that. It now compares link positions in the pelvis
frame, and separately asserts the thing calibration is actually answerable for: that the
re-measured scales reproduce the 1.0× targets. That part is exact on **both** robots
(1.8e-7 m on G1, 2.4e-7 m on K1), so calibration itself is correct on G1.

**The solve genuinely does not recover, though.** Feeding identical targets, G1's post-
calibration pose fits the targets worse than the pre-calibration one (397 mm against a
281 mm reference; K1 is unaffected at 379.5 mm either way). Re-converging from the rest
pose when new scales land recovers most of it (397 → 338 mm) and is neutral on K1, so the
worker now does that — but **only on a re-calibration**. Doing it on the opening
calibration too measured worse on G1 (281 → 338 mm): there is no stale pose to escape at
startup, only convergence to throw away.

This also reverses an earlier call in this document. That same warm-start reset was tried
before and rejected as "measures worse, 11.7° against 3.6°" — but that comparison used
joint angles on the symmetric rest pose, where they are neither unique nor meaningful. In
task space it is a gain on G1 and free on K1.

G1 still ends 158 mm from the reference after a re-calibration where K1 ends at 0.1 mm.
The test records that bound as a known defect rather than a pass.

### So is the distillation still worth it?

Less urgent, and better targeted if it happens. A per-joint rest weight plus refitted
offsets put both robots near 8° against the teacher on dynamic motion, so the case for
training a network to imitate the rest is weaker than it looked. What distillation would
still buy is the part a constant offset and a nullspace weight cannot express — an 8°
question now, not a 22° one. Worth asking the policy owners whether 8° is already inside
tolerance before spending label time.

The G1 label pipeline is built either way (`make_labels.py`), because the same teacher
output is what any future comparison is judged against.

### `make_labels.py`

    GEM-X/.venv/bin/python make_labels.py clip.mp4 labels.npz --robot g1 [--limit N]

Writes `joints (N,14,3)`, `conf (N,14)`, `teacher_q (N,ndof)`, `joint_names`, `src`. Two
decisions worth keeping:

- **Inputs are collected the streaming way** — sliding window, last frame, whatever image
  token has arrived — not from an offline full-window pass. Both halves of a pair then sit
  on the same pose estimate, so a student learns the retargeting map rather than the gap
  between offline and streaming perception. That gap is 4.82% vs 0.71% and would dominate.
- **The teacher runs in a second process.** soma-retargeter captures a CUDA graph and the
  SAM-3D-Body worker thread is exactly the torch-on-the-legacy-stream trigger that breaks
  it. A fresh process has no such thread, so image features stay on.

Alignment is asserted, not assumed: the pipeline prepends 10 initialisation and 5
stabilisation frames internally (its progress bar counts n+15) but does not emit them, so
buffer row *i* is input frame *i*. If that ever stops being true the script refuses to
write rather than pairing every input with the wrong label.

Measured on 120 frames of `outputs/input_capture.mp4`: 105 pairs survive the warm-up
window, and the G1 teacher itself runs at ~100 frames/s in one whole-motion call.

## Distillation: measured, and the IK's real number is worse than reported

Done properly this time — 4037 frames over 11 clips (dance, taekwondo, cheer, jumps,
handheld), each scored against its own soma-retargeter output, **held out by clip**. A
frame-wise split is meaningless here: neighbouring frames of one clip are nearly identical.

**First, a correction. The IK's gap to the teacher is ~17°, not the ~8° reported earlier.**
That 8° was the constant offsets fitted *and* evaluated on the same clip. On motion it has
not seen, the same IK reads 17.0° — and the offsets barely earn their place there
(18.1° without them, 17.0° with). They are largely an in-clip fit.

Then the network, same inputs, same teacher, same split:

| | test, held-out clips |
|---|---|
| IK (14 points) | 17.04° |
| MLP on the same 14 points + confidence | 15.38° |
| MLP on the residual over the IK | 17.05° |
| **MLP on SOMA's 76-joint articulation** | **9.35°** |

**The network was not short of capacity or data — it was short of information.** Width
128→1024 moves it 15.7→15.1. Temporal context (t−1 … t−8) moves it 15.4→15.2. Training
clips 6→9 moves it 15.6→15.4. All noise. Feeding it what the *teacher* actually consumes —
`body_pose`, SOMA's 76 joint rotations — takes it to 9.35° in one step.

That is the same fact this document already recorded from the other direction: "GEM-X does
produce joint orientations — that information is available and discarded." We have been
solving from 14 points while the thing we are trying to match reads the whole skeleton.

**It costs nothing live.** `body_pose` is the articulation, already computed every frame in
`body_params_incam`; it is only the *global* root that needs the 35 ms rollout. Sending it
to the worker is 228 more floats on a pipe that already carries 182.

So the honest ranking of what to do next is no longer "port objectives or train a network".
It is: **give the solver the orientations it is missing before doing either.** The IK has
never been tried with SOMA's joint rotations as targets on a settled configuration —
`--soma_rot` exists but was measured back when `bone_frames` was turning the legs backwards
and the twist DOFs were unpinned. A network reaching 9.35° from exactly that input is
evidence the information is sufficient; whether the solver or the network should consume it
is then a latency and robustness question, not an accuracy one.

PPO is not indicated. The teacher is a deterministic function and supervised regression
already fits it; RL would only be needed for physical feasibility, which this document has
already assigned to the downstream policy.

    GEM-X/.venv/bin/python make_labels.py a.mp4 b.mp4 --out labels.npz --robot g1
    .venv-ik/bin/python distill.py labels.npz --features pose|points

## The solver route was tried first, and lost to the network

The plan was: before putting a model in the live loop, feed the solver the same joint
orientations the MLP won with. Measured on the same 4037-frame, clip-held-out split:

**The shipped `soma_frames` constant was upside down.** The A-pose→now delta acts on
SOMA-canonical directions (y-up), and the code conjugated it by the camera→world
permutation, which maps SOMA's up to the robot world's *down*. Verified independently of
any solve: with the corrected alignment the rotation-derived pelvis target agrees with the
position-derived `torso_frame` to **4.4°** (the alternatives read 90° and 180°). The
extraction itself is exact — the delta reproduces observed bone directions to 0.0° and FK
matches the stored joints to 5e-7 m. Fixed in `soma_frames`; the flag stays off by default.

**Corrected frames still barely help the solver.** Best configuration found (pelvis +
chest + feet orientations, weight ×2): held-out 17.09° → 16.35°. On single clips it can
look dramatic (love_attack: 17.5 → 11.3) but it does not generalise, and per-joint
breakdown shows why: orientation targets on the limbs flip legs into 150–170° wrong
branches (the old `bone_frames` disease), and the arms carry a systematic error because
SOMA's A-pose and the robot's arms-down rest differ by 60–90° exactly where the delta is
applied. The mechanism, not the information, is the bottleneck.

Two more measurement corrections along the way:

- The *global*-frame body params are unusable for this: GEM's world rollout re-optimises
  the body, leaving them 40–66 mm and 5–18° away from the incam stream per frame. Only
  incam rotations are consistent with the incam positions the IK consumes.
- The MLP trains to the same 9.5° from incam `body_pose` as from global — so the deployed
  path needs nothing the live loop does not already compute.

## The distilled MLP is live

    GEM-X/.venv/bin/python demo_webcam.py --flip --robot g1 --stream 8080 \
        --mlp models/g1_retarget_mlp.pt

`--mlp` swaps the retarget source: perception runs the distilled net (2×512 GELU, 0.25 M
params, CPU, ~0.3 ms) on the incam `body_pose` and appends the 29 joint angles to the
worker payload; the worker skips the solve and Kabsch-fits only the free base so the
display stands where the targets stand. The IK remains the default and everything else
(calibration, motion_command, streaming) works unchanged in both modes.

| retarget source | vs soma-retargeter, held-out clips |
|---|---|
| PyRoki IK (offsets fitted on train clips) | 16.46° |
| **distilled MLP** (`distill.py`, direct) | **9.48°** |

Known limitation: the MLP has no equivalent of the IK's confidence weighting. When the
body is mostly out of frame the `body_pose` it reads is the estimator's guess, and it
outputs a pose for that guess — the IK would have parked unseen limbs at rest instead.
Seen live with a seated close-up operator. Options if it matters: feed `conf` as input and
retrain, or fall back to the IK when mean confidence drops.

Model card: `models/g1_retarget_mlp.pt` — trained by `distill.py` on `make_labels.py`
output (11 clips, 2518 train / 1519 test frames, test = last 2 clips), input 228-dim incam
body_pose, output 29 G1 joint angles, direct (not residual) head. K1 needs the same two
commands run once (~90 min of teacher time at 1179 ms/frame).

## Prediction: measured, and it does not pay (yet)

The pipeline is causal — the robot shows where the person was ~140 ms ago — so a predictor
that outputs the pose k frames *ahead* looks like the way to buy the latency back. Three
classes measured on the label sequences (30 fps, clip-held-out, same split as everything
else):

**Constant-velocity / constant-acceleration extrapolation of the 14 targets loses
immediately.** One frame ahead (33 ms): holding the last frame is 51 mm off the future,
linear extrapolation 67 mm, quadratic 92 mm. Differencing the perception stream amplifies
its noise faster than it recovers motion. On the smooth teacher joint angles linear wins a
hair at ≤66 ms (2.34° vs 2.45°) and loses beyond — the signal is not the problem, our
noise is.

**A learned predictor loses too.** Same recipe as the deployed MLP, trained to output the
teacher k frames ahead:

| staleness | current MLP shown late | learned predictor |
|---|---|---|
| 66 ms | **10.08°** | 10.79° |
| 99 ms | **10.17°** | 11.74° |
| 132 ms | **10.26°** | 12.73° |

(A 3-frame input window makes it worse still — 13.4° at 66 ms.)

**Why: staleness is nearly free in pose-error terms.** Being 132 ms late costs only
+0.28° on average (10.26 vs 9.98), because averaged over real motion the joints move
slowly; the retargeting gap (~10°) dominates. A predictor pays 1.6–2.5° of prediction
error to recover 0.3° of staleness. Prediction starts to pay only when the base error is
several times smaller than today's, or the motion is much faster than these clips.

One caveat this measurement cannot see: pose error is not perceptual synchrony. A robot
that moves *in time* with the operator but slightly wrong may demo better than one that is
right and 140 ms behind. If the demo feels laggy, the 66 ms predictor (+0.7°) is the
cheapest thing to wire — but the numbers say do not build it until someone actually
complains about the feel.

## Retraining the MLP: more data helps, but it is flattening

The dataset grew from 11 to 15 usable clips (9,396 pairs; three AV1 clips still refuse to
decode even transcoded, and `1품 기본동작` yields no frames — multi-person, track never
settles). The test set is now the *full* held-out clips (4,697 frames), not their 800-frame
prefixes, so numbers moved: the same recipe that read 9.48° on the old test reads 12.06°
here, and the IK baseline reads 19.61°. All numbers below share this split.

Learning curve (pose features, width 1024): 445 frames → 13.3°, 2.5k (the old model's
size) → 12.6°, full 4.7k → **11.48°**. Still descending, but ~0.5° per doubling now.
Width 512 → 1024 is worth 0.6°. Confidence as an extra input is neutral (11.67°) and
input-noise augmentation slightly hurts (11.89°) — the occlusion problem will not be
solved by either alone.

**Shipped: `models/g1_retarget_mlp_v2.pt`** (pose, w1024, 13 train clips, 11.48°). The
v1 checkpoint stays; both load through `--mlp PATH`, and the checkpoint carries its own
feature spec so `pose_conf` models work unchanged.

Two ops lessons from the run: the teacher pipeline does not free its buffers when the
`Retargeter` is dropped, so one process building 18 of them accumulated 48 GB and the
kernel OOM-killed the last clip — `make_labels.py` now runs one subprocess per clip, and
the per-clip `.npy` doubles as a resume point. And `distill.py`'s test split is
`len(clips)//4`, which silently diverged from the 2-clip split every comparison here uses —
watch for that when clip counts change.

## Next lever if the MLP needs to be better: cut into GEM-X itself

The loop is VitPose 26 ms → denoiser 23 ms (network ~7, rest pre/post) → FK 9 ms →
retarget. Three cuts, shallowest first:

- **FK (9 ms)**: in MLP mode it only feeds the display overlay and the base fit; a 2D
  overlay drops it.
- **Denoiser+FK (~32 ms)**: train the student one stage earlier — VitPose 2D keypoint
  window → robot q, absorbing the 2D→3D lift into the network. `make_labels.py` now saves
  per-frame `kp2d` (all 77) precisely so this student can be trained from the next dataset
  pass. This is the experiment to run before touching VitPose.
- **VitPose (26 ms)**: biggest chunk, but breaks SOMA-77 compatibility — last resort,
  as before.

## The 30 Hz cut: profiled, sized, built

The three questions asked of this stage — where does the time go, how big may the network
be, what gets cut — all have measured answers now.

**Full-stage profile** (uncontended GPU, 601-frame replay, worker rendering; the worker
runs in parallel so only the perception column blocks the loop):

| perception (blocks) | ms | worker (parallel) | ms |
|---|---|---|---|
| VitPose 2D | 24.5 | parse+scale | 1.7 |
| bbox | 0.9 | solve / base fit | 0.7 |
| make_data | 0.9 | mj_forward | 0.4 |
| GEM predict | 16.5 (denoiser net 8.3 + decode 3.3 + pre ~5) | render | 1.0 |
| SOMA FK | 6.2 | JPEG+put | 1.5 |
| MLP | 1.2 | | |
| **loop** | **~57 → 17.5 Hz** | round trip | ~5 |

**Network size is a free variable at these rates.** CPU, batch 1: even a 16-frame kp2d
window into a 4096-wide 3-layer net (32 M params) is 2.0 ms; the deployed sizes are
0.03–0.24 ms. 15–30 Hz constrains the *pipeline*, not the network — width and depth
should be chosen on accuracy alone.

**One production lesson worth the retelling: torch's default CPU thread pool.** The
student initially ran at 11 FPS live — 34–60 ms per forward for a 3 M-param net — because
torch claims one intra-op thread per core (24 here) and Isaac training owns the cores, so
the pool thrashes. Capped at 2 threads the same forward is 0.09 ms, ~700× faster. Any CPU
inference sharing this box with Isaac needs `torch.set_num_threads(2)`.

**The cut: the kp2d student** (`--kp2d_mlp models/g1_kp2d_student.pt`). VitPose 2D window
(t, t−2, t−4, t−8, hip-centered/scale-normalized, conf included) → 29 joint angles + the
14 targets as an aux head, so the worker payload — and the worker — are unchanged. The GEM
model and SAM-3D-Body are not even loaded in this mode: 20+ s less startup, half the GPU
memory, and the denoiser/FK stages simply do not exist.

| retarget source | vs teacher (held-out 4,697 fr) | loop | rate |
|---|---|---|---|
| PyRoki IK | 19.61° | ~57 ms | ~17 Hz |
| body_pose MLP v2 | **11.48°** | ~57 ms | ~17 Hz |
| **kp2d student** (w4, 1024×3) | 15.56° | **~28 ms** | **~30 Hz** |

The accuracy/rate trade is real: the student pays 4.1° to absorb the denoiser's 2D→3D
lift. Window 4 beats 1 slightly (15.90 → 15.56°); 8 and 16 add nothing. Both modes stay
selectable — `--mlp` when 17 Hz is fine, `--kp2d_mlp` when the rate matters.

Live under Isaac contention the student mode holds 25–50 FPS (the wall numbers swing with
Isaac's load; the compute medians above are the stable facts).

> **2026-08-14: K1 수치 전면 무효.** 이 문서의 모든 K1 정확도 숫자(IK 28.70°, MLP
> 15.33–13.97° 등)는 soma-retargeter PR#1 시점의 `ai_sapiens` 설정으로 만든 라벨 기준이며,
> 그 설정은 파일 자체가 "G1 IK objective policy를 얹은 실험 후보"라고 밝히고 있고 K1 전용
> feet stabilizer도 프록시 바디도 없었다. 전체 클립 기준 프레임의 **50.4%**에서 왼다리가
> 관절 한계로 밀려 사람과 다른 자세가 나왔다(`outputs/k1_teacher_fix.mp4`).
> 업스트림 PR#3으로 올린 뒤 같은 클립에서 병리가 **1.0%**로 떨어졌다. 오염 라벨과 그로부터
> 파생된 모델·오프셋은 모두 삭제했고 재라벨링 후 다시 측정한다. G1은 영향 없음(설정 무변경,
> 후처리 정상).

## K1, 업그레이드된 교사로 다시 측정 (2026-08-15)

soma-retargeter PR#3으로 재라벨링한 뒤의 값. 이전 K1 수치는 전부 폐기.

| | 교사 대비 (홀드아웃 4,697프레임) |
|---|---|
| PyRoki IK | **26.75°** (오프셋 없이 27.05°) |
| MLP plain 2층 | 14.94° |
| **MLP residual 4블록** | **13.97°** |

교사 품질 회복 확인:

| | 구 라벨 | 새 라벨 |
|---|---|---|
| 병리 프레임(전체) | 26% | **6.6%** |
| 〃 테스트셋 | 26% | **0.5%** |
| 무릎 한계접촉 좌/우 | 45.5 / 41.3% | **19.4 / 19.9%** (대칭 회복) |
| hip_roll std 좌/우 | 55.0 / 18.4 | 33.2 / 18.6 |

**주의 두 가지.**

첫째, 상수 오프셋이 이제 거의 듣지 않는다 — 22/23 관절에서 **7/23**로 줄었고 개선폭도
25.5 → 25.0°뿐이다. 구 라벨에서 오프셋이 잘 들었던 것은 교사의 계통 오차를 상수가
흡수하고 있었기 때문이고, 교사가 고쳐지자 남은 불일치는 상수로 설명되지 않는 구조적
차이다. IK가 26.75°에 머무는 이유이기도 하다.

둘째, `test_apose.py`의 **왼무릎 비대칭(22.3° vs 0.0°)은 그대로다.** 업그레이드로
사라질 것이라 예상했으나 이 결함은 교사가 아니라 **우리 IK 자체**의 것이었다.

### 합성 데이터: 샘플링 방식이 전부다

`make_synth.py`의 기존 방식(관절별 범위를 1.5배로 넓혀 균일 샘플링, `--mode box`)은
**K1에서 무효**였다. 원인은 양이 아니라 라벨 품질 — 분포 밖 자세를 넣으면 교사가 다리를
관절 한계로 밀어내, 우리가 방금 고친 병리가 합성 라벨에서 41.3%로 재발한다.

`--mode interp`를 추가했다: 키프레임을 균일 난수가 아니라 **실제 자세 두 개의 블렌드**로
만든다. 새로운 조합이지만 모든 키프레임이 사람이 실제로 취한 자세의 혼합이라 매니폴드를
벗어나지 않는다. 병리 0.0%.

| 합성 방식 | 병리율 | 학생 오차 (실+합성) |
|---|---|---|
| 없음 (실데이터 2.6분만) | — | 14.26 ± 0.12° |
| box, margin 1.5, 24클립 14.4k | 41.3% | 14.46° (이득 없음) |
| box, margin 1.0, 8클립 4.8k | 10.3% | 14.10° |
| interp, blend 0.35, 8클립 | 0.0% | 11.74 ± 0.07° |
| **interp, blend 0.6, 8클립** | **0.0%** | **10.79 ± 0.07°** |
| interp, blend 1.0, 8클립 | 0.0% | 11.57 ± 0.05° |
| interp, blend 0.35, **24클립 14.4k** | 0.0% | 12.18 ± 0.26° |

두 가지가 반직관적이다. **양을 늘리면 나빠진다** — 8클립(실데이터와 1:1)이 24클립(3:1)을
이긴다. 합성이 실데이터를 수적으로 압도하면 학습이 그쪽으로 치우친다. 그리고 **blend는
중간값이 최적** — 0.35는 원본에 너무 가까워 새 정보가 적고, 1.0은 두 자세의 중점이라
평균 쪽으로 쏠린다.

**K1 최종: 10.71°** (`models/k1_retarget_mlp_res.pt`, res4 w1024, 실 4,699 + interp 4,800).
IK 26.75° 대비 −16.0°.

### 극단 자세 오버샘플링: 실패

남은 오차가 "손을 어깨 위로 든 / 다리를 벌린 / 팔을 뻗은" 자세에 몰려 있어(5.3° 대
21.4°), `make_synth.py`에 `--bias extreme`을 넣어 그런 실제 프레임을 키프레임으로
과표집했다. 상위 10% 프레임이 가중치의 76%를 가져가게 하니 **전체 10.79 → 15.50°로
악화**했고, 정작 목표였던 어려운 자세에서도 20.72 → 27.94°로 더 나빴다. 합성 집합이
극단으로 쏠리면서 흔한 자세의 표현이 무너진 것으로 보인다. 옵션은 남겨두되 기본은
`--bias uniform`.

### 부위별 전문가 구조: 정확도 이득 없음

다리(12) / 팔(10) / 몸통(1) 관절을 나눠 각각 res 전문가를 두고 상위 융합층으로 합치는
구조를 시험했다. 전부 시드 편차 안이다:

| 구조 | 오차 | 파라미터 |
|---|---|---|
| 단일 res4 w1024 | 11.74 ± 0.07 | 8.7M |
| 부위별 3개 w512 + 융합 | 11.68 ± 0.06 | **5.9M** |
| 부위별 3개 w512, 융합 없음 | 11.72 ± 0.05 | 5.1M |
| 부위별 3개 w1024 + 융합 | 11.78 ± 0.07 | 22.8M |

정확도로는 의미 없지만 **부위별 w512가 파라미터 32% 적게 동등하다** — 엣지 배포에서
메모리가 빡빡하면 쓸 카드. 융합층 자체는 −0.04°로 무의미하다.

## Next, in order

1. **K1 distilled model** — the teacher labels are generating now (perception is
   robot-agnostic, so only the K1 teacher re-runs: ~3 h at 1179 ms/frame). Then
   `distill.py` for the body_pose model and the kp2d student, same splits.
3. **MLP robustness to occlusion** — feed per-joint confidence into the net and retrain,
   or fall back to the IK below a confidence floor (see the limitation above).
4. **G1 does not fully recover from a re-calibration** — 158 mm from the reference where
   K1 lands at 0.1 mm. Calibration hands it the right targets; the solver keeps a worse
   minimum. Bounded in `test_calibrate.py g1`.
5. **The left-knee asymmetry** on the rest pose (22.3° against 0.0°), previously masked by
   the twist noise.
6. **YOLOX identity tracking**, now unblocked. Needed in any room with more than one person.
7. **Perception cost** (VitPose 26 ms + denoiser 23 ms) only if the delay must go lower.
   Fast SAM 3D Body claims 10.9× on the model we already run.

## Regression checks

    .venv-ik/bin/python test_apose.py           # 38 mm, 31.8 deg max asymmetry
    .venv-ik/bin/python motion_command.py       # filter/velocity self-check
    .venv-ik/bin/python test_calibrate.py k1    # re-measures exactly: 0.1 mm vs 77 mm
    .venv-ik/bin/python test_calibrate.py g1    # targets exact; pose recovery is the open defect
    GEM-X/.venv/bin/python replay_delay.py <clip> out.mp4 --ik k1 --profile
    # freeze check: --ik_async --ik_render 8091 should hold the frame-gap max near 300 ms
