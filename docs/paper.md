# Distilling an Optimization-Based Retargeter into a Part-Isolated Network for Real-Time RGB-to-Humanoid Motion Targets

HIWooI · mrbluehw@snu.ac.kr · Seoul National University

*Draft, 2026-08-16. Source material: `PLAN.md`, `docs/SUMMARY.md` of the `shape15` repository.*

---

## Abstract

We present a monocular-RGB-to-humanoid pipeline that produces 50 Hz reference motion for a
23-DOF humanoid (K1) at 30 FPS live, with 133 ms of end-to-end perceived delay, starting
from a baseline that took 970 ms and produced 44.8° of joint error. Three findings carry
the result. First, latency is a scheduling problem before it is a compute problem: splitting
perception (GPU/PyTorch) from retargeting (CPU/JAX) into separate processes and making the
exchange non-blocking reduced the worst inter-frame gap from 4017 ms to 308 ms without
changing any model. Second, a student network distilled from an optimization-based
retargeter beats the online solver by a wide margin (9.1° vs 26.8° against the same
teacher), but only once it receives the same input the teacher consumes — feeding the
student 14 keypoints saturates at 15.4° regardless of width, depth, data, or temporal
context, while feeding it the 76-joint skeleton rotation the teacher itself reads drops it
to 9.5° immediately. Third, cross-limb contamination in a monolithic regressor is a
*structural* property that no amount of data or symmetry augmentation removes: cutting the
input, so that the leg expert never observes anything above the pelvis, drives arm-to-leg
leakage to exactly zero at a cost of 0.11° in overall accuracy — and along the way we show
that the contaminating path was the *spine*, not the arm joints or the clavicles we first
suspected. We also report an unusually instructive failure: a self-consistent but wrong
teacher configuration cost a full day of work and was invisible to every statistical check
we ran, and was caught only by rendering the labels next to the human.

---

## 1. Introduction

The task is narrow and its constraint is explicit. A person moves in front of a single RGB
webcam; the system must emit humanoid joint targets that a downstream whole-body policy can
follow, and it must do so with as little delay as possible. It is a demonstration system,
not a controller: ground contact, self-collision, and velocity limits are the downstream
policy's responsibility and are deliberately kept off the critical path.

The system has two outputs:

1. **Display** — a MuJoCo view of the robot imitating the person.
2. **Policy reference** — a 50 Hz motion command consumed by a TWIST-derived whole-body
   policy: 23 reference joint positions, 23 reference joint velocities, and a 6-dimensional
   torso orientation (the first two columns of the reference-vs-robot rotation).

The target robot is K1 (23 DOF). G1 is retained only as a cheap proxy: its teacher runs
roughly 10× faster, so ideas are validated there before being paid for on K1.

The contributions of this report are, in order of how much they moved the numbers:

- A **process-split, non-blocking pipeline** that removes both the latency and the display
  freezing of a chunked retargeter (§3).
- A **distillation recipe** whose decisive variable is the *input representation*, not
  capacity (§4), together with a synthetic-data sampling scheme (interpolation between real
  poses) that is the difference between synthetic data helping and hurting (§5).
- **Structural part isolation**: cutting the input of each per-part expert rather than
  regularizing a shared one, which makes cross-limb leakage identically zero and *improves*
  accuracy (§6).
- A catalogue of **measured negative results** (§8) and **operational pitfalls** (§9) that
  are, in our experience, the more expensive half of the knowledge.

## 2. System

### 2.1 Perception

`demo_webcam.py` runs the perception half: VitPose (DINOv3 ViT-H, fp16, no flip test)
produces 77 SOMA-convention 2D keypoints; a GEM-X denoiser over a 64-frame sliding window
lifts them; SOMA forward kinematics produces an in-camera skeleton, which is drawn on the
frame. A SAM-3D-Body pose token, computed on a background thread, supplies image features.

Image features dominate accuracy, especially under the waist-up framing a desk webcam
gives, because 2D keypoints alone cannot resolve scale and depth:

| configuration | joint error (% of bbox) |
|---|---|
| full window + image features (offline reference) | 0.71 % |
| streaming + image features, token ~7 frames stale | 4.82 % |
| streaming, no image features | 7.95 % |

A second perception backend (MediaPipe, 33 landmarks, 22 ms, CPU) is kept selectable. It
is built for webcam framing and is cheaper, but produces no joint orientations at all.

### 2.2 Retargeting

Three retargeting paths remain selectable and none was removed in place:

| path | mechanism | cost |
|---|---|---|
| `soma-retargeter` | in-house whole-motion optimizer, 30-frame chunks | 870 ms/chunk; **the teacher** |
| PyRoki | per-frame differential IK, warm-started | 6–18 ms |
| distilled network | MLP / part-expert, CPU | 0.3–2.6 ms |

The distilled network is the deployed default (`models/<robot>_retarget.pt`); `--ik` forces
the solver.

### 2.3 Interface to the policy

`motion_command.py` converts joint angles to the policy's observation: a one-euro filter on
the angles, resampling to 50 Hz by interpolation (holding the last value instead alternates
zero velocity with spikes), clamping each step to the URDF's own joint velocity limits, then
differencing to obtain velocity.

Three properties of this interface shape the whole design. No root position is required, so
floor calibration matters only for the display. Velocity is *ours* to produce, and
differencing divides target jitter by `dt` — an IK branch flip became 55 rad/s in the first
version, which is why the clamp exists. And physical feasibility belongs to the policy,
which is what licenses us to leave contact and balance off the critical path.

## 3. Latency is a scheduling problem first

The original pipeline called the in-house retargeter in 30-frame chunks and took 970 ms
end-to-end (~100 ms perception, ~870 ms chunk compute). Two changes removed most of it.

**Per-frame differential IK, on in-camera joints.** Replacing the whole-motion API with a
warm-started per-frame solver removes the chunk minimum entirely. Consuming *in-camera*
rather than world-frame joints additionally drops GEM-X's global-trajectory rollout (35 ms):
a display does not need a world trajectory.

**Process split.** Perception is torch/GPU, the IK is JAX/CPU, and their virtual
environments are mutually exclusive; separating them was forced. It also removed GPU
contention and a warp/torch CUDA-graph capture conflict. The stdin/stdout round trip costs
0.5–0.9 ms, so isolation is effectively free.

A stage-by-stage profile (`replay_delay.py --profile`) then exposed a pure bookkeeping
error: SOMA's `prepare_identity` was rebuilding the person's rest shape on every frame,
although its own docstring says to call it once per identity. Freezing it took FK from 19.8
to 6.2 ms standalone.

| stage | before | after freezing identity |
|---|---|---|
| VitPose | 29.3 ms | 25.9 ms |
| GEM denoiser (`predict`) | 21.0 ms | 23.1 ms |
| SOMA FK | 17.4 ms | **9.2 ms** |
| IK round trip (incl. IPC) | 8.9 ms | 8.1 ms |
| bbox + make_data + rest | 3.5 ms | 3.2 ms |
| **loop total** | **80.1 ms** | **69.5 ms** |
| perceived skeleton / robot-target delay | 167 / 175 ms | **133 / 140 ms** |

The perceived delay exceeds the loop total because it is measured as a viewer sees it: a
12.7 FPS loop against a 30 FPS display means each result is shown for two or three display
frames and ages while it is up. Only cutting compute moves it.

**Freezing was a separate problem from delay.** Perception *waited* for the worker's reply,
so a single slow worker frame stalled the camera. Replacing the exchange with a non-blocking
IPC link that drops a frame rather than stall took the maximum inter-frame gap from **4017
to 308 ms**. Verifying this required first repairing the measurement tool, which had been
taking a different code path from the live system (its own blocking exchange, no worker
rendering) — a recurring theme in this work.

Live figures on the deployed configuration: 30 FPS, inter-frame gap p99 43.7 ms, max 351 ms
over a measured 6,272-frame session.

## 4. Distillation: the student's bottleneck is information, not capacity

The teacher is `soma-retargeter` on K1. The student is trained to imitate its output.
Evaluation is against the teacher on a **clip-level** holdout — the last two clips, 4,697
frames — trained on the remaining 13 clips (4,699 frames). Frame-level splits are
meaningless here because adjacent frames of one clip are nearly identical.

| stage | K1 error | what changed |
|---|---|---|
| initial IK | 44.8° | differential IK on 14 position targets |
| + twist nullspace | — | pulls position-undetermined twist DOFs toward zero |
| + constant offsets | 26.75° | absorbs the teacher/robot joint-zero difference |
| distilled MLP (plain) | 14.94° | a network imitating the teacher's output |
| + residual, 4 blocks | 13.97° | plain depth *hurts*; residual + LayerNorm is what pays |
| + interpolated synthetic data | 10.71° | §5 |
| + mirror augmentation | 10.21° | left-right reflected pairs |
| + part-isolated experts | **9.14°** | §6 |
| + tight torso split | **9.01°** | §6.2 |
| + spine cut (`solo`, deployed) | 9.12° | §6.3 — zero leakage |

**The decisive finding.** With 14 keypoints as input the student saturates at 15.4°, and
neither width, depth, more data, nor temporal context moves it. Given `body_pose` — the
76-joint SOMA rotation vector the teacher actually consumes, and which is already available
for free at run time — it drops to 9.5° in a single step. We had been asking a network to
imitate something that reads the whole skeleton, using 14 points.

The relevant lesson generalizes: before scaling a student, check whether it can *see* what
the teacher sees.

## 5. Synthetic data: the sampling scheme is the whole result

The teacher is a deterministic function of SOMA parameters, so labels can be generated
without video. The naive scheme — widen each joint's range by 1.5× and sample uniformly
(`--mode box`) — was **worthless**, and not because of quantity. Out-of-distribution poses
push the teacher's legs into joint limits, so 41.3 % of the synthetic labels are pathological.

`--mode interp` instead builds each keyframe as a **blend of two real poses**. The result is
novel combinations that never leave the human manifold: 0.0 % pathological.

| synthetic set | pathological | student error |
|---|---|---|
| none (2.6 min of real data) | — | 14.26 ± 0.12° |
| box, margin 1.5, 24 clips | 41.3 % | 14.46° |
| box, margin 1.0, 8 clips | 10.3 % | 14.10° |
| interp, blend 0.35, 8 clips | 0.0 % | 11.74 ± 0.07° |
| **interp, blend 0.6, 8 clips** | **0.0 %** | **10.79 ± 0.07°** |
| interp, blend 1.0, 8 clips | 0.0 % | 11.57 ± 0.05° |
| interp, blend 0.35, 24 clips | 0.0 % | 12.18 ± 0.26° |

Two counterintuitive effects. **More synthetic data is worse**: 8 clips (1:1 with real data)
beats 24 clips (3:1), because once synthetic frames outnumber real ones the fit tilts toward
them. And **the blend coefficient has an interior optimum**: 0.35 stays too close to the
originals to add information, 1.0 is the midpoint of two poses and pulls toward the mean.

## 6. Structural part isolation

### 6.1 The leak is architectural

Live, the user observed that raising only the arms moved the legs. Measured, by perturbing
only the arm dimensions of the input:

| arm-joint perturbation | mean leg change | max leg change |
|---|---|---|
| ±10° | 1.87° | 31.6° |
| ±30° | 4.56° | 38.3° |
| ±60° | 7.89° | 63.8° |

The cause is not the retargeter. A single MLP maps 228-dimensional `body_pose` to all 23
joints at once, so any input change moves every output.

**A methodological error is worth recording here.** Led by the observation that *one* leg
reacted, we set left-right symmetry as the objective and added mirror augmentation. It
delivered symmetry (5.05/4.61 → 4.76/5.13) and **did not reduce total leakage at all**
(9.66 → 9.89°). We had made both legs react. Symmetry was the wrong metric; the mirror
augmentation was kept only because it independently improved accuracy (10.79 → 10.21°).

The correct fix is structural: give the leg expert no arm input, and leakage is zero by
construction rather than by training.

| architecture | overall error | leg change under ±10° arm perturbation |
|---|---|---|
| single residual net (+ mirror) | 10.21° | mean 1.87°, max 31.58° |
| **part experts with input cut** | **9.14°** | **exactly 0.000000°** |

Accuracy *improved* by 1.07°, apparently because each expert stops being perturbed by
irrelevant inputs. Per part: legs 5.93° (66-dim input), arms 13.85° (198-dim), torso 0.46°
(36-dim). CPU inference rises from 0.3 to 2.56 ms, negligible in a 42 ms loop.

Note that the same part-expert decomposition **without** the input cut gives no accuracy
gain and no leakage reduction — parameter count aside, the cut is the whole mechanism.

### 6.2 The clavicles

The synthetic perturbation test read 0.000000°, and the legs still moved on screen. The test
was wrong: in real perception output, arm and leg inputs are correlated at +0.718 and arm
and torso at +0.815, so raising an arm moves the torso input too. The leak path was the
*torso* input, not the arm input.

A dedicated instrument was needed: a synthetic test set in which the legs are held exactly
fixed while the upper body moves (`data/legstatic_test_k1.npz`, 4 stances). Any leg motion
there is leakage by definition.

The "torso" group of `fixtures/soma_part77.npy` contains the clavicles (11, 39) and the head
chain (4–10). Clavicles rotate whenever an arm is raised, so the leg expert — which saw no
arm joint at all — was receiving arm motion in real time. `--part_in tight` gives the legs
only the spine (1, 2, 3).

| model | holdout | leg std | leg inter-frame |
|---|---|---|---|
| teacher (ground truth) | — | 2.546° | 0.1555° |
| `parts` (wide) | 9.14° | 4.110° | 0.2836° |
| **`--part_in tight`** | **9.01°** | 3.946° | **0.2394°** |
| `--fuse` (wide) | 9.11° | 4.138° | 0.2867° |
| `tight --fuse` | 8.99° | 3.951° | 0.2406° |
| `tight` + leg-static training data | 9.60° | **3.474°** | **0.2217°** |

Reducing the leg expert's input from 66 to 39 dimensions *improved* its error from 5.93 to
5.69°: clavicles and head were noise, not information. Visible on-screen jitter fell 16 %.

The fusion correction layer (assembled 23 angles → 256 → 256 → 23 deltas) was **rejected**:
9.14→9.11 and 9.01→8.99 are seed noise, and jitter increased in the wide variant. A
23-dimensional bottleneck carries nothing the individual experts missed. It stays available
behind `--fuse`, off by default.

### 6.3 The spine

Contamination reports continued with `tight` running live. Perturbing input groups
individually (±20°, measuring leg output change) located the path structurally:

| perturbation | `parts` | `tight` | **`solo`** |
|---|---|---|---|
| arm axes only | 0.000° | 0.000° | 0.000° |
| clavicles only (11, 39) | mean 5.96 / **max 168°** | 0.000° | 0.000° |
| **spine only (1–3)** | 17.67 / 187.8° | **19.05 / 217.3°** | **0.000°** |
| everything above the pelvis | 18.00 / 188.8° | 19.54 / 244.5° | **0.000°** |

Cutting the clavicles left the spine path intact, and the spine path is an order of
magnitude larger. In perception output, spine and arm motion are correlated at **+0.737**,
and in the top 5 % of frames by arm speed the spine moves **4.3×** its usual amount. Raising
an arm makes perception bend the spine; the leg expert, highly sensitive to the spine,
follows. That is the contamination the user was seeing.

`--part_in solo` restricts the leg expert to the 30 leg dimensions. Nothing above the pelvis
reaches the legs.

| | holdout | leg std (leg-static set) | above-pelvis perturbation |
|---|---|---|---|
| teacher | — | 2.546° | — |
| `tight` | **9.01°** | 3.946° | max 244.5° |
| **`solo` (deployed)** | 9.12° | **0.000°** | **0.000°** |

The cost is 0.11°. What is given up is the genuine torso→leg coupling the teacher has
(2.546°), and we argue this is the right trade for our interface: the downstream TWIST
policy balances on its own, so there is little reason to bake an upper-body-inferred weight
shift into a *reference* pose. `tight` remains selectable for torso-led stances.

## 7. Temporal behavior: the student amplifies the teacher's discontinuities

Reports of the output "jumping at certain angles" were decomposed on real data. In the top
1 % of frames by inter-frame jump:

| | value |
|---|---|
| perception input motion | 2.89° (typical 0.59°) |
| teacher jump | 55.6° |
| **student jump** | **122.3°** |

The teacher is itself discontinuous there, and the student amplifies it 2.2×. The affected
joints are all in the arms — `right_shoulder_pitch` 61.8°, `right_shoulder_yaw` 57.9°,
`left_shoulder_pitch` 51.5°, elbow 41.8° — consistent with the twist-DOF analysis of §8.

No new component was written: the one-euro filter already in `motion_command.py` was reused
on the retarget output. Because the student's inter-frame noise exceeds the teacher's,
low-pass filtering pulls the output *toward* the teacher rather than away:

| `--smooth` | holdout (15 fps) | worst single-frame jump |
|---|---|---|
| 0 (off) | 9.01° | 237.9° |
| 3.0 Hz | 8.86° | 133.2° |
| **2.0 Hz (default)** | **8.93°** | **108.1°** |
| 1.5 Hz | 9.04° | 90.1° |

Jumps fall 2.2× at no accuracy cost. The one-euro `beta` (speed adaptation) is pinned to 0:
its usual virtue — widening the cutoff when motion is fast — here passes the spike through
at exactly the moment it should be suppressed.

**A frame-rate caveat that generalizes.** A 3 Hz cutoff has τ = 53 ms, so at 16 fps (62 ms
period) it barely acts. The filter is therefore given the *measured* `dt`, and
`motion_command.py`'s self-check covers this property. It is also why numbers measured at
30 fps were re-measured at 15.

The root cause is untouched: the teacher's own 55.6° discontinuity is masked, not removed.
Fixing it properly means a continuity loss during training, or predicting the twist DOFs in
a representation other than position.

## 8. Where the residual error lives

For the deployed model:

| part | error | relative to teacher std |
|---|---|---|
| **arms** (10 joints) | 15.74° | 0.298 |
| legs (12 joints) | 7.35° | 0.237 |
| torso | 0.79° | 0.107 |

The worst joints are `shoulder_yaw` (19.7–21.2°) and `wrist_roll` (relative 0.50) — the
**twist DOFs that position targets do not determine**. This is a property of the objective,
not of the network: the solver builds its targets with `SO3.identity()` and `ori_weight=0`,
so it sees 14 *points*, while GEM-X does produce joint orientations that are discarded.

By pose:

| condition | error |
|---|---|
| hands below shoulders | 5.3° |
| **hands above shoulders** | **21.4°** |
| narrow stance | 5.1° |
| **stance ≥ 0.67 m** | **20.6°** |
| **arms fully extended** | **19.7°** |

Keypoint confidence is not the cause: the worst 5 % of frames average 0.86 confidence
against a global mean of 0.87. These poses are intrinsically hard, not badly perceived.

### 8.1 Measured negative results

Recorded so they are not retried:

| attempt | outcome |
|---|---|
| train longer | test minimum at epoch 12; by 600 train error is 0.12°, pure memorization |
| scale width/depth | plain nets degrade at w4096 × 6 layers (residual is the exception) |
| temporal context (past frames) | 15.4 → 15.2°, noise |
| future frames (diagnostic) | 12.4°, *worse* |
| latency predictor | 132 ms of lag costs +0.28°; the predictor pays 1.6–2.5° |
| FK skip (auxiliary t14 output) | 179.7 mm, barely better than a fixed mean pose (223.7) |
| orientation targets in the solver | 17.1 → 16.4° only |
| teacher post-processing forced on | pathological rate 1.0 % → 44.4 % (the author's default was right) |
| oversampling extreme synthetic poses | 10.79 → 15.50°, and *worse* on the hard poses it targeted (20.72 → 27.94°) |
| adopting the shape10 retargeter | identical to the PR#1 we had already deleted |
| part experts **without** the input cut | accuracy unchanged, leakage unchanged |

## 9. Two failures worth publishing

### 9.1 A self-consistent teacher can still be wrong

Label quality was checked **numerically only**. The network memorized labels to 0.18°, and
nearest-neighbor label differences were 1.8°, from which we concluded "the labels are
self-consistent, therefore the teacher is fine."

The user watched a side-by-side video of the human and the teacher and reported that **the
left leg did not match the person**. Self-consistency is not accuracy: a function can return
the same wrong answer every time.

The cause was a pinned submodule. `soma-retargeter` was fixed at PR#1, whose K1
configuration described *itself*, in its own `_provenance` field, as an experimental
candidate using "G1 IK objective policy and G1-style target frames", with
`feet_stabilizer_config` pointing at the G1 one. Upstream PR#3 had already added a K1-specific
feet stabilizer and proxy bodies.

| clip 17, all 2,418 frames | pathological frames |
|---|---|
| PR#1 (what we were using) | 50.4 % |
| **PR#3 (upgraded)** | **1.0 %** |
| PR#3 + post-processing forced on | 44.4 % |

Across the dataset the pathological rate went 26 % → 6.6 % and teacher throughput 1.2 → 10
fps. Every contaminated K1 label, model, and offset was deleted and regenerated. The
practical rule we now follow: **render before concluding.** Statistics cannot see a
consistent bias; a human watching two skeletons side by side sees it in seconds.

A related risk was found and removed during the investigation: the parent index still
recorded PR#1, so a plain `git submodule update` would have silently reverted the fix.

### 9.2 Standard deviation cannot tell left from right

`left_hip_roll` std was 1.79× the right, which looked like teacher asymmetry. A mirror test
exonerated it: the disagreement between `teacher(mirror(x))` and `mirror(teacher(x))` is
0.62° mean, 0.05° median. The retargeter is symmetric.

Per clip, the asymmetry disappears — clips 0, 2, 4, 11 run 0.60–0.69× (right larger), clips
7 and 17 run 1.07–1.21×, and two clips dominate: clip 5 (taekwondo) at 1.94× and clip 14
(cheerleading) at 1.75×, with standard deviations 5–10× the others. The global 1.79× is
those two clips, and they are genuinely one-legged choreography.

The earlier evidence that "the human input is symmetric" (foot-position std 792.9 vs
792.6 mm) was equally invalid: **std does not distinguish left from right.** It returns the
same value whether the left foot or the right foot is the one being raised.

## 10. Operational pitfalls

Three properties of this system that cost us time and are not obvious:

1. **Perception is not reproducible.** The SAM-3D-Body token worker is a background thread,
   so which token lands on which frame differs between runs. Rebuilding a dataset from the
   same video shifts `body_pose` by 0.011 rad on average and invalidates comparison with
   earlier numbers. **The `.npz` files are the asset, not the recipe that produced them.**
2. **Hold out by clip.** Adjacent frames of one clip are nearly identical; a frame-level
   split reports a meaningless number.
3. **CPU networks need `torch.set_num_threads(2)`.** The default (one thread per core)
   contends with neighboring jobs such as Isaac and turns a 0.09 ms forward pass into 60 ms
   — a measured 700× regression.

## 11. Results summary

| metric | value |
|---|---|
| accuracy vs teacher, 4,697-frame clip-level holdout | **9.12°** (deployed `solo`; `tight` 9.01°, `parts` 9.14°) |
| baseline: PyRoki differential IK | 26.75° |
| session start (2026-08-13) | 44.8° |
| live throughput | 30 FPS (local GUI, `--no_imgfeat`) |
| perceived delay | 133 ms skeleton / 140 ms robot target (from 970 ms) |
| arm→leg leakage | **0.000000°** (structural) |
| above-pelvis→leg leakage | **0.000000°** (structural) |
| inter-frame gap, p99 / max | 43.7 / 351 ms over 6,272 live frames |
| worst single-frame jump, `--smooth 2.0` | 108.1° (from 237.9°) |
| policy reference | 50 Hz, 46 + 6 dimensions, matching the downstream schema |

Reproducing the best model:

```bash
.venv-ik/bin/python distill.py data/big4_k1.npz --features pose --width 1024 \
    --arch parts --part_in solo --extra data/synth_interp_k1.npz --mirror \
    --model models/k1_retarget.pt
```

Running the live system:

```bash
DISPLAY=:1 GEM-X/.venv/bin/python demo_webcam.py --flip --robot k1 --no_imgfeat
```

## 12. Limitations and next steps

**Twist DOFs remain the dominant error.** They are undetermined by position targets, and
GEM-X's joint orientations are available and currently discarded. This is the single largest
identified lever on the remaining 9°.

**The teacher's own discontinuities are masked, not fixed.** A continuity loss during
training, or a rotation representation for the twist DOFs, is the proper repair.

**Mirror mode is unimplemented on the network path.** `--mirror` at run time reflects target
points inside the IK worker; a network that emits joint angles directly has no such point.
The rule is already validated (`fixtures/soma_pair77.npy`; on the robot side, swap left/right
and negate roll and yaw, confirmed to 0.00 mm by FK) and needs only to be applied to the
network output.

**Deployment.** Three stages remain: driving the downstream policy in simulation (the
interface already matches — our output is the leading 46 + 6 of its 124-dimensional student
observation — but its own teacher→student distillation is unverified, its runtime reads CSV
rather than a stream, and a feasibility gate is open: our maximum is 20.9 rad/s against its
4.6–12.6 safe range); hardware deployment; and on-board inference, where the perception
budget (47 ms on an RTX 5090) is the gate and a 2D-keypoint student may become mandatory.

## References

Recorded as they appear in the project's design document.

- PyRoki — differential IK toolkit. https://pyroki-toolkit.github.io/
- mink — MuJoCo differential IK. https://kevinzakka.github.io/mink/
- NMR — RL experts repair human motion onto a robot's feasible manifold; a CNN-Transformer
  learns the mapping. https://arxiv.org/pdf/2603.22201
- H2O / OmniH2O — human keypoints fed directly to an RL policy.
  https://github.com/LeCAR-Lab/human2humanoid
- MIRROR — visual skeleton estimation → GPU-parallel continuation-based differential IK with
  control-barrier-function self-collision avoidance. https://arxiv.org/abs/2603.23995
- G. Casiez, N. Roussel, D. Vogel. "1€ Filter: A Simple Speed-based Low-pass Filter for Noisy
  Input in Interactive Systems." CHI 2012.
- Internal: `soma-retargeter` (PR#3), GEM-X, SOMA-77 skeleton convention, AI-Sapiens K1,
  TWIST-derived whole-body policy (`shape14`).
