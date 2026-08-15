# data/

Datasets for the distillation work. These were expensive to produce (perception is
~1 h of GPU for the real clips, and the teacher pass is minutes-to-hours per robot), and
they used to live only in a scratch directory that disappears with the job — hence this
copy inside the project.

| file | frames | what |
|---|---|---|
| `big4_g1.npz.perception.npz` | 9,396 / 15 clips | **the expensive one.** Streaming SOMA params + kp2d from 18 real videos (3 AV1 clips failed to decode). Robot-agnostic — every robot's labels are made from this. |
| `big4_g1.npz` | 9,396 | + G1 teacher labels (29 DOF) |
| `big4_k1.npz` | 9,396 | + K1 teacher labels (23 DOF), soma-retargeter PR#3 |
| `synth.perception.npz` | 14,400 / 24 clips | synthetic SOMA trajectories from `make_synth.py`, no video involved |
| `synth_g1.npz` | 14,400 | + G1 teacher labels |
| `synth_interp*.npz` | 4,800 / 8 clips | `--mode interp --blend 0.6` — the synthetic set that actually helps |
| `synth_legstatic*.npz` | 4,800 / 8 clips | `--fix leg`: a planted stance, upper body moves. Training data for leg wobble |
| `legstatic_test*.npz` | 1,200 / 4 stances | **the leakage measurement.** Legs held exactly fixed, so any leg motion a model produces is leakage. Never train on this one |
| `*_h264.mp4`, `dance.mp4` | — | source clips that needed transcoding (the AV1 originals in ~/Downloads do not decode here) |

**Splits.** Every number in `PLAN.md` holds out the **last two clips** (16 `taeguk_1st`,
17 `video.mov`, 4,697 frames) and trains on the other 13 (4,699 frames). Hold out by clip,
never by frame — neighbouring frames of one clip are nearly identical.

**Nothing here predates 2026-08-14.** The K1 labels made before then came from a
soma-retargeter config that broke the left leg on half the frames and were deleted; see
the note at the top of `PLAN.md`. Regenerate with:

    GEM-X/.venv/bin/python make_labels.py x --out data/big4_k1.npz --robot k1 \
        --mid data/big4_g1.npz.perception.npz --teacher

**Perception is not reproducible.** The SAM-3D-Body token worker is a background thread,
so which token lands on which frame varies run to run: `big3` and `big4` are the same
clips but differ by 0.011 rad mean in body_pose. Numbers from different perception runs
are not comparable — that is why the dataset file is kept rather than the recipe.
