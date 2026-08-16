# shape15 — real-time RGB → humanoid motion targets

One RGB webcam in, 50 Hz reference motion for a 23-DOF humanoid (K1) out, at 30 FPS live
with 133 ms of perceived delay. A network distilled from an optimization-based retargeter
does the retargeting (9.1° vs the solver's 26.8° against the same teacher), with per-part
input isolation that makes cross-limb leakage structurally zero.

| | |
|---|---|
| `docs/paper.md` | **the write-up** — method, measurements, negative results |
| `docs/SUMMARY.md` | 목적 · 도달점 · 경로 (Korean) |
| `docs/RUNNING.md` | how to run it |
| `PLAN.md` | full experiment log |
| `todo.md` | what is left |

```bash
DISPLAY=:1 GEM-X/.venv/bin/python demo_webcam.py --flip --robot k1 --no_imgfeat
```

Datasets (`data/*.npz`), weights (`models/*.pt`), the GEM-X tree, and virtualenvs are not in
git — see `.gitignore` and `data/README.md`. Perception is not reproducible frame-for-frame
(background-thread token scheduling), so the `.npz` files are the asset, not the recipe.
