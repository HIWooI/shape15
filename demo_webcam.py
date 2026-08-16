"""Real-time webcam SOMA skeleton demo (GEM-X).

Streams /dev/videoN through VitPose (77 SOMA keypoints) and the GEM denoiser
over a sliding window, then draws the projected 3D SOMA skeleton on the frame.

Run with the GEM-X venv:
    GEM-X/.venv/bin/python demo_webcam.py
"""

# ruff: noqa: E402
import argparse
import os
import sys
import threading
import time
from collections import deque
from pathlib import Path

HERE = Path(__file__).parent.resolve()
# the 14 ik_map joints, as indices into SOMA's 77
SOMA_IDX = [i - 1 for i in (1, 4, 13, 14, 15, 41, 42, 43, 68, 69, 70, 73, 74, 75)]
# How long the T-pose prompt stays up. Display only — the worker decides when it has
# enough frames; this just has to outlast its 15, so the operator holds the pose that long.
CALIB_FRAMES = 18
GEM_ROOT = Path(__file__).parent / "GEM-X"
sys.path.insert(0, str(GEM_ROOT.resolve()))
os.chdir(GEM_ROOT.resolve())  # hydra configs + inputs/ are relative to the repo root

def _fix_onnxruntime_cuda():
    """Make onnxruntime find CUDA, which unlocks YOLOX and the ONNX/TensorRT paths.

    torch ships libcudart.so.13 inside `nvidia/cu13/lib` and loads it through its own
    mechanism, so the dynamic linker never learns that directory. onnxruntime asks the
    linker and fails with `libcudart.so.13: cannot open shared object file`, which takes
    YOLOXDetector down with it — it imports onnxruntime in __init__ with no fallback.
    Loading the library globally first fixes it without an env var.
    """
    import ctypes
    import glob

    for so in glob.glob(str(GEM_ROOT / ".venv/lib/*/site-packages/nvidia/cu13/lib/libcudart.so*")):
        try:
            ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
            return so
        except OSError:
            pass
    return None


_fix_onnxruntime_cuda()

import cv2
import numpy as np
import torch

from motion_command import OneEuro  # plain numpy, so it imports from either venv
from gem.utils.cam_utils import estimate_K
from gem.utils.geo_transform import compute_cam_angvel, get_bbx_xys_from_xyxy
from gem.utils.kp2d_utils import (
    _BONE_STICKWIDTH_77,
    _draw_ellipse_bone,
    _JOINT_GROUP_77,
    _JOINT_RADIUS_77,
    _PART_COLORS_77,
    PARENTS_77,
)

import gem.pipeline.gem_pipeline as _gp
import gem.utils.vitpose_extractor as _vpe

CONF_THR = 0.35
IDENTITY_ANGVEL = torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]], device="cuda")  # rot6d(I)

# A progress bar over a 1-frame batch, thirty times a second.
_vpe.tqdm = lambda it, **kw: it

# We only draw the in-camera skeleton, but the pipeline always rolls out the global
# trajectory too — a python for-loop costing ~35 ms/frame. Stub it out. (Kept around
# under another name: postproc=True needs the real one.)
_gp.get_body_params_w_Rt_v2_full = _gp.get_body_params_w_Rt_v2
_gp.get_body_params_w_Rt_v2 = lambda *, global_orient_c, **kw: {
    "global_orient": global_orient_c,
    "transl": torch.zeros_like(global_orient_c),
}


def build_model(ckpt=None, exp="gem_soma_regression"):
    import hydra
    from hydra import compose, initialize_config_dir

    overrides = [
        f"exp={exp}",
        "video_name=webcam",
        "video_path=webcam.mp4",
        "static_cam=true",
        "render_mhr=false",
        "use_wandb=false",
        "task=test",
    ]
    if ckpt is not None:
        overrides.append(f"ckpt_path={ckpt}")
    with initialize_config_dir(version_base="1.3", config_dir=str(GEM_ROOT / "configs")):
        cfg = compose(config_name="demo_soma", overrides=overrides)

    model = hydra.utils.instantiate(cfg.model, _recursive_=False)
    ckpt_path = cfg.ckpt_path
    if ckpt_path is None:
        from gem.utils.hf_utils import download_checkpoint

        ckpt_path = download_checkpoint()
    model.load_pretrained_model(ckpt_path)
    model = model.eval().cuda()
    # predict() re-encodes the (always empty) caption every call — ~30 ms wasted per frame.
    soma = model.body_model.soma if hasattr(model.body_model, "soma") else model.body_model
    freeze_identity(soma)
    model._rot_stash = capture_joint_rotations(soma)
    empty_text = model.encode_text([""], torch.tensor([True]))
    model.encode_text = lambda caption, has_text: empty_text
    return model


def patch_single_decode(sam):
    """Stop SAM-3D-Body from running its decoder twice per frame.

    forward_pose_branch already computes the pose token (sam3d_body.py:1124) but leaves it
    out of the returned dict, so GEM's extractor patch runs the whole decoder a second time
    just to recover it. Capture it from the first pass instead: ~530 ms -> ~280 ms.
    """
    m = sam.estimator.model
    stash = {}
    orig_decoder = type(m).forward_decoder.__get__(m)
    orig_branch = type(m).forward_pose_branch.__get__(m)  # pre-patch, from the class

    def forward_decoder(*a, **kw):
        out = orig_decoder(*a, **kw)
        stash["tokens"] = out[0]
        return out

    def forward_pose_branch(batch):
        out = orig_branch(batch)
        out["pose_token"] = stash.pop("tokens", None)  # makes GEM's patch return early
        return out

    m.forward_decoder = forward_decoder
    m.forward_pose_branch = forward_pose_branch


@torch.no_grad()
def sam3db_token(sam, frame, bbx):
    """One frame's SAM-3D-Body pose token: the body of SAM3DBExtractor.extract_video_features.

    Without this conditioning the model has only 2D keypoints to work from, which is
    weakest exactly when the body is cropped (waist-up webcam framing).
    """
    from sam_3d_body.data.utils.prepare_batch import prepare_batch
    from sam_3d_body.utils import recursive_to

    cx, cy, s = (float(v) for v in bbx)
    box = np.array([cx - s / 2, cy - s / 2, cx + s / 2, cy + s / 2], np.float32).reshape(1, 4)
    # RGB here — extract_video_features feeds it read_video_np output. (VitPose is the
    # opposite: its get_batch flips BGR itself, so that one takes the cv2 frame as-is.)
    batch = prepare_batch(frame[..., ::-1], sam.estimator.transform, box, masks=None, masks_score=None)
    batch = recursive_to(batch, sam.device)
    sam.estimator.model._initialize_batch(batch)
    tok = sam.estimator.model.forward_step(batch, decoder_type="body")["pose_token"]
    return (tok[:, 0] if tok.ndim == 3 else tok).float()


def bootstrap_bbox(vitpose, frame, W, H, rounds=3):
    """Find the person on the first frame by re-running VitPose on its own output.

    There is no detector to lean on (YOLOX is ONNX-only and onnxruntime is broken here), so
    the loop starts from a full-frame box — on which VitPose is poor, because the person is
    a small part of a very wide crop. Normally that corrects itself over a few frames, but
    identity and bone lengths are now frozen from the start, so a bad first frame is a bad
    session. Two extra passes cost ~50 ms once and converge immediately.
    """
    bbx = get_bbx_xys_from_xyxy(torch.tensor([[0.0, 0.0, W - 1.0, H - 1.0]]))[0].float()
    was, vitpose.flip_test = vitpose.flip_test, True  # one-off: accuracy over speed
    try:
        for _ in range(rounds):
            with torch.autocast("cuda", dtype=torch.float16):
                kp = vitpose.extract(frame[None], bbx[None], path_type="np", batch_size=1)[0]
            nb = bbox_from_kps(kp, W, H)
            if nb is None:
                break
            bbx = nb
    finally:
        vitpose.flip_test = was
    return bbx


def bbox_from_kps(kp2d, W, H):
    """Bbox for the next frame from the current keypoints; None if track is lost."""
    ok = kp2d[:, 2] > CONF_THR
    if ok.sum() < 10:
        return None
    xy = kp2d[ok, :2]
    xyxy = torch.tensor([[xy[:, 0].min(), xy[:, 1].min(), xy[:, 0].max(), xy[:, 1].max()]])
    xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clamp(0, W - 1)
    xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clamp(0, H - 1)
    return get_bbx_xys_from_xyxy(xyxy, base_enlarge=1.2).float()[0]


def make_data(kp2d_win, bbx_win, K, f_imgseq=None):
    """Same layout as demo_soma_onnx.load_data_dict(no_imgfeat=True), static cam.

    Everything is built on the GPU: predict() would otherwise copy the whole
    window up host-to-device on every single frame.
    """
    L = len(kp2d_win)
    R_w2c = torch.eye(3, device="cuda").unsqueeze(0).repeat(L, 1, 1)
    cam_angvel = IDENTITY_ANGVEL.repeat(L, 1) if L == 1 else compute_cam_angvel(R_w2c)
    ones = torch.ones(L, dtype=torch.bool, device="cuda")
    zeros = torch.zeros(L, dtype=torch.bool, device="cuda")
    has_img = ones if f_imgseq is not None else zeros
    return {
        "meta": [{"vid": "webcam"}],
        "length": torch.tensor(L),
        "bbx_xys": torch.stack(list(bbx_win)),
        "kp2d": torch.stack(list(kp2d_win)),
        "K_fullimg": K.cuda().repeat(L, 1, 1),
        "cam_angvel": cam_angvel,
        "cam_tvel": torch.zeros(L, 3, device="cuda"),
        "R_w2c": R_w2c,
        "T_w2c": torch.eye(4, device="cuda").unsqueeze(0).repeat(L, 1, 1),
        "f_imgseq": torch.zeros(L, 1024, device="cuda") if f_imgseq is None else f_imgseq,
        "noisy_pred_cam": None,
        "has_text": torch.tensor([False]),
        "mask": {
            "valid": ones,
            "has_img_mask": has_img.clone(),
            "has_2d_mask": ones.clone(),
            "has_cam_mask": ones.clone(),
            "has_audio_mask": zeros.clone(),
            "has_music_mask": zeros.clone(),
        },
    }


def capture_joint_rotations(soma):
    """Keep the per-joint world rotations SOMA computes and then discards.

    `soma.pose()` builds `T_world` for all 78 joints and returns only its translation
    column (soma.py: `joints = T_world[..., :3, 3]`). The rotations are the one thing
    positions cannot give — the twist about a limb — and re-deriving them from points is
    what leaves an arm free to fold into the wrong branch. Stash them instead.
    """
    orig = soma.pose
    stash = {}

    def pose(*a, **kw):
        out = orig(*a, **kw)
        stash["T_world"] = soma.batched_skinning.last_T_world
        return out

    # batched_skinning.pose already returns T_world to soma.pose; capture it there
    orig_bs = soma.batched_skinning.pose

    def bs_pose(*a, **kw):
        res = orig_bs(*a, **kw)
        if isinstance(res, tuple) and len(res) == 2:
            soma.batched_skinning.last_T_world = res[1]
        return res

    soma.batched_skinning.pose = bs_pose
    soma.pose = pose
    return stash


class _ResBlock(torch.nn.Module):
    """Pre-norm residual block, matching distill.py --arch res."""

    def __init__(self, w, drop=0.1):
        super().__init__()
        self.n = torch.nn.LayerNorm(w)
        self.f = torch.nn.Sequential(torch.nn.Linear(w, w), torch.nn.GELU(),
                                     torch.nn.Dropout(drop), torch.nn.Linear(w, w))

    def forward(self, x):
        return x + self.f(self.n(x))


def _log_summary(path, rows, frames, secs, n_calib, n_drop, n_reset=0):
    """What the session actually did, in the terms that matter for a freeze."""
    if not rows:
        print(f"[log] no frames recorded -> {path}")
        return
    a = np.array(rows)                       # gap, loop, vitpose, predict, fk
    gap = a[1:, 0] if len(a) > 1 else a[:, 0]
    pc = lambda v, q: float(np.percentile(v, q))
    print(f"\n[log] {frames} frames in {secs:.0f} s ({frames / max(secs, 1e-6):.1f} FPS)")
    print(f"  frame gap ms : median {np.median(gap):6.1f}  p99 {pc(gap, 99):6.1f}  "
          f"max {gap.max():6.1f}   <- 멈춤은 이 값으로 판정한다")
    print(f"  loop ms      : median {np.median(a[:, 1]):6.1f}  p99 {pc(a[:, 1], 99):6.1f}")
    for k, col in (("vitpose", 2), ("predict", 3), ("fk", 4)):
        v = a[a[:, col] > 0, col]
        if len(v):
            print(f"  {k:12s} : median {np.median(v):6.1f}  p99 {pc(v, 99):6.1f}")
    print(f"  calibrations {n_calib}, frames the worker could not take {n_drop}")
    # every reset throws away the denoiser's 64-frame window and shows as a pose jump
    print(f"  tracking resets {n_reset}" + (f" (one per {secs / n_reset:.0f} s)" if n_reset else ""))
    print(f"  per-frame rows -> {path}")


class _PartExperts(torch.nn.Module):
    """One expert per body part, each reading only its own chain of the input.

    The point is not accuracy — it matches a single net (9.14 deg here) — it is that the
    leg expert never sees an arm input, so raising an arm cannot move a leg. A single net
    maps all 228 inputs to all 23 outputs and moved the legs up to 31 deg for a 10 deg
    arm perturbation, which is visible and wrong on screen.

    Each expert carries its own normalisation, so this holds its inputs unnormalised and
    scales per part.
    """

    def __init__(self, ck, ndof):
        super().__init__()
        self.ndof = ndof
        self.nets, self.idx = torch.nn.ModuleDict(), {}
        for name, p in ck["parts"].items():
            if name == "_fuse":
                continue
            w, nb = ck["width"], ck.get("blocks", 4)
            net = torch.nn.Sequential(
                torch.nn.Linear(len(p["in"]), w), *[_ResBlock(w) for _ in range(nb)],
                torch.nn.LayerNorm(w), torch.nn.Linear(w, len(p["out"])))
            net.load_state_dict(_remap_block_keys(p["state"]))
            net.eval()
            self.nets[name] = net
            self.idx[name] = (
                torch.tensor(np.asarray(p["in"]), dtype=torch.long),
                torch.tensor(np.asarray(p["out"]), dtype=torch.long),
                torch.tensor(np.asarray(p["mu"]), dtype=torch.float32),
                torch.tensor(np.asarray(p["sd"]), dtype=torch.float32),
            )

        f = ck["parts"].get("_fuse")
        if f is not None:
            self.fuse = torch.nn.Sequential(
                torch.nn.Linear(ndof, 256), torch.nn.GELU(),
                torch.nn.Linear(256, 256), torch.nn.GELU(),
                torch.nn.Linear(256, ndof))
            self.fuse.load_state_dict(f["state"])
            self.fuse.eval()
            self.fmu = torch.tensor(np.asarray(f["mu"]), dtype=torch.float32)
            self.fsd = torch.tensor(np.asarray(f["sd"]), dtype=torch.float32)
        else:
            self.fuse = None

    def forward(self, x):
        out = x.new_zeros(self.ndof)
        for name, net in self.nets.items():
            ci, co, mu, sd = self.idx[name]
            out[co] = net((x[ci] - mu) / sd)
        if self.fuse is not None:
            out = out + self.fuse((out - self.fmu) / self.fsd)
        return out


def _remap_block_keys(state):
    """distill.py names the block's LayerNorm `norm`, this file's `_ResBlock` names it `n`.

    Both spellings exist in saved checkpoints, and the mismatch fails the load with a
    bare "Missing key(s)". Accept either rather than invalidating trained weights.
    """
    return {k.replace(".norm.", ".n."): v for k, v in state.items()}


def build_student(ck, ndof, extra_out=0):
    """Rebuild a distill.py checkpoint's network from the shape it recorded.

    The checkpoint carries `arch`/`blocks`, so plain, residual and per-part models all
    load through the same call — older checkpoints have neither and default to plain.
    A per-part model normalises internally, so callers must feed it raw features.
    """
    if str(ck.get("arch", "plain")) == "parts":
        return _PartExperts(ck, ndof)
    w, nb = ck["width"], ck.get("blocks", 4)
    n_out = ndof + extra_out
    if str(ck.get("arch", "plain")) == "plain":
        return torch.nn.Sequential(
            torch.nn.Linear(len(ck["mu"]), w), torch.nn.GELU(),
            torch.nn.Linear(w, w), torch.nn.GELU(), torch.nn.Linear(w, n_out))
    return torch.nn.Sequential(
        torch.nn.Linear(len(ck["mu"]), w), *[_ResBlock(w) for _ in range(nb)],
        torch.nn.LayerNorm(w), torch.nn.Linear(w, n_out))


def freeze_identity(soma):
    """Compute the person's rest shape once instead of every frame.

    `prepare_identity` rebuilds the identity model and the skeleton transfer on every
    call — 12 ms of the 20 ms FK — even though its own docstring says to call it once per
    identity and then just `pose()`. It is the same person for a whole session.
    """
    orig = soma.prepare_identity

    def once(*a, **kw):
        if not getattr(soma, "_identity_frozen", False):
            orig(*a, **kw)
            soma._identity_frozen = True

    soma.prepare_identity = once
    return soma


@torch.no_grad()
def project_last_frame(model, pred, K):
    """FK the last window frame's in-camera SOMA params and project to pixels."""
    p = {k: v[-1:].cuda() for k, v in pred["body_params_incam"].items()}
    poses = torch.cat([p["global_orient"][:, None], p["body_pose"].reshape(1, 76, 3)], dim=1)
    # static_forward, not forward(), to skip the vertex computation we never draw
    joints = model.body_model.static_forward(
        poses, p["identity_coeffs"], p["scale_params"], p["transl"], return_joints_only=True
    )["joints"][0]  # (77, 3) camera space
    uv = joints @ K.to(joints.device).T
    xy = (uv[:, :2] / uv[:, 2:3].clamp(min=1e-3)).cpu().numpy()
    # T_world covers 78 joints with a dummy root at 0, so SOMA joint i sits at i+1
    rows = [i + 1 for i in SOMA_IDX]
    T = model._rot_stash.get("T_world")
    rot = None
    if T is not None:
        cur = T[0, rows, :3, :3]
        if getattr(model, "_rest_R", None) is None:
            # A-pose reference: same identity, zero pose. Anchoring on the first frame
            # instead would make the robot's rest pose whatever the person happened to be
            # doing when the demo started.
            zero = torch.zeros_like(poses)
            model.body_model.static_forward(
                zero, p["identity_coeffs"], p["scale_params"], p["transl"],
                return_joints_only=True,
            )
            model._rest_R = model._rot_stash["T_world"][0, rows, :3, :3].clone()
            model._rot_stash["T_world"] = T  # restore, the zero-pose call overwrote it
        rot = (cur @ model._rest_R.transpose(-1, -2)).cpu().numpy()  # rest -> current
    return xy, joints.cpu().numpy(), rot


def draw_skeleton(img, xy, conf=None):
    """Bones + joints, same style as gem.utils.kp2d_utils.render_2d_keypoints."""
    for child, parent in enumerate(PARENTS_77):
        if parent < 0 or (conf is not None and min(conf[parent], conf[child]) <= 0.5):
            continue
        canvas = img.copy()
        _draw_ellipse_bone(
            canvas,
            xy[parent].tolist(),
            xy[child].tolist(),
            _PART_COLORS_77[_JOINT_GROUP_77[child]],
            _BONE_STICKWIDTH_77[child],
        )
        img = cv2.addWeighted(img, 0.4, canvas, 0.6, 0)
    for j in range(77):
        if conf is not None and conf[j] <= 0.5:
            continue
        pt = tuple(xy[j].astype(int))
        r = _JOINT_RADIUS_77[j]
        cv2.circle(img, pt, r, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(img, pt, max(r - 1, 1), _PART_COLORS_77[_JOINT_GROUP_77[j]], -1, cv2.LINE_AA)
    return img


class IKLink:
    """Talk to the IK worker without ever waiting on it.

    The loop used to write a frame's targets and then block on the reply. IK is ~8 ms so
    that looked free, but any hiccup in the worker — a slow solve, a MuJoCo render stall —
    froze perception too: no camera read, no frame served, both panels stuck. Now a reader
    thread drains replies and the loop takes whatever is newest, so the robot may lag a
    frame while the camera never stops. Writes are capped at a couple in flight, otherwise
    a slow worker would just back up the pipe and block the write instead.
    """

    def __init__(self, cmd, cwd, ndof, max_inflight=2):
        import subprocess

        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, cwd=cwd
        )
        self.proc.stdout.read(4)  # ready marker
        self.n_out = (1 + ndof) * 4
        self.max_inflight = max_inflight
        self.inflight = 0
        self.latest = None
        self.lock = threading.Lock()
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        while True:
            buf = self.proc.stdout.read(self.n_out)
            if len(buf) < self.n_out:
                break
            with self.lock:
                self.latest = buf
                self.inflight = max(0, self.inflight - 1)

    def submit(self, payload):
        with self.lock:
            if self.inflight >= self.max_inflight:
                return False  # worker is behind; drop this frame rather than stall
            self.inflight += 1
        try:
            self.proc.stdin.write(payload)
            self.proc.stdin.flush()
        except (BrokenPipeError, ValueError):
            return False
        return True

    def close(self):
        try:
            self.proc.stdin.close()
        except Exception:
            pass


class TokenWorker:
    """SAM-3D-Body in the background.

    One call costs ~530 ms at batch 1 (the decoder has a huge fixed per-call cost), so
    it cannot sit in the loop. It does not need to: holding a token for 8 frames measured
    5.45% joint error vs 4.29% refreshing every frame, against 7.95% with no features at
    all. So the loop uses whatever the latest token is and never waits for one.
    """

    def __init__(self, sam):
        patch_single_decode(sam)
        self.sam, self.job, self.token = sam, None, None
        self.lock, self.stop = threading.Lock(), False
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while not self.stop:
            with self.lock:
                job, self.job = self.job, None
            if job is None:
                time.sleep(0.005)
                continue
            tok = sam3db_token(self.sam, *job)
            with self.lock:
                self.token = tok[0]

    def submit(self, frame, bbx):
        with self.lock:
            self.job = (frame, bbx)
            return self.token

    def reset(self):
        with self.lock:
            self.job, self.token = None, None


class View3D:
    """Optional open3d window showing the same skeleton in 3D camera space."""

    def __init__(self):
        import open3d as o3d

        self.o3d = o3d
        bones = [(p, c) for c, p in enumerate(PARENTS_77) if p >= 0]
        self.lines = o3d.geometry.LineSet()
        self.lines.points = o3d.utility.Vector3dVector(np.zeros((77, 3)))
        self.lines.lines = o3d.utility.Vector2iVector(bones)
        self.lines.colors = o3d.utility.Vector3dVector(
            [[v / 255 for v in _PART_COLORS_77[_JOINT_GROUP_77[c]][::-1]] for _, c in bones]
        )
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window("SOMA 3D", 720, 720)
        self.vis.add_geometry(self.lines)
        self.reset = True

    def update(self, joints):
        joints = joints * np.array([1.0, -1.0, -1.0])  # camera frame is y-down / z-forward
        self.lines.points = self.o3d.utility.Vector3dVector(joints)
        self.vis.update_geometry(self.lines)
        if self.reset:  # frame the skeleton once it exists
            self.vis.reset_view_point(True)
            self.reset = False
        self.vis.poll_events()
        self.vis.update_renderer()


class Camera:
    """Capture thread keeping only the newest frame, so inference never lags behind."""

    def __init__(self, index):
        self.index = index
        self.cap = cv2.VideoCapture(index)
        assert self.cap.isOpened(), f"cannot open camera {index}"
        self.frame = None
        self.lock = threading.Lock()
        self.stop = False
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        misses = 0
        while not self.stop:
            ok, frame = self.cap.read()
            if not ok:
                # a USB hiccup used to end the demo outright; reopen instead
                misses += 1
                if misses > 30:
                    self.cap.release()
                    time.sleep(0.5)
                    self.cap = cv2.VideoCapture(self.index)
                    misses = 0
                time.sleep(0.03)
                continue
            misses = 0
            with self.lock:
                self.frame = frame

    def read(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--window", type=int, default=64, help="sliding window length")
    p.add_argument("--min_window", type=int, default=16, help="frames before 3D inference starts")
    p.add_argument("--infer_every", type=int, default=1, help="run the denoiser every N frames")
    p.add_argument(
        "--no_imgfeat",
        action="store_true",
        help="skip SAM-3D-Body conditioning: faster, noticeably worse when the body is cropped",
    )
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--flip", action="store_true", help="mirror the display")
    p.add_argument("--view3d", action="store_true", help="also open a 3D skeleton window")
    p.add_argument("--robot", choices=["g1", "k1"], help="retarget and show the robot in MuJoCo")
    p.add_argument("--soma_rot", action="store_true",
                   help="use SOMA's joint rotations as orientation targets (unverified)")
    p.add_argument("--mirror", action="store_true",
                   help="mirror the robot so it behaves like a reflection of the operator")
    p.add_argument("--frames", action="store_true",
                   help="enable orientation targets (measured worse; see test_apose.py)")
    p.add_argument("--stream", type=int, metavar="PORT",
                   help="serve the view over HTTP instead of opening local windows "
                        "(camera on PORT, robot on PORT+1)")
    p.add_argument("--mlp", metavar="PT",
                   help="distilled-network checkpoint to retarget with. Defaults to "
                        "models/<robot>_retarget.pt when that exists, so the network is "
                        "the normal path and the IK is the fallback — on K1 the network "
                        "reads 10.2 deg against soma-retargeter where the IK reads 26.8.")
    p.add_argument("--ik", action="store_true",
                   help="force the PyRoki IK solver instead of the distilled network")
    p.add_argument("--smooth", type=float, default=2.0, metavar="HZ",
                   help="low-pass the network's joint angles at this cutoff. Free at the "
                        "default: at the demo's real 15 fps the holdout goes 9.01 -> 8.93 "
                        "deg while the worst single-frame pop goes 238 -> 108. 0 disables; "
                        "1.5 cuts the pops to 90 for +0.03 deg.")
    p.add_argument("--log", metavar="CSV",
                   help="record one row per frame (wall time, frame gap, per-stage ms, "
                        "tracking state, whether the worker took the frame) and print a "
                        "session summary on exit. The frame gap is the column that "
                        "diagnoses a freeze — throughput does not.")
    p.add_argument("--motion_command", metavar="NPZ",
                   help="capture the retargeted motion as a reference clip. Recording "
                        "starts at Calibrate and runs to exit, so the clip only ever "
                        "contains frames measured against the current operator.")
    p.add_argument("--kp2d_mlp", metavar="PT",
                   help="the stage-cut student: VitPose 2D window -> joint angles + the "
                        "14 targets, skipping the GEM denoiser and SOMA FK entirely. "
                        "~30 Hz instead of ~17, at 15.6 deg vs the full path's 11.5.")
    args = p.parse_args()
    if args.mlp and args.kp2d_mlp:
        p.error("--mlp and --kp2d_mlp are different retarget sources; pick one")
    if args.ik and (args.mlp or args.kp2d_mlp):
        p.error("--ik forces the solver; drop --mlp/--kp2d_mlp")
    # The network is the default path. Fall back to the IK only when asked (--ik), when
    # another student is chosen, or when no checkpoint has been trained for this robot.
    if not (args.ik or args.mlp or args.kp2d_mlp) and args.robot:
        default_ck = HERE / "models" / f"{args.robot}_retarget.pt"
        if default_ck.exists():
            args.mlp = str(default_ck.relative_to(HERE))
            print(f"[retarget] {args.mlp} (--ik for the solver instead)", flush=True)
        else:
            print(f"[retarget] no {default_ck.name}; falling back to the IK solver",
                  flush=True)
    if (args.mlp or args.kp2d_mlp) and args.mirror:
        p.error("--mlp joint angles are not mirrored; drop --mirror")

    from gem.utils.vitpose_extractor import VitPoseExtractor

    vitpose = VitPoseExtractor(device="cuda:0", pose_type="soma", tqdm_leave=False)
    vitpose.flip_test = False
    # the kp2d student consumes VitPose directly: no GEM model, no SAM-3D-Body, and
    # 20+ s less startup — the denoiser stages simply do not exist in this mode
    model = None if args.kp2d_mlp else build_model(args.ckpt)
    worker = None
    if not args.no_imgfeat and model is not None:
        from gem.utils.sam3db_extractor import SAM3DBExtractor

        worker = TokenWorker(SAM3DBExtractor(device="cuda:0"))

    cam = Camera(args.camera)
    while cam.read() is None:
        time.sleep(0.05)
    H, W = cam.read().shape[:2]
    K = estimate_K(W, H)
    full_bbx = get_bbx_xys_from_xyxy(torch.tensor([[0.0, 0.0, W - 1.0, H - 1.0]]))[0].float()

    kp2d_win = deque(maxlen=args.window)
    bbx_win = deque(maxlen=args.window)
    tok_win = deque(maxlen=args.window)
    bbx, xy3d, i, fps = bootstrap_bbox(vitpose, cam.read(), W, H), None, 0, 0.0
    view3d = View3D() if args.view3d else None
    # Calibration is a deliberate act now: the operator stands in a T-pose and presses the
    # button, instead of the session being defined by whatever the first frames contained.
    # The HTTP handler only raises this flag; the loop below does the work, because the
    # identity model and the IK link belong to the loop's thread.
    recalibrate = [False]
    soma = (None if model is None else
            (model.body_model.soma if hasattr(model.body_model, "soma") else model.body_model))

    stream = None
    if args.stream:
        from mjpeg import Streamer

        # the robot panel is served by the IK worker on the next port; the page pulls both
        stream = Streamer(args.stream, peer=(args.stream + 1) if args.robot else None,
                          on_calibrate=lambda: recalibrate.__setitem__(0, True))
    ik, ik_ndof, mlp = None, 0, None
    if args.robot:
        import subprocess

        ik_ndof = {"g1": 29, "k1": 23}[args.robot]
        ik = IKLink(
            [str(HERE / ".venv-ik/bin/python"), "ik_server.py", "--robot", args.robot]
            + (["--stream", str(args.stream + 1)] if args.stream else ["--view"])
            + (["--frames"] if args.frames else [])
            + (["--mirror"] if args.mirror else [])
            + (["--mlp"] if (args.mlp or args.kp2d_mlp) else [])
            + (["--motion_command", args.motion_command] if args.motion_command else []),
            HERE, ik_ndof,
        )
    kp2d_net = None
    if args.mlp or args.kp2d_mlp:
        # The CPU nets are tiny, but torch's default intra-op pool (one thread per core)
        # thrashes when Isaac has the cores: the 3M-param student measured 60 ms at 24
        # threads against 0.09 ms at 2. Cap it — the GPU pipeline never needed them.
        torch.set_num_threads(2)
    if args.kp2d_mlp:
        ck = torch.load(HERE / args.kp2d_mlp, map_location="cpu", weights_only=False)
        w, nl = ck["width"], ck.get("layers", 3)
        mods = [torch.nn.Linear(len(ck["mu"]), w), torch.nn.GELU()]
        for _ in range(nl - 2):
            mods += [torch.nn.Linear(w, w), torch.nn.GELU()]
        mods += [torch.nn.Linear(w, ik_ndof + 42)]
        kp2d_net = torch.nn.Sequential(*mods)
        kp2d_net.load_state_dict(ck["state"])
        kp2d_net.eval()
        k_mu = torch.tensor(np.asarray(ck["mu"]), dtype=torch.float32)
        k_sd = torch.tensor(np.asarray(ck["sd"]), dtype=torch.float32)
        k_hist = deque(maxlen=9)  # enough for lags 0,2,4,8

        def norm_kp(k):
            xy, c = k[:, :2], k[:, 2:]
            xy = xy - xy[[68, 73]].mean(0)
            return np.concatenate([xy / (np.abs(xy).max() + 1e-6), c], 1).ravel()

        def kp2d_student(kp2d):
            k_hist.append(norm_kp(kp2d).astype(np.float32))
            h = list(k_hist)
            feats = np.concatenate([h[-1 - min(lag, len(h) - 1)] for lag in (0, 2, 4, 8)])
            with torch.no_grad():
                out = kp2d_net((torch.tensor(feats) - k_mu) / k_sd).numpy()
            return out[:ik_ndof].astype("<f4"), out[ik_ndof:].reshape(14, 3).astype("<f4")

    if args.mlp:
        # tiny (0.25M param) distilled net; CPU on purpose so it never contends with
        # perception for the GPU, same reasoning as the IK worker
        ck = torch.load(HERE / args.mlp, map_location="cpu", weights_only=False)
        assert len(ck["names"]) == ik_ndof, f"model is for {len(ck['names'])} DOF, robot has {ik_ndof}"
        mlp_net = build_student(ck, ik_ndof)
        if "state" in ck:
            mlp_net.load_state_dict(_remap_block_keys(ck["state"]))
        mlp_net.eval()
        mlp_parts = str(ck.get("arch", "plain")) == "parts"   # normalises per expert
        mlp_mu = None if mlp_parts else torch.tensor(np.asarray(ck["mu"]), dtype=torch.float32)
        mlp_sd = None if mlp_parts else torch.tensor(np.asarray(ck["sd"]), dtype=torch.float32)
        mlp_feats = str(ck.get("features", "pose"))  # pose_conf models also read confidence

        # The student's frame-to-frame noise is larger than the teacher's, so low-passing
        # the output moves it *towards* the teacher -- the holdout improves rather than
        # degrades, at 15 and 30 fps alike, while the worst single-frame pop more than
        # halves. Those pops are shoulder pitch/yaw and elbow: the student amplifies a
        # 56 deg teacher discontinuity into 122. beta stays 0 on purpose, since one-euro's
        # speed adaptation widens the cutoff exactly when the pops happen and passes them.
        smooth = OneEuro(ik_ndof, min_cutoff=args.smooth, beta=0.0) if args.smooth else None
        t_smooth = [None]

        def mlp(bp, conf):
            x = bp
            if mlp_feats == "pose_conf":
                x = torch.cat([bp, torch.tensor(conf, dtype=torch.float32)])
            q = (mlp_net(x) if mlp_parts else mlp_net((x - mlp_mu) / mlp_sd)).numpy()
            if smooth is not None:
                now = time.time()
                # real dt, so the cutoff means the same thing at 16 fps as at 30
                # capped: a stalled frame must not widen the cutoff just as the pose jumps
                dt = 1 / 30 if t_smooth[0] is None else min(max(now - t_smooth[0], 1e-3), 0.15)
                t_smooth[0] = now
                q = smooth(q, dt)
            return q.astype("<f4")

    calib_left, calib_send = 0, False  # prompt frames left; request still to reach the worker
    LOST_AFTER = 3          # consecutive frames without a box before the window is dropped
    n_miss, n_reset = 0, 0
    logf, log_rows, t_prev, n_calib, n_drop = None, [], None, 0, 0
    if args.log:
        logf = open(HERE / args.log, "w", buffering=1)
        logf.write("frame,wall,gap_ms,vitpose_ms,predict_ms,fk_ms,retarget_ms,loop_ms,"
                   "tracked,sent,calib\n")
    t_start = time.time()
    try:
      while not cam.stop:
          frame = cam.read()
          t0 = time.time()
          gap_ms = 0.0 if t_prev is None else (t0 - t_prev) * 1000.0
          t_prev = t0
          st = {"vitpose": 0.0, "predict": 0.0, "fk": 0.0, "retarget": 0.0}
          sent_ok, tracked = 0, 1
          if recalibrate[0]:
              recalibrate[0] = False
              n_calib += 1
              if soma is not None:
                  soma._identity_frozen = False  # next FK re-measures the person's rest shape
              calib_left, calib_send = CALIB_FRAMES, True

          _ts = time.time()
          with torch.autocast("cuda", dtype=torch.float16):
              kp2d = vitpose.extract(frame[None], bbx[None], path_type="np", batch_size=1)[0]
          st["vitpose"] = (time.time() - _ts) * 1000.0
          kp2d_win.append(kp2d.cuda())
          bbx_win.append(bbx.cuda())
          if worker is not None:
              token = worker.submit(frame, bbx)  # returns the latest finished token
              if token is not None:
                  tok_win.append(token)
          nbx = bbox_from_kps(kp2d, W, H)
          if nbx is None:
              tracked = 0
              n_miss += 1
          else:
              n_miss = 0
          if nbx is None and n_miss < LOST_AFTER:
              # A single bad frame is not a lost person. Measured over 258 s of live use:
              # 149 dropouts, 143 of them exactly one frame — and each one was clearing the
              # 64-frame window and resetting the token worker, so the denoiser restarted
              # from 16 fresh frames roughly every 1.7 s and the pose visibly jumped. Hold
              # the last box instead; `tracked` still records the miss.
              nbx = bbx
          if nbx is None:  # really gone, re-detect from scratch
              # one pass, not three: re-detection happens mid-stream and a 1.7 s stall is
              # worse than a slightly worse box, which the next frames correct anyway
              nbx, xy3d = bootstrap_bbox(vitpose, frame, W, H, rounds=1), None
              kp2d_win.clear()
              bbx_win.clear()
              tok_win.clear()
              n_reset, n_miss = n_reset + 1, 0   # so a long absence re-detects every
              if worker is not None:             # LOST_AFTER frames, not every frame
                  worker.reset()
          bbx = nbx

          if kp2d_net is not None and ik is not None:
              # stage-cut path: no denoiser, no FK — the student answers from the 2D window
              q_pred, t14_pred = kp2d_student(kp2d.numpy())
              conf = kp2d[SOMA_IDX, 2].numpy().astype("<f4")
              if calib_send:
                  conf[0] = -abs(conf[0])  # same sign-bit calibration request as below
              payload = (t14_pred.tobytes() + conf.tobytes()
                         + np.zeros((14, 3, 3), "<f4").tobytes() + q_pred.tobytes())
              sent = ik.submit(payload)
              sent_ok = int(sent); n_drop += (not sent)
              calib_send = calib_send and not sent
          if model is not None and len(kp2d_win) >= args.min_window and i % args.infer_every == 0:
              # frames before the first token arrived reuse it, same as any held token
              f_img = None
              if tok_win:
                  pad = [tok_win[0]] * (len(kp2d_win) - len(tok_win))
                  f_img = torch.stack(pad + list(tok_win))
              _ts = time.time()
              pred = model.predict(
                  make_data(kp2d_win, bbx_win, K, f_img), static_cam=True, postproc=False
              )
              st["predict"] = (time.time() - _ts) * 1000.0
              _ts = time.time()
              xy3d, joints3d, jrot = project_last_frame(model, pred, K)
              st["fk"] = (time.time() - _ts) * 1000.0
              if view3d is not None:
                  view3d.update(joints3d)
              if ik is not None:
                  # camera-frame joints; the worker converts to the robot world and solves
                  conf = kp2d[SOMA_IDX, 2].numpy().astype("<f4")
                  if calib_send:
                      # Ask the worker to re-measure scales and floor. The sign of the first
                      # confidence carries the request, so the fixed-size protocol is
                      # unchanged and the value itself survives; the worker takes abs().
                      conf[0] = -abs(conf[0])  # the worker reads the sign bit, so -0.0 counts
                  _ts = time.time()
                  rot = (jrot if (args.soma_rot and jrot is not None)
                         else np.zeros((14, 3, 3), np.float32))
                  payload = (joints3d[SOMA_IDX].astype("<f4").tobytes()
                             + conf.tobytes() + rot.astype("<f4").tobytes())
                  if mlp is not None:
                      with torch.no_grad():
                          bp = pred["body_params_incam"]["body_pose"][-1].reshape(-1).float().cpu()
                          # kp2d confidences, not `conf` — that copy's sign bit doubles as
                          # the calibration signal and must not reach the network
                          payload += mlp(bp, kp2d[SOMA_IDX, 2].numpy()).tobytes()
                  # Retry until a submit lands, then stop. IKLink drops frames when the
                  # worker is behind, so the request can miss; but sending it for a whole
                  # window overshoots the worker's own 15-frame measurement and kicks off a
                  # second calibration as soon as the first one finishes.
                  sent = ik.submit(payload)
                  sent_ok = int(sent)
                  n_drop += (not sent)
                  st["retarget"] = (time.time() - _ts) * 1000.0
                  calib_send = calib_send and not sent

          if xy3d is not None:
              frame = draw_skeleton(frame, xy3d)
          else:  # warm-up / re-detect: show the raw 2D keypoints so the view is never blank
              frame = draw_skeleton(frame, kp2d[:, :2].numpy(), conf=kp2d[:, 2].numpy())

          fps = 0.9 * fps + 0.1 / max(time.time() - t0, 1e-6)
          cv2.putText(
              frame, f"{fps:5.1f} FPS", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
          )
          if i % 30 == 0:
              print(f"{fps:5.1f} FPS", flush=True)
          shown = np.ascontiguousarray(frame[:, ::-1]) if args.flip else frame
          if calib_left > 0:  # after the flip, or the prompt comes out mirrored
              calib_left -= 1
              cv2.putText(shown, "CALIBRATING - stand in a T-pose", (10, H - 20),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
          if stream is not None:
              stream.put(shown)
          else:
              # local window: the Calibrate button lives on the streamed page, so bind the
              # same request to a key here
              cv2.putText(shown, "c: calibrate   q: quit", (10, H - 12),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
              cv2.imshow("GEM-X SOMA skeleton", shown)
              k = cv2.waitKey(1) & 0xFF
              if k == ord("q"):
                  break
              if k == ord("c"):
                  recalibrate[0] = True
          if logf is not None:
              loop_ms = (time.time() - t0) * 1000.0
              log_rows.append((gap_ms, loop_ms, st["vitpose"], st["predict"], st["fk"]))
              logf.write(f"{i},{t0 - t_start:.3f},{gap_ms:.1f},{st['vitpose']:.1f},"
                         f"{st['predict']:.1f},{st['fk']:.1f},{st['retarget']:.1f},"
                         f"{loop_ms:.1f},{tracked},{sent_ok},{1 if calib_left else 0}\n")
          i += 1

    except KeyboardInterrupt:
        print("\n[log] interrupted", flush=True)
    if logf is not None:
        logf.close()
        _log_summary(args.log, log_rows, i, time.time() - t_start, n_calib, n_drop, n_reset)
    cam.stop = True
    if ik is not None:
        ik.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
