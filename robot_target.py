"""Turn GEM-X SOMA poses into robot joint targets, robot-agnostic.

The retargeting pipeline takes whole motions (`add_input_motions` + `execute`), so a
live loop cannot hand it one frame at a time: callers accumulate a chunk of poses and
retarget them together.

Measured on an RTX 5090, 30-frame chunks:
    G1  ~32 ms/frame   -> ~1.0 s behind, and blocks the caller for ~1 s per chunk
    K1  ~1180 ms/frame -> ~35 s behind, i.e. not usable live as configured

A background *thread* does not work: the G1 path captures a CUDA graph
(`newton_pipeline.py:2322`, `feet_stabilizer.py:241`) and torch issuing work on the
legacy stream meanwhile raises `cudaErrorStreamCaptureImplicit`. Warp exposes no
global switch for this, so decoupling the retarget from perception needs a separate
*process*, not a thread.
"""




import torch

ROBOTS = {  # name -> (soma-retargeter robot type, config file)
    "g1": ("unitree_g1", "soma_to_g1_retargeter_config.json"),
    "k1": ("ai_sapiens", "soma_to_ai_sapiens_retargeter_config.json"),
}


class Retargeter:
    """One robot's retargeting pipeline, built once and reused across chunks."""

    def __init__(self, robot, fps=30.0):
        import warp as wp
        from scripts.demo import retarget_utils as RU
        from soma_retargeter.pipelines.newton_pipeline import NewtonPipeline
        from soma_retargeter.utils import io_utils as retargeter_io
        from soma_retargeter.utils.space_conversion_utils import FacingDirectionType, SpaceConverter

        if robot not in ROBOTS:
            raise ValueError(f"unknown robot {robot!r}, expected one of {sorted(ROBOTS)}")
        self.robot, self.fps = robot, fps
        self.RU, self.NewtonPipeline, self.pipe, self.skel = RU, NewtonPipeline, None, None
        self.robot_type, config_file = ROBOTS[robot]
        if robot == "k1":
            # K1 needs its packaged MJCF resolved and verified
            self.cfg = RU._k1_retarget_config()
        else:
            self.cfg = retargeter_io.load_json(
                retargeter_io.get_config_file(self.robot_type, config_file)
            )
        self.offset = SpaceConverter(FacingDirectionType.MUJOCO).transform(wp.transform_identity())

    def run(self, params):
        """params: list of per-frame global body-param dicts. Returns CSVAnimationBuffer."""
        g = {k: torch.cat([p[k] for p in params]) for k in params[0]}
        if self.pipe is None:
            # The pipeline is bound to one skeleton. It is the same person throughout, so
            # fix the identity on the first chunk and reuse it for the rest of the session.
            self.skel = self.RU.build_soma_skeleton_from_model(
                g["identity_coeffs"], g["scale_params"]
            )
            self.pipe = self.NewtonPipeline(
                self.skel, "soma", self.robot_type, retarget_config=self.cfg
            )
        else:
            self.pipe.clear()
        anim = self.RU.soma_params_to_animation_buffer(g, self.skel, self.fps)
        self.pipe.add_input_motions([anim], [self.offset], scale_animation=True)
        return self.pipe.execute()[0]
