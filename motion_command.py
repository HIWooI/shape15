"""Turn solved IK poses into the reference motion a whole-body policy expects.

The policy observes, per step:
    motion_command      46 = 23 reference joint positions + 23 reference joint velocities
    motion_anchor_ori_b  6 = reference torso orientation vs the robot's (first two columns
                             of the rotation matrix), so we supply the reference half

It does not take a root position, so the floor height and hip re-centring the display path
cares about are irrelevant here.

Velocity is the reason this module exists. It comes from differencing consecutive samples,
which divides any jitter in the targets by dt and amplifies it — so the joint angles are
filtered *before* differencing, and resampled to the policy's rate first, otherwise a
reference held flat between our frames alternates between zero and a spike.
"""

import numpy as np


class OneEuro:
    """One-euro filter: smooths when still, gets out of the way when moving fast.

    A fixed low-pass would either leave jitter in or add lag to fast motion; this trades
    between the two by widening its cutoff with the observed speed.
    """

    def __init__(self, n, min_cutoff=1.0, beta=0.3, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x_prev = np.zeros(n)
        self.dx_prev = np.zeros(n)
        self.started = False

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, dt):
        if not self.started:
            self.x_prev, self.started = x.copy(), True
            return x.copy()
        dx = (x - self.x_prev) / max(dt, 1e-6)
        dx = self._alpha(self.d_cutoff, dt) * dx + (1 - self._alpha(self.d_cutoff, dt)) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * np.abs(dx)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev, self.dx_prev = x_hat, dx
        return x_hat


class MotionCommand:
    """Resample solved joint angles onto a fixed policy rate and emit position+velocity.

    Push samples whenever the IK produces one (our loop runs at 12-30 Hz, unevenly); pull
    steps at the policy rate. Between pushes the pull interpolates rather than holding the
    last value, so the velocity stays continuous.
    """

    # What the downstream policy was trained on, not what the motors can do. shape14
    # excluded a clip at 114 rad/s and its safe recorded clip ran 4.6-12.6 rad/s, so a
    # reference above that band is outside the distribution the policy ever saw. The URDF
    # limits (11.5-20.9) describe the actuators and are the wrong ceiling for a reference.
    POLICY_MAX_VEL = 12.6

    def __init__(self, ndof, rate=50.0, joint_names=None, vel_limits=None):
        self.dt = 1.0 / rate
        self.rate = rate
        self.joint_names = list(joint_names) if joint_names is not None else None
        # A reference the robot cannot physically follow is worse than a late one: an IK
        # solution that flips branch between frames differences into hundreds of rad/s.
        # Clamp the step to what the joint can actually do (URDF limits when we have them).
        lim = (np.asarray(vel_limits, np.float64) if vel_limits is not None
               else np.full(ndof, 10.0))
        self.max_step = np.minimum(lim, self.POLICY_MAX_VEL) * self.dt
        self.t0 = None
        self.filter = OneEuro(ndof)
        self.prev = None  # (t, q) most recent push
        self.last = None  # (t, q) push before that
        self.t_next = None
        self.q_prev_out = None
        self.steps = []  # (t, q, qdot, torso_ori6)

    def reset(self):
        """Throw away what has been recorded and start over from the next push.

        Bound to the Calibrate button: everything before it was captured with the
        previous person's bone scales, so it is not a reference for anyone.
        """
        self.filter = OneEuro(len(self.max_step))
        self.prev = self.last = self.t_next = self.q_prev_out = None
        self.steps = []

    @staticmethod
    def ori6(R):
        """First two columns of a rotation matrix, as the policy's 6D convention."""
        return np.asarray(R, np.float32)[:, :2].T.reshape(-1)

    def push(self, t, q, torso_R, root=None):
        if self.t0 is None:
            self.t0 = t
        t = t - self.t0  # relative: absolute unix time loses all resolution in float32
        q = np.asarray(q, np.float64)
        if self.prev is None:
            self.prev, self.t_next = (t, q, torso_R, root), t
            return []
        self.last, self.prev = self.prev, (t, q, torso_R, root)
        out = []
        while self.t_next <= self.prev[0]:  # emit every policy step we have data for
            t0, q0, R0, root0 = self.last
            t1, q1, R1, root1 = self.prev
            u = 0.0 if t1 <= t0 else (self.t_next - t0) / (t1 - t0)
            qi = self.filter((1 - u) * q0 + u * q1, self.dt)
            if self.q_prev_out is not None:
                step_lim = np.clip(qi - self.q_prev_out, -self.max_step, self.max_step)
                qi = self.q_prev_out + step_lim
            Ri = R0 if u < 0.5 else R1  # orientation: nearest, not interpolated
            rooti = root0 if u < 0.5 else root1
            qdot = np.zeros_like(qi) if self.q_prev_out is None else (qi - self.q_prev_out) / self.dt
            self.q_prev_out = qi
            step = (self.t_next, qi.astype(np.float32), qdot.astype(np.float32),
                    self.ori6(Ri), rooti)
            self.steps.append(step)
            out.append(step)
            self.t_next += self.dt
        return out

    def save(self, path):
        t = np.array([s[0] for s in self.steps], np.float32)
        np.savez(
            path,
            t=t,
            joint_pos=np.stack([s[1] for s in self.steps]) if self.steps else np.zeros((0,)),
            joint_vel=np.stack([s[2] for s in self.steps]) if self.steps else np.zeros((0,)),
            torso_ori6=np.stack([s[3] for s in self.steps]) if self.steps else np.zeros((0,)),
            root=np.stack([np.asarray(s[4], np.float32) if s[4] is not None
                           else np.array([0, 0, 0, 0, 0, 0, 1], np.float32)
                           for s in self.steps]) if self.steps else np.zeros((0,)),
            joint_names=np.array(self.joint_names or [], dtype=object),
            rate=self.rate,
        )
        return len(self.steps)


def _self_check():
    """A ramp in, a constant velocity out — and jitter must not blow the velocity up."""
    mc = MotionCommand(3, rate=50.0)
    for i in range(30):  # 15 Hz input, 1 rad/s ramp on joint 0
        t = i / 15.0
        q = np.array([t, 0.0, 0.0])
        mc.push(t, q, np.eye(3))
    v = np.stack([s[2] for s in mc.steps])[5:, 0]
    assert 0.5 < v.mean() < 1.5, f"ramp velocity should be ~1 rad/s, got {v.mean():.2f}"

    jump = MotionCommand(3, rate=50.0, vel_limits=[11.5, 11.5, 11.5])
    for i in range(20):
        jump.push(i / 15.0, np.array([0.0 if i < 10 else 3.0, 0.0, 0.0]), np.eye(3))
    vj = np.abs(np.stack([s[2] for s in jump.steps])).max()
    cap = min(11.5, MotionCommand.POLICY_MAX_VEL)
    assert vj <= cap + 1e-3, f"branch flip leaked {vj:.1f} rad/s past {cap}"

    noisy = MotionCommand(3, rate=50.0)
    rng = np.random.RandomState(0)
    for i in range(30):
        t = i / 15.0
        noisy.push(t, np.array([0.0, 0.0, 0.0]) + rng.randn(3) * 0.01, np.eye(3))
    vn = np.abs(np.stack([s[2] for s in noisy.steps])).max()
    assert vn < 5.0, f"jitter amplified into {vn:.1f} rad/s"

    assert MotionCommand.ori6(np.eye(3)).tolist() == [1, 0, 0, 0, 1, 0]
    assert len(mc.steps[0]) == 5 and mc.steps[0][4] is None  # root is optional

    # demo_webcam's --smooth passes the *measured* dt rather than a nominal frame period,
    # because the camera rate moves (16-30 fps depending on what else has the GPU) and a
    # fixed dt would silently change how much smoothing the same cutoff applies. A
    # first-order filter is only exactly rate-independent in the continuous limit, so the
    # check is that the measured-dt path tracks that limit better than a fixed-dt one.
    T, cut = 0.25, 3.0
    want = 1 - np.exp(-T * 2 * np.pi * cut)
    fps, assumed = 16.0, 30.0
    real, wrong = OneEuro(1, cut, beta=0.0), OneEuro(1, cut, beta=0.0)
    real(np.zeros(1), 1 / fps)
    wrong(np.zeros(1), 1 / assumed)
    for _ in range(int(fps * T)):
        a, b = real(np.ones(1), 1 / fps), wrong(np.ones(1), 1 / assumed)
    assert abs(a[0] - want) < abs(b[0] - want), (
        f"measured dt should beat assuming {assumed:.0f} fps: {a[0]:.3f} vs {b[0]:.3f}, "
        f"continuous {want:.3f}")
    print(f"ok: ramp {v.mean():.2f} rad/s, jitter peak {vn:.2f}, clamped jump {vj:.2f} rad/s, "
          f"dt-aware step {a[0]:.3f} vs fixed-dt {b[0]:.3f} (continuous {want:.3f})")


if __name__ == "__main__":
    _self_check()
