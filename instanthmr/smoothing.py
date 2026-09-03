"""Temporal smoothing of per-person HMR predictions.

InstantHMR is a frame-by-frame model: every frame is estimated independently,
so its outputs carry per-frame noise that reads as jitter in a video — most
visibly in ``cam_trans``, which is a free 3-vector from a linear head with
almost nothing tying it to the observed pixels.

A non-learned moving-average filter over a short window removes much of that.
NLF (arXiv:2407.07532, Tables 2-3) reports a 5-frame filter of this kind
improving 3DPW MPJPE 59.0 -> 57.2 mm and EMDB 68.4 -> 66.7 mm, at zero
training cost. This module is that filter.

Two things the filter needs that the pipeline does not provide:

* **Identity across frames.** ``PosePipeline.predict`` returns an unordered
  list of detections per frame, so a person's index is not stable. We attach a
  minimal greedy IoU tracker (:class:`_Tracker`) purely to decide which
  predictions belong to the same person before averaging them.
* **Lookahead**, for ``mode="centered"``. A centred window has no phase lag
  but needs ``(window - 1) // 2`` future frames, so :meth:`TemporalSmoother.push`
  returns results delayed by that many frames and :meth:`flush` drains the
  tail. ``mode="causal"`` trades phase lag for zero latency.

The moving average has one structural weakness: it is a *fixed*-cutoff filter,
so it cannot tell jitter from fast intentional motion and lags during both. On
a waving hand that is plainly visible. ``mode="1euro"`` fixes it with the
1 Euro filter (Casiez, Roussel & Vogel, CHI 2012): an exponential smoother
whose cutoff frequency rises with the signal's own speed. At rest the cutoff is
low and jitter is crushed; during fast motion the cutoff opens up and the
output tracks. It is causal, needs no lookahead, and is the right choice for
demos and live camera. The centred moving average remains the right choice for
benchmark numbers, since that is the filter NLF measured.

Caveat on ``mhr_params``: these are MHR joint parameters (angles and scales),
and this filter averages them linearly. That is well-behaved for continuous
motion but is *wrong* across an angle wraparound — a global yaw crossing +/-pi
will be averaged toward zero instead of through the discontinuity, producing a
brief flip. Short windows on real footage rarely hit it; if you see one, drop
``mhr_params`` from ``SMOOTHED_FIELDS`` and accept a mesh that jitters while
the keypoints do not.
"""

from __future__ import annotations

from collections import deque
from dataclasses import replace
from typing import Iterator, Sequence

import numpy as np

from .inference import HMRPrediction

#: Fields averaged across the window. All are Euclidean quantities for which a
#: linear mean is the right operation, except ``mhr_params`` — see the module
#: docstring. ``joints_3d_cam`` is deliberately absent: it is recomputed from
#: the smoothed ``joints_3d_local`` and ``cam_trans`` so the three stay
#: consistent with each other. ``joints_3d_local`` is filtered only when no
#: ``keypoint_fn`` is available; with one, it is re-derived from the filtered
#: ``mhr_params`` so the keypoints and the mesh cannot disagree.
SMOOTHED_FIELDS = (
    "joints_3d_local",
    "cam_trans",
    "joints_2d",
    "mhr_params",
    "shape_params",
)


#: Per-field ``(mincutoff_hz, beta)`` for ``mode="1euro"``.
#:
#: ``beta`` scales a *speed*, so its right value depends on the field's units —
#: this is the classic tuning trap with the 1 Euro filter. ``joints_2d`` is in
#: full-frame pixels (a fast hand runs ~500-1500 px/s), the 3D fields are in
#: metres (~1-3 m/s), ``mhr_params`` in radians and scale units. The values
#: below put all of them at a comparable cutoff (~9 Hz) during fast motion
#: while holding ~1 Hz at rest. ``shape_params`` gets beta 0 on purpose:
#: identity does not change, so it should be filtered as hard as possible —
#: the per-track limit of that idea is NLF's shared-beta-per-track fitting.
ONE_EURO_DEFAULTS: dict[str, tuple[float, float]] = {
    "joints_2d":       (1.0, 0.01),
    "joints_3d_local": (1.0, 4.0),
    "cam_trans":       (0.6, 2.0),
    "mhr_params":      (1.0, 2.0),
    "shape_params":    (0.3, 0.0),
}


class _OneEuro:
    """1 Euro filter (Casiez, Roussel & Vogel, CHI 2012), vectorised.

    Operates elementwise on an array of any shape, so every joint — indeed
    every coordinate — adapts on its own speed. That is the property that
    matters here: a still torso stays heavily smoothed in the same frame where
    a waving hand is barely smoothed at all.
    """

    def __init__(self, mincutoff: float, beta: float, dcutoff: float = 1.0):
        self.mincutoff = float(mincutoff)
        self.beta = float(beta)
        self.dcutoff = float(dcutoff)
        self._x: np.ndarray | None = None
        self._dx: np.ndarray | None = None
        self._t: float | None = None

    @staticmethod
    def _alpha(cutoff, dt: float):
        tau = 1.0 / (2.0 * np.pi * np.maximum(cutoff, 1e-6))
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if self._x is None:
            self._x, self._dx, self._t = x, np.zeros_like(x), t
            return x
        dt = t - self._t
        if not (dt > 0.0):
            # Repeated or non-monotonic timestamps would blow up the derivative;
            # fall back to a nominal 30 fps step rather than dividing by zero.
            dt = 1.0 / 30.0
        self._t = t

        a_d = self._alpha(self.dcutoff, dt)
        self._dx = a_d * ((x - self._x) / dt) + (1.0 - a_d) * self._dx
        a = self._alpha(self.mincutoff + self.beta * np.abs(self._dx), dt)
        self._x = a * x + (1.0 - a) * self._x
        return self._x


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection-over-union of two ``[x1, y1, x2, y2]`` boxes."""
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


class _Tracker:
    """Greedy IoU association of detections to persistent track ids.

    Deliberately minimal — it exists only to group predictions of the same
    person before averaging, not to be a general-purpose tracker. Tracks
    survive ``max_age`` frames without a match so a brief miss does not restart
    the filter (and reset the smoothing) for that person.
    """

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 5):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self._boxes: dict[int, np.ndarray] = {}
        self._age: dict[int, int] = {}
        self._next_id = 0

    def assign(self, persons: Sequence[HMRPrediction]) -> list[int]:
        """Return one track id per person, in the order given."""
        ids: list[int] = [-1] * len(persons)
        free = set(self._boxes)

        # Score every (detection, track) pair, then take them best-first. With
        # the handful of people this pipeline handles, an O(n*m) sort is far
        # cheaper than pulling in a Hungarian solver.
        pairs = [
            (_iou(p.bbox, self._boxes[t]), i, t)
            for i, p in enumerate(persons)
            for t in free
        ]
        pairs.sort(reverse=True)
        taken: set[int] = set()
        for score, i, t in pairs:
            if score < self.iou_threshold:
                break
            if ids[i] != -1 or t not in free or i in taken:
                continue
            ids[i] = t
            free.discard(t)
            taken.add(i)

        for i, p in enumerate(persons):
            if ids[i] == -1:
                ids[i] = self._next_id
                self._next_id += 1
            self._boxes[ids[i]] = np.asarray(p.bbox, dtype=np.float64)
            self._age[ids[i]] = 0

        for t in list(free):
            self._age[t] = self._age.get(t, 0) + 1
            if self._age[t] > self.max_age:
                self._boxes.pop(t, None)
                self._age.pop(t, None)
        return ids

    @property
    def active_ids(self) -> set[int]:
        return set(self._boxes)


class TemporalSmoother:
    """Temporal filter over per-person predictions.

    Args:
        window: frames averaged by the moving-average modes. Forced odd in
            ``"centered"``. Ignored by ``"1euro"``.
        mode: ``"centered"`` (moving average, no phase lag, ``(window-1)//2``
            frames of latency — the filter NLF measured, so the one to use for
            benchmark numbers), ``"causal"`` (trailing average, zero latency,
            real phase lag) or ``"1euro"`` (adaptive, zero latency, no phase
            lag during fast motion — the one to use for demos and live camera).
        beta_scale: multiplies every per-field ``beta`` in ``"1euro"`` mode.
            This is *the* jitter-versus-lag knob: raise it if motion still lags,
            lower it if the output looks noisy while you hold still.
        mincutoff_scale: multiplies every per-field ``mincutoff``. Governs how
            hard the signal is smoothed at rest.
        keypoint_fn: optional ``f(mhr_params, shape_params) -> (70, 3)``. When
            given, ``joints_3d_local`` is *re-derived* from the filtered pose
            parameters instead of being filtered on its own. This is what keeps
            the keypoints welded to the mesh — see :meth:`_finalize`.
        iou_threshold: minimum IoU to associate a detection with a track.
        max_age: frames a track survives without a detection.

    Usage mirrors a delay line — in the zero-latency modes ``push`` simply
    always returns a frame and ``flush`` yields nothing::

        for persons in stream:
            out = smoother.push(persons, timestamp=t)
            if out is not None:
                render(out)
        for out in smoother.flush():
            render(out)
    """

    def __init__(
        self,
        window: int = 5,
        mode: str = "centered",
        beta_scale: float = 1.0,
        mincutoff_scale: float = 1.0,
        keypoint_fn=None,
        iou_threshold: float = 0.3,
        max_age: int = 5,
    ):
        if mode not in ("centered", "causal", "1euro"):
            raise ValueError(
                f"mode must be 'centered', 'causal' or '1euro', got {mode!r}")
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        if mode == "centered" and window % 2 == 0:
            window += 1
        self.window = window
        self.mode = mode
        self.lag = (window - 1) // 2 if mode == "centered" else 0
        self.beta_scale = float(beta_scale)
        self.mincutoff_scale = float(mincutoff_scale)
        self.keypoint_fn = keypoint_fn

        self._tracker = _Tracker(iou_threshold=iou_threshold, max_age=max_age)
        self._buf: deque[dict[int, HMRPrediction]] = deque(maxlen=window)
        # mode="1euro" only: one filter bank per track, built on first sight and
        # dropped when the tracker retires the track.
        self._banks: dict[int, dict[str, _OneEuro]] = {}
        self._clock = 0.0
        # Frames in and frames out. Every pushed frame is emitted exactly once,
        # so a caller holding its own delay line can pop one entry per returned
        # frame and stay in step with the video.
        self._n_pushed = 0
        self._n_emitted = 0

        # Jitter accounting, so the effect is measurable rather than only
        # visible. Both are mean frame-to-frame deltas over the same tracks.
        self._prev_raw: dict[int, HMRPrediction] = {}
        self._prev_out: dict[int, HMRPrediction] = {}
        self._raw_cam: list[float] = []
        self._raw_j3d: list[float] = []
        self._out_cam: list[float] = []
        self._out_j3d: list[float] = []

    # ------------------------------------------------------------------
    def push(
        self,
        persons: Sequence[HMRPrediction],
        timestamp: float | None = None,
    ) -> list[HMRPrediction] | None:
        """Feed one frame. Returns a smoothed frame, or ``None`` while filling.

        ``timestamp`` (seconds) is used only by ``mode="1euro"``, which needs a
        real time base to turn deltas into speeds — pass wall-clock for a live
        camera and ``frame_idx / fps`` for a file, so ``--frame-skip`` is
        accounted for. It defaults to a nominal 30 fps clock.
        """
        ids = self._tracker.assign(persons)
        frame = {t: p for t, p in zip(ids, persons)}
        self._accumulate(frame, self._prev_raw, self._raw_cam, self._raw_j3d)
        self._prev_raw = frame

        if timestamp is None:
            self._clock += 1.0 / 30.0
            timestamp = self._clock

        if self.mode == "1euro":
            self._n_pushed += 1
            out = self._one_euro(frame, timestamp)
            self._n_emitted += 1
            return out

        self._buf.append(frame)
        self._n_pushed += 1

        # Emit frame ``_n_emitted`` as soon as ``lag`` frames after it have
        # arrived. Near the start of the stream that window is one-sided rather
        # than centred — the usual boundary treatment for a moving average, and
        # what keeps frames-in equal to frames-out.
        if self._n_pushed - 1 < self._n_emitted + self.lag:
            return None
        return self._emit_next()

    def flush(self) -> Iterator[list[HMRPrediction]]:
        """Drain the frames still held in the delay line at end of stream."""
        while self._n_emitted < self._n_pushed:
            yield self._emit_next()

    def _one_euro(self, frame: dict[int, HMRPrediction], t: float) -> list[HMRPrediction]:
        """Filter each track's fields independently, elementwise, on its own speed."""
        alive = self._tracker.active_ids
        for dead in [k for k in self._banks if k not in alive]:
            del self._banks[dead]

        out: list[HMRPrediction] = []
        emitted: dict[int, HMRPrediction] = {}
        for track_id, pred in frame.items():
            bank = self._banks.get(track_id)
            if bank is None:
                bank = {
                    name: _OneEuro(
                        mincutoff=ONE_EURO_DEFAULTS[name][0] * self.mincutoff_scale,
                        beta=ONE_EURO_DEFAULTS[name][1] * self.beta_scale,
                    )
                    for name in SMOOTHED_FIELDS
                }
                self._banks[track_id] = bank
            fields = {
                name: bank[name](getattr(pred, name), t).astype(np.float32)
                for name in SMOOTHED_FIELDS
            }
            smoothed = self._finalize(pred, fields)
            out.append(smoothed)
            emitted[track_id] = smoothed

        self._accumulate(emitted, self._prev_out, self._out_cam, self._out_j3d)
        self._prev_out = emitted
        return out

    def _emit_next(self) -> list[HMRPrediction]:
        buf_start = self._n_pushed - len(self._buf)
        out = self._emit(self._n_emitted - buf_start)
        self._n_emitted += 1
        return out

    # ------------------------------------------------------------------
    def _finalize(self, anchor: HMRPrediction, fields: dict) -> HMRPrediction:
        """Assemble one smoothed prediction, keeping its parts self-consistent.

        Filtering ``mhr_params`` and ``joints_3d_local`` independently makes the
        green keypoints drift off the skin during fast motion, because the mesh
        is decoded from the former while the keypoints come from the latter and
        forward kinematics is *nonlinear* — ``filter(FK(p)) != FK(filter(p))``.
        Adaptive filtering makes it worse still, since each field then gets its
        own cutoff driven by its own speed in its own units.

        So when a keypoint backend is available we do not filter the keypoints
        at all: we filter the pose parameters and re-derive the keypoints from
        them, which is exactly how ``inference.py`` produces them for an
        MHR-only graph. The two agree by construction, in every mode.
        """
        if self.keypoint_fn is not None:
            fields["joints_3d_local"] = np.asarray(
                self.keypoint_fn(fields["mhr_params"], fields["shape_params"]),
                dtype=np.float32,
            ).reshape(anchor.joints_3d_local.shape)
        fields["joints_3d_cam"] = (
            fields["joints_3d_local"] + fields["cam_trans"][None, :]
        ).astype(np.float32)
        return replace(anchor, **fields)

    def _emit(self, centre: int) -> list[HMRPrediction]:
        """Average every track present in frame ``centre`` over the buffer."""
        out: list[HMRPrediction] = []
        emitted: dict[int, HMRPrediction] = {}
        for track_id, anchor in self._buf[centre].items():
            samples = [f[track_id] for f in self._buf if track_id in f]
            fields = {
                name: np.mean(
                    np.stack([np.asarray(getattr(s, name), dtype=np.float64)
                              for s in samples]),
                    axis=0,
                ).astype(np.float32)
                for name in SMOOTHED_FIELDS
            }
            smoothed = self._finalize(anchor, fields)
            out.append(smoothed)
            emitted[track_id] = smoothed

        self._accumulate(emitted, self._prev_out, self._out_cam, self._out_j3d)
        self._prev_out = emitted
        return out

    @staticmethod
    def _accumulate(cur, prev, cam_acc: list[float], j3d_acc: list[float]) -> None:
        """Record mean frame-to-frame movement for tracks seen in both frames."""
        for track_id, p in cur.items():
            q = prev.get(track_id)
            if q is None:
                continue
            cam_acc.append(float(np.linalg.norm(p.cam_trans - q.cam_trans)))
            j3d_acc.append(
                float(np.linalg.norm(p.joints_3d_local - q.joints_3d_local, axis=-1).mean())
            )

    # ------------------------------------------------------------------
    def jitter_report(self) -> str | None:
        """One-line before/after summary, or ``None`` if nothing was measured.

        These are mean per-frame *movement* magnitudes, not errors: real motion
        contributes to both columns. The filter is doing its job when the
        numbers drop without the pose visibly lagging the video.
        """
        if not self._raw_cam or not self._out_cam:
            return None
        rc, oc = float(np.mean(self._raw_cam)), float(np.mean(self._out_cam))
        rj, oj = float(np.mean(self._raw_j3d)), float(np.mean(self._out_j3d))
        how = (f"1euro, beta x{self.beta_scale:g}" if self.mode == "1euro"
               else f"window={self.window}, {self.mode}")
        return (
            f"smoothing ({how}) — mean frame-to-frame delta\n"
            f"  cam_trans        {rc * 1000:7.2f} -> {oc * 1000:7.2f} mm  "
            f"({100.0 * (1.0 - oc / rc) if rc > 0 else 0.0:+.0f}%)\n"
            f"  joints_3d_local  {rj * 1000:7.2f} -> {oj * 1000:7.2f} mm  "
            f"({100.0 * (1.0 - oj / rj) if rj > 0 else 0.0:+.0f}%)"
        )
