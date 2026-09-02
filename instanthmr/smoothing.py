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
  tail. Use ``mode="causal"`` for live sources, which trades phase lag for
  zero latency.

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
#: consistent with each other.
SMOOTHED_FIELDS = (
    "joints_3d_local",
    "cam_trans",
    "joints_2d",
    "mhr_params",
    "shape_params",
)


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


class TemporalSmoother:
    """Moving-average filter over per-person predictions.

    Args:
        window: number of frames averaged. Forced odd in ``"centered"`` mode.
        mode: ``"centered"`` (no phase lag, ``(window-1)//2`` frames of
            latency) or ``"causal"`` (trailing average, zero latency).
        iou_threshold: minimum IoU to associate a detection with a track.
        max_age: frames a track survives without a detection.

    Usage mirrors a delay line::

        for persons in stream:
            out = smoother.push(persons)
            if out is not None:
                render(out)          # a frame (window-1)//2 back
        for out in smoother.flush():
            render(out)              # the tail
    """

    def __init__(
        self,
        window: int = 5,
        mode: str = "centered",
        iou_threshold: float = 0.3,
        max_age: int = 5,
    ):
        if mode not in ("centered", "causal"):
            raise ValueError(f"mode must be 'centered' or 'causal', got {mode!r}")
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        if mode == "centered" and window % 2 == 0:
            window += 1
        self.window = window
        self.mode = mode
        self.lag = (window - 1) // 2 if mode == "centered" else 0

        self._tracker = _Tracker(iou_threshold=iou_threshold, max_age=max_age)
        self._buf: deque[dict[int, HMRPrediction]] = deque(maxlen=window)
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
    def push(self, persons: Sequence[HMRPrediction]) -> list[HMRPrediction] | None:
        """Feed one frame. Returns a smoothed frame, or ``None`` while filling."""
        ids = self._tracker.assign(persons)
        frame = {t: p for t, p in zip(ids, persons)}
        self._accumulate(frame, self._prev_raw, self._raw_cam, self._raw_j3d)
        self._prev_raw = frame

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

    def _emit_next(self) -> list[HMRPrediction]:
        buf_start = self._n_pushed - len(self._buf)
        out = self._emit(self._n_emitted - buf_start)
        self._n_emitted += 1
        return out

    # ------------------------------------------------------------------
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
            fields["joints_3d_cam"] = (
                fields["joints_3d_local"] + fields["cam_trans"][None, :]
            ).astype(np.float32)
            smoothed = replace(anchor, **fields)
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
        return (
            f"smoothing (window={self.window}, {self.mode}) — mean frame-to-frame delta\n"
            f"  cam_trans        {rc * 1000:7.2f} -> {oc * 1000:7.2f} mm  "
            f"({100.0 * (1.0 - oc / rc) if rc > 0 else 0.0:+.0f}%)\n"
            f"  joints_3d_local  {rj * 1000:7.2f} -> {oj * 1000:7.2f} mm  "
            f"({100.0 * (1.0 - oj / rj) if rj > 0 else 0.0:+.0f}%)"
        )
