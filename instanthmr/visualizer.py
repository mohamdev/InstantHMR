"""Rerun-based 3D visualization for the InstantHMR demo.

Logs, per frame:
  - ``camera/image``        : annotated RGB frame (2D skeleton + bbox + a
                              text panel showing detector / HMR / total ms).
  - ``camera``              : pinhole intrinsics so the image becomes a
                              proper frustum in the 3D scene.
  - ``world/persons/...``   : 3D joints + skeleton lines per person.
  - ``timing/detector_ms``  : scalar plot of RF-DETR latency.
  - ``timing/hmr_ms``       : scalar plot of InstantHMR latency.
  - ``timing/total_ms``     : scalar plot of full pipeline latency.
  - ``timing/fps``          : scalar plot of effective frames-per-second.

``rr.init(..., spawn=True)`` is called automatically so the viewer opens
on construction (pass ``spawn_viewer=False`` to suppress).
"""

from __future__ import annotations

from typing import Optional, Sequence

import cv2
import numpy as np

from .inference import HMRPrediction
from .skeleton import edges_for


class RerunVisualizer:
    def __init__(
        self,
        application_id: str = "instanthmr_demo",
        spawn_viewer: bool = True,
        save_path: str | None = None,
    ):
        import rerun as rr

        self._rr = rr
        rr.init(application_id, spawn=spawn_viewer)

        # SAM3D / MHR convention: right-handed, Y-down camera frame.
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

        if save_path:
            rr.save(save_path)

        self._prev_num_persons = 0
        self._blueprint_sent = False

    # ------------------------------------------------------------------
    # Per-frame logging
    # ------------------------------------------------------------------

    def log_frame(
        self,
        image_rgb: np.ndarray,
        persons: Sequence[HMRPrediction],
        frame_idx: int,
        timestamp: float | None = None,
        detector_ms: Optional[float] = None,
        hmr_ms: Optional[float] = None,
        total_ms: Optional[float] = None,
    ) -> None:
        """Log one frame of the pipeline to Rerun.

        Args:
            image_rgb: full-frame RGB image, ``(H, W, 3)`` uint8.
            persons: per-person HMR predictions for this frame.
            frame_idx: monotonic frame counter (sets the Rerun timeline).
            timestamp: optional wall-clock timestamp (seconds).
            detector_ms: RF-DETR latency for this frame (ms).
            hmr_ms: InstantHMR latency for this frame, summed across
                persons (ms).
            total_ms: full pipeline latency for this frame (ms).
        """
        rr = self._rr
        rr.set_time("frame", sequence=frame_idx)
        if timestamp is not None:
            rr.set_time("timestamp", duration=timestamp)

        # Per-stage scalar plots
        if detector_ms is not None:
            rr.log("timing/detector_ms", rr.Scalars(float(detector_ms)))
        if hmr_ms is not None:
            rr.log("timing/hmr_ms", rr.Scalars(float(hmr_ms)))
        if total_ms is not None:
            rr.log("timing/total_ms", rr.Scalars(float(total_ms)))
            if total_ms > 0:
                rr.log("timing/fps", rr.Scalars(1000.0 / float(total_ms)))

        # Clear stale person entities from previous frames
        n = len(persons)
        for stale in range(n, self._prev_num_persons):
            rr.log(f"world/persons/person_{stale}", rr.Clear(recursive=True))
        self._prev_num_persons = n

        h, w = image_rgb.shape[:2]

        # Always log the image — even when no person was detected — so the
        # viewer keeps showing the camera feed and the timing HUD.
        if not persons:
            blank = self._draw_overlay(image_rgb, [], detector_ms, hmr_ms, total_ms)
            rr.log("camera/image", rr.Image(blank))
            return

        for idx, person in enumerate(persons):
            self._log_person(idx, person)

        annotated = self._draw_overlay(
            image_rgb, persons, detector_ms, hmr_ms, total_ms,
        )
        rr.log("camera/image", rr.Image(annotated))

        # Pinhole camera in the 3D scene — full-frame focal so the image
        # plane sits in front of the joints, not on top of them.
        first = persons[0]
        fx, fy = float(first.focal_length[0]), float(first.focal_length[1])
        cx, cy = w / 2.0, h / 2.0
        depth = float(first.joints_3d_cam[:, 2].mean())
        plane_dist = max(depth * 0.25, 0.2)

        rr.log(
            "camera",
            rr.Pinhole(
                width=w,
                height=h,
                focal_length=[fx, fy],
                principal_point=[cx, cy],
                image_plane_distance=plane_dist,
            ),
        )

        if not self._blueprint_sent:
            self._send_blueprint(first)
            self._blueprint_sent = True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _log_person(self, idx: int, person: HMRPrediction) -> None:
        rr = self._rr
        path = f"world/persons/person_{idx}"

        joints = person.joints_3d_cam
        rr.log(
            f"{path}/joints",
            rr.Points3D(positions=joints, radii=0.012, colors=[0, 230, 0]),
        )

        edges = edges_for(joints.shape[0])
        if edges:
            lines = [
                [joints[i].tolist(), joints[j].tolist()]
                for i, j in edges
            ]
            rr.log(
                f"{path}/skeleton",
                rr.LineStrips3D(lines, colors=[255, 230, 0], radii=0.004),
            )

    @staticmethod
    def _draw_overlay(
        image_rgb: np.ndarray,
        persons: Sequence[HMRPrediction],
        detector_ms: Optional[float],
        hmr_ms: Optional[float],
        total_ms: Optional[float],
    ) -> np.ndarray:
        """Draw per-person bbox + 2D skeleton + a timing HUD onto the frame."""
        out = image_rgb.copy()
        h, w = out.shape[:2]

        for person in persons:
            x1, y1, x2, y2 = person.bbox.astype(int)
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 255), 2)
            label = f"{person.confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 200, 255), -1)
            cv2.putText(
                out, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
            )

            j2d = person.joints_2d
            valid = (
                (j2d[:, 0] >= 0) & (j2d[:, 0] < w)
                & (j2d[:, 1] >= 0) & (j2d[:, 1] < h)
            )

            for i, j in edges_for(j2d.shape[0]):
                if valid[i] and valid[j]:
                    pt1 = tuple(j2d[i].astype(int))
                    pt2 = tuple(j2d[j].astype(int))
                    cv2.line(out, pt1, pt2, (255, 230, 0), 2)

            for k, (x, y) in enumerate(j2d):
                if valid[k]:
                    cv2.circle(out, (int(x), int(y)), 3, (0, 230, 0), -1)

        # ---- Timing HUD (top-left, semi-transparent black panel) ----
        lines: list[str] = []
        if detector_ms is not None:
            lines.append(f"RF-DETR : {detector_ms:6.1f} ms")
        if hmr_ms is not None:
            lines.append(f"InstantHMR: {hmr_ms:6.1f} ms")
        if total_ms is not None:
            fps = 1000.0 / total_ms if total_ms > 0 else 0.0
            lines.append(f"Total   : {total_ms:6.1f} ms ({fps:5.1f} fps)")
        if lines:
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = max(0.5, min(w, h) / 1500.0)
            thick = max(1, int(round(scale * 1.5)))
            sizes = [
                cv2.getTextSize(s, font, scale, thick)[0] for s in lines
            ]
            line_h = max(s[1] for s in sizes) + 8
            box_w = max(s[0] for s in sizes) + 16
            box_h = line_h * len(lines) + 8

            overlay = out.copy()
            cv2.rectangle(overlay, (8, 8), (8 + box_w, 8 + box_h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.55, out, 0.45, 0, out)

            y = 8 + line_h
            for s in lines:
                cv2.putText(
                    out, s, (16, y),
                    font, scale, (255, 255, 255), thick, cv2.LINE_AA,
                )
                y += line_h

        return out

    def _send_blueprint(self, first: HMRPrediction) -> None:
        """Send a 3-panel blueprint: camera image | 3D scene | timings."""
        import rerun.blueprint as rrb
        rr = self._rr

        joints = first.joints_3d_cam
        center = joints.mean(axis=0)
        extent = float(np.abs(joints - center).max())
        cam_distance = max(extent * 4.0, 1.5)
        # Y-down convention: place the orbit camera "in front" along -Z.
        eye_pos = [
            float(center[0]),
            float(center[1]),
            float(center[2] - cam_distance),
        ]

        blueprint = rrb.Blueprint(
            rrb.Horizontal(
                rrb.Vertical(
                    rrb.Spatial2DView(origin="camera/image", name="Camera"),
                    rrb.TimeSeriesView(
                        origin="timing",
                        name="Latency (ms)",
                    ),
                    row_shares=[3, 1],
                ),
                rrb.Spatial3DView(
                    origin="/",
                    name="3D Scene",
                    eye_controls=rrb.EyeControls3D(
                        position=eye_pos,
                        look_target=[float(center[0]), float(center[1]), float(center[2])],
                        eye_up=[0.0, -1.0, 0.0],
                        kind=rrb.Eye3DKind.Orbital,
                        tracking_entity="world/persons/person_0",
                    ),
                    background=rrb.Background(color=[25, 25, 25]),
                ),
                column_shares=[2, 3],
            ),
        )
        rr.send_blueprint(blueprint)
