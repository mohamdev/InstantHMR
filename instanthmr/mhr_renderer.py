"""MHR (Momentum Human Rig) mesh decoder for InstantHMR.

Takes the 204-dim pose parameters and 45-dim shape parameters produced by
InstantHMR and decodes them into full body mesh vertices by running a forward
pass through Meta's MHR body model.

Quick setup (requires Python >= 3.12)::

    pip install -r requirements-mhr.txt
    curl -OL https://github.com/facebookresearch/MHR/releases/download/v1.0.0/assets.zip
    unzip assets.zip -d models/mhr_assets
    python demo.py --image photo.jpg --mhr-assets models/mhr_assets

WARNING — wrong pymomentum package on PyPI
------------------------------------------
``pip install pymomentum`` installs an **unrelated** legacy SMS library
(pyMomentum v0.1.x by MomentumAS).  The correct Meta package is::

    pip install pymomentum-gpu   # NVIDIA GPU
    pip install pymomentum-cpu   # macOS / CPU-only

See README.md §'MHR body mesh' for the full installation walkthrough.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class MHRRenderer:
    """Loads and caches the MHR body model, runs per-person forward passes.

    Args:
        assets_folder: path to the unpacked MHR assets folder (the directory
            that contains the blend-shape files, template mesh, etc.).
        device: ``'cuda'``, ``'cpu'``, or a ``torch.device``.
        lod: level-of-detail mesh resolution:
            0 = 73 639 verts (highest), 3 = 4 899, 6 = 595 (lowest).
            LOD 3 is a good default for live demos.
    """

    def __init__(
        self,
        assets_folder: str | Path = "models/mhr_assets",
        device: str = "cuda",
        lod: int = 3,
    ) -> None:
        import torch
        from mhr.mhr import MHR

        self._torch = torch
        self._device = torch.device(device)
        self._model = MHR.from_files(
            folder=Path(assets_folder),
            device=self._device,
            lod=lod,
        )

        faces_raw = self._model.character.mesh.faces
        if isinstance(faces_raw, torch.Tensor):
            self._faces: np.ndarray = faces_raw.cpu().numpy().astype(np.int32)
        else:
            self._faces = np.asarray(faces_raw, dtype=np.int32)

    @property
    def faces(self) -> np.ndarray:
        """Triangle face indices ``(F, 3)`` int32, shared across all frames."""
        return self._faces

    def forward(
        self,
        mhr_params: np.ndarray,
        shape_params: np.ndarray,
    ) -> np.ndarray:
        """Decode one person's parameters into body-space mesh vertices.

        Args:
            mhr_params: ``(204,)`` float32 — pose parameters from InstantHMR.
            shape_params: ``(45,)`` float32 — shape parameters from InstantHMR.

        Returns:
            ``(V, 3)`` float32 vertices in **body-centred** space (metres,
            Y-down). Add the person's ``cam_trans`` to move to camera space.
        """
        torch = self._torch
        identity = torch.from_numpy(shape_params).unsqueeze(0).to(self._device)  # (1, 45)
        pose = torch.from_numpy(mhr_params).unsqueeze(0).to(self._device)        # (1, 204)
        # MHR also accepts facial expression coefficients (72-D); InstantHMR
        # does not predict them, so we pass zeros (neutral expression).
        expr = torch.zeros(1, 72, dtype=torch.float32, device=self._device)

        with torch.no_grad():
            verts, _ = self._model(identity, pose, expr)

        v = verts[0].cpu().numpy().astype(np.float32)  # (V, 3)

        # --- Coordinate-space alignment -----------------------------------
        # MHR body template (FBX) is Y-up right-handed and in centimetres.
        # InstantHMR / SAM3D uses Y-down right-handed and metres.
        #
        # 1. cm → m
        v /= 100.0
        # 2. Y-up → Y-down: negate Y.
        #    Negate Z as well to preserve right-handedness after the Y flip.
        v[:, 1] *= -1.0
        v[:, 2] *= -1.0

        return v
