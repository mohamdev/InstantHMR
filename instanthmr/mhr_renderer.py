"""MHR (Momentum Human Rig) decoder for InstantHMR.

Takes the 204-dim pose parameters and 45-dim shape parameters produced by
InstantHMR and runs a forward pass through Meta's MHR body model to obtain

  * the full body **mesh** (skinned vertices), and
  * the 70 native **keypoints**, regressed from the 127 skeleton joints.

The keypoints matter because ``train_distill_mhr_only.py`` deletes the
student's 3D coordinate head: its ONNX graph has four outputs instead of five
and the 3D pose must be derived here, exactly the way the teacher derives it
from the mesh.  The (70, 127) affine map lives in
``instanthmr_distill_train/assets/mhr_j127_to_kp70.npy`` and was fitted against
the teacher's own labels (held-out residual: 1.36 mm mean).

Two interchangeable backends
----------------------------
``MHRScriptRenderer`` (recommended)
    Loads the self-contained TorchScript rig at ``checkpoints/mhr_model.pt`` —
    the same artefact the distillation used.  Needs nothing but PyTorch, so it
    is immune to the ``pymomentum`` / ``torch`` ABI pairing below.

``MHRRenderer``
    Loads the unpacked MHR asset folder through the ``mhr`` pip package.  This
    lets you pick a mesh LOD, but ``pymomentum-gpu`` is a torch C++ extension
    and **must** match the installed torch (0.1.113 wants ``torch>=2.8,<2.9``);
    a mismatch segfaults the process rather than raising.

Use :func:`build_mhr` to get whichever one the caller configured.

Quick setup for the asset backend (requires Python >= 3.12)::

    pip install -r requirements-mhr.txt
    curl -OL https://github.com/facebookresearch/MHR/releases/download/v1.0.0/assets.zip
    unzip assets.zip -d models/mhr_assets
    python demo.py --image photo.jpg --mhr-assets models/mhr_assets/assets

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
from typing import Optional

import numpy as np


_REPO_ROOT = Path(__file__).resolve().parent.parent

#: (70, 127) affine map from MHR skeleton joints to the 70 native keypoints.
DEFAULT_KP_REGRESSOR = _REPO_ROOT / "instanthmr_distill_train/assets/mhr_j127_to_kp70.npy"
#: TorchScript rig shipped with the training code.
DEFAULT_MHR_SCRIPT = _REPO_ROOT / "checkpoints/mhr_model.pt"

NUM_MODEL_PARAMS = 204      # 136 pose + 68 scale
NUM_IDENTITY = 45           # identity blend-shape coefficients
NUM_EXPRESSION = 72         # facial expression coefficients (InstantHMR predicts none)
NUM_SKELETON_JOINTS = 127
NUM_KEYPOINTS = 70


class _MHRBase:
    """Parameter marshalling, coordinate conversion, rig evaluation and keypoints.

    Both backends expose the same three objects, so everything below the loader
    is shared. A subclass' ``__init__`` must set:

      * ``self._model``      — callable ``(identity, pose, expr) -> (verts, skel)``
      * ``self._character``  — exposes ``model_parameters_to_joint_parameters``
        and ``joint_parameters_to_skeleton_state``
      * ``self._n_params``   — width its parameter transform expects, i.e. 204
        model parameters plus that rig's blend-shape slots

    and it must implement the ``faces`` property.
    """

    def __init__(self, device: str = "cuda", kp_regressor: str | Path | None = None):
        import torch

        self._torch = torch
        self._device = self._resolve_device(torch, device)
        self._kp_regressor = None

        path = Path(kp_regressor) if kp_regressor is not None else DEFAULT_KP_REGRESSOR
        if path.exists():
            W = np.load(path).astype(np.float32)
            if W.shape != (NUM_KEYPOINTS, NUM_SKELETON_JOINTS):
                raise ValueError(
                    f"expected a ({NUM_KEYPOINTS}, {NUM_SKELETON_JOINTS}) keypoint "
                    f"regressor in {path}, got {W.shape}"
                )
            self._kp_regressor = torch.from_numpy(W).to(self._device)
        elif kp_regressor is not None:
            raise FileNotFoundError(f"keypoint regressor not found: {path}")

    @staticmethod
    def _resolve_device(torch, device):
        """Map the demo's device string onto a torch device.

        ``--device coreml`` selects an onnxruntime execution provider, not a
        torch backend, so anything torch cannot run on falls back to CPU.
        """
        name = str(device).lower()
        if "cuda" in name and torch.cuda.is_available():
            return torch.device(device)
        if "cuda" in name:
            return torch.device("cpu")
        try:
            return torch.device(device)
        except (RuntimeError, TypeError):
            return torch.device("cpu")

    # ------------------------------------------------------------------
    # Capabilities
    # ------------------------------------------------------------------

    @property
    def has_keypoints(self) -> bool:
        """True when the 70-keypoint regressor was found and loaded."""
        return self._kp_regressor is not None

    @property
    def faces(self) -> np.ndarray:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Coordinate conventions
    # ------------------------------------------------------------------

    @staticmethod
    def _to_vision(xyz):
        """Raw MHR (centimetres, Y-up / Z-back) -> InstantHMR (metres, Y-down / Z-forward).

        Negating both Y and Z keeps the frame right-handed after the Y flip.
        This is the exact transform ``train_distill_mhr_only.MHRForwardPass``
        applies before the keypoint regression, so student and demo agree.
        """
        out = xyz / 100.0
        out[..., 1] *= -1.0
        out[..., 2] *= -1.0
        return out

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def keypoints(
        self,
        mhr_params: np.ndarray,
        shape_params: np.ndarray,
    ) -> np.ndarray:
        """Regress the 70 native keypoints from the MHR skeleton.

        Skeleton branch only — no blend shapes, no skinning — so this is ~3x
        cheaper than a full mesh forward (≈0.9 ms vs ≈2.6 ms at batch 2 on an
        RTX 4070).

        Args:
            mhr_params: ``(204,)`` or ``(B, 204)`` float32 pose parameters.
            shape_params: ``(45,)`` or ``(B, 45)`` — accepted for symmetry with
                :meth:`forward`; identity coefficients do not move the skeleton
                (they only displace the rest-pose vertices), which is why the
                distillation passes zeros here too.

        Returns:
            ``(70, 3)`` or ``(B, 70, 3)`` float32 keypoints in the **rig-local**
            frame (metres, Y-down) — the frame ``cam_trans`` is relative to.
            Add ``cam_trans`` for camera space.
        """
        if self._kp_regressor is None:
            raise RuntimeError(
                f"no MHR keypoint regressor loaded (looked for {DEFAULT_KP_REGRESSOR}).\n"
                "It ships with the training code; regenerate it with\n"
                "  python instanthmr_distill_train/train_distill_mhr_only.py --fit-regressor"
            )
        torch = self._torch
        pose, squeeze = self._as_batch(mhr_params, NUM_MODEL_PARAMS)

        with torch.no_grad():
            skel = self._skeleton_state(pose)
            kps = self._regress(skel)

        out = kps.cpu().numpy().astype(np.float32)
        return out[0] if squeeze else out

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
            ``(V, 3)`` float32 vertices in the **rig-local** frame (metres,
            Y-down). Add the person's ``cam_trans`` to move to camera space.
        """
        verts, _ = self.forward_full(mhr_params, shape_params)
        return verts

    def forward_full(
        self,
        mhr_params: np.ndarray,
        shape_params: np.ndarray,
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Mesh vertices **and** the 70 keypoints from a single MHR forward pass.

        Returns ``(verts, keypoints)`` with the same leading shape convention as
        :meth:`forward` / :meth:`keypoints`. ``keypoints`` is ``None`` when no
        regressor is available.
        """
        torch = self._torch
        pose, squeeze = self._as_batch(mhr_params, NUM_MODEL_PARAMS)
        identity, _ = self._as_batch(shape_params, NUM_IDENTITY)
        if identity.shape[0] != pose.shape[0]:
            identity = identity.expand(pose.shape[0], -1)

        with torch.no_grad():
            verts, skel = self._skin(identity, pose)
            verts = self._to_vision(verts.float())
            kps = self._regress(skel) if self._kp_regressor is not None else None

        v = verts.cpu().numpy().astype(np.float32)
        k = kps.cpu().numpy().astype(np.float32) if kps is not None else None
        if squeeze:
            return v[0], (None if k is None else k[0])
        return v, k

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _as_batch(self, arr: np.ndarray, dim: int):
        """numpy ``(dim,)`` / ``(B, dim)`` -> torch ``(B, dim)`` on the rig device."""
        torch = self._torch
        t = torch.as_tensor(np.ascontiguousarray(arr), dtype=torch.float32)
        squeeze = t.dim() == 1
        if squeeze:
            t = t.unsqueeze(0)
        if t.shape[-1] != dim:
            raise ValueError(f"expected trailing dimension {dim}, got {tuple(t.shape)}")
        return t.to(self._device), squeeze

    def _regress(self, skel):
        """``(B, 127, 8)`` skeleton state -> ``(B, 70, 3)`` keypoints (vision frame)."""
        joints = self._to_vision(skel[..., :3].float())
        return self._torch.einsum("kj,bjc->bkc", self._kp_regressor, joints)

    def _pad(self, pose, total: int):
        """Right-pad the 204 model parameters with zero blend-shape coefficients."""
        torch = self._torch
        extra = total - pose.shape[-1]
        if extra <= 0:
            return pose
        return torch.cat(
            [pose, torch.zeros(pose.shape[0], extra, dtype=pose.dtype, device=pose.device)],
            dim=1,
        )

    def _skeleton_state(self, pose):
        """``(B, 204)`` model parameters -> ``(B, 127, 8)`` raw skeleton state.

        Skips blend shapes and skinning entirely: only the two rigid-transform
        stages the keypoint regression needs.
        """
        ch = self._character
        joint_params = ch.model_parameters_to_joint_parameters(self._pad(pose, self._n_params))
        return ch.joint_parameters_to_skeleton_state(joint_params)

    def _skin(self, identity, pose):
        """Full rig evaluation -> ``((B, V, 3) verts, (B, 127, 8) skeleton state)``.

        InstantHMR predicts no facial expression, so those coefficients are zero.
        """
        torch = self._torch
        expr = torch.zeros(
            pose.shape[0], NUM_EXPRESSION, dtype=pose.dtype, device=pose.device,
        )
        verts, skel = self._model(identity, pose, expr)
        return verts, skel


class MHRScriptRenderer(_MHRBase):
    """MHR rig loaded from the TorchScript artefact used by the distillation.

    Args:
        script_path: path to ``mhr_model.pt`` (default: ``checkpoints/mhr_model.pt``).
        device: ``'cuda'``, ``'cpu'``, or a ``torch.device``.
        kp_regressor: optional override for the (70, 127) keypoint map.

    The mesh LOD is baked into the artefact (18 439 vertices / 36 874 faces);
    use :class:`MHRRenderer` if you need to choose one.
    """

    def __init__(
        self,
        script_path: str | Path = DEFAULT_MHR_SCRIPT,
        device: str = "cuda",
        kp_regressor: str | Path | None = None,
    ) -> None:
        super().__init__(device=device, kp_regressor=kp_regressor)
        torch = self._torch

        script_path = Path(script_path)
        if not script_path.exists():
            raise FileNotFoundError(f"MHR TorchScript rig not found: {script_path}")

        self._model = torch.jit.load(str(script_path), map_location=self._device).eval()
        for p in self._model.parameters():
            p.requires_grad_(False)

        self._character = self._model.character_torch
        # 204 model parameters + 45 identity blend-shape slots in this rig's
        # parameter transform; read it off rather than assuming.
        self._n_params = len(self._character.parameter_transform.parameter_names)

        faces = self._character.mesh.faces
        self._faces = faces.cpu().numpy().astype(np.int32)

    @property
    def faces(self) -> np.ndarray:
        """Triangle face indices ``(F, 3)`` int32, shared across all frames."""
        return self._faces



class MHRRenderer(_MHRBase):
    """MHR rig loaded from the unpacked asset folder via the ``mhr`` pip package.

    Args:
        assets_folder: path to the folder that holds ``lod*.fbx`` and
            ``compact_v6_1.model`` (i.e. the ``assets/`` directory inside the
            unpacked ``assets.zip``).
        device: ``'cuda'``, ``'cpu'``, or a ``torch.device``.
        lod: level-of-detail mesh resolution:
            0 = 73 639 verts (highest), 3 = 4 899, 6 = 595 (lowest).
            LOD 3 is a good default for live demos.
        kp_regressor: optional override for the (70, 127) keypoint map.

    Note:
        ``pymomentum-gpu`` is a torch C++ extension. If its build does not match
        the installed torch (0.1.113 requires ``torch>=2.8,<2.9``) the process
        segfaults inside the native library instead of raising — prefer
        :class:`MHRScriptRenderer` when in doubt.
    """

    def __init__(
        self,
        assets_folder: str | Path = "models/mhr_assets",
        device: str = "cuda",
        lod: int = 3,
        kp_regressor: str | Path | None = None,
    ) -> None:
        super().__init__(device=device, kp_regressor=kp_regressor)
        from mhr.mhr import MHR

        folder = Path(assets_folder)
        # assets.zip unpacks to <dir>/assets/ — accept either level.
        if not (folder / f"lod{lod}.fbx").exists() and (folder / "assets" / f"lod{lod}.fbx").exists():
            folder = folder / "assets"

        self._model = MHR.from_files(folder=folder, device=self._device, lod=lod)
        self._character = self._model.character_torch
        self._n_params = self._model.character.parameter_transform.size

        faces_raw = self._model.character.mesh.faces
        if hasattr(faces_raw, "cpu"):
            self._faces: np.ndarray = faces_raw.cpu().numpy().astype(np.int32)
        else:
            self._faces = np.asarray(faces_raw, dtype=np.int32)

    @property
    def faces(self) -> np.ndarray:
        """Triangle face indices ``(F, 3)`` int32, shared across all frames."""
        return self._faces



def build_mhr(
    script_path: str | Path | None = None,
    assets_folder: str | Path | None = None,
    device: str = "cuda",
    lod: int = 3,
    kp_regressor: str | Path | None = None,
) -> _MHRBase | None:
    """Build whichever MHR backend the caller configured.

    ``script_path`` wins when both are given. Returns ``None`` when neither is
    set, so callers can treat "no MHR" as a plain absence.
    """
    if script_path is not None:
        return MHRScriptRenderer(script_path, device=device, kp_regressor=kp_regressor)
    if assets_folder is not None:
        return MHRRenderer(assets_folder, device=device, lod=lod, kp_regressor=kp_regressor)
    return None
