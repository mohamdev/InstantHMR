#!/usr/bin/env python3
# ============================================================
# OPTIMIZED full-MHR distillation: train_distill.py + geometric
# augmentation + SimCC soft-argmax 2D head + Soft Bone-Length Loss.
# ------------------------------------------------------------
# Keeps the ENTIRE MHR pipeline (params / shape / cam heads, MHR
# forward pass, structural / feet / reprojection losses) — the
# auxiliary supervision that made train_distill.py generalize
# better than the pose3d variants — and adds:
#
#  (1) GEOMETRIC AUG (applied with prob geom_p; else identity)
#  (2) SimCC soft-argmax 2D head: per-joint 1D bin logits over
#      [-kp2d_range, +kp2d_range].
#  (3) SOFT BONE-LENGTH CONSISTENCY: Enforces kinematic rigidity
#      on the 3D coordinate regression head without restricting 
#      rotational degrees of freedom for OOD poses.
#
# Sanity:  python3 train_distill_optimized.py --self-test    --data_root ../data/sam3d_gt_mpii
#          python3 train_distill_optimized.py --overfit-test --data_root ../data/sam3d_gt_mpii
# Full:    python3 -u train_distill_optimized.py --data_root /datasets/instanthmr_data --num_workers 8
# ============================================================
import os

# Must be set before the first CUDA allocation (read lazily by the allocator).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import math
import random
import gc
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

import torchvision.transforms as transforms
import torchvision.transforms.functional as F_t
from torchvision.transforms import InterpolationMode

import timm
from tqdm import tqdm

# --- Notebook Cell 0 setup (headless-safe) ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
import onnxruntime as ort

print(ort.get_available_providers())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = "/pfcalcul/datasets/instanthmr_data/"

# MHR70 joint ordering
JOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
    "left_big_toe_tip", "left_small_toe_tip", "left_heel",
    "right_big_toe_tip", "right_small_toe_tip", "right_heel",
    "right_thumb_tip", "right_thumb_first_joint", "right_thumb_second_joint", "right_thumb_third_joint",
    "right_index_tip", "right_index_first_joint", "right_index_second_joint", "right_index_third_joint",
    "right_middle_tip", "right_middle_first_joint", "right_middle_second_joint", "right_middle_third_joint",
    "right_ring_tip", "right_ring_first_joint", "right_ring_second_joint", "right_ring_third_joint",
    "right_pinky_tip", "right_pinky_first_joint", "right_pinky_second_joint", "right_pinky_third_joint",
    "right_wrist",
    "left_thumb_tip", "left_thumb_first_joint", "left_thumb_second_joint", "left_thumb_third_joint",
    "left_index_tip", "left_index_first_joint", "left_index_second_joint", "left_index_third_joint",
    "left_middle_tip", "left_middle_first_joint", "left_middle_second_joint", "left_middle_third_joint",
    "left_ring_tip", "left_ring_first_joint", "left_ring_second_joint", "left_ring_third_joint",
    "left_pinky_tip", "left_pinky_first_joint", "left_pinky_second_joint", "left_pinky_third_joint",
    "left_wrist",
    "left_olecranon", "right_olecranon", "left_cubital_fossa", "right_cubital_fossa",
    "left_acromion", "right_acromion", "neck",
]

def _build_flip_perm(names):
    idx = {n: i for i, n in enumerate(names)}
    perm = list(range(len(names)))
    for i, n in enumerate(names):
        if n.startswith("left_"):
            perm[i] = idx.get("right_" + n[5:], i)
        elif n.startswith("right_"):
            perm[i] = idx.get("left_" + n[6:], i)
    perm = np.array(perm, dtype=np.int64)
    assert (perm[perm] == np.arange(len(perm))).all(), "flip permutation is not an involution"
    return perm

FLIP_PERM = _build_flip_perm(JOINT_NAMES)

# ============================================================
# Cell 1 — Configuration
# ============================================================
@dataclass
class DistillConfig:
    """All hyperparameters in one place."""
    data_root: str = str(DEFAULT_DATA_ROOT)
    mhr_model_path: str = str(PROJECT_ROOT / "checkpoints/mhr_model.pt")  
    log_dir: str = str(PROJECT_ROOT / "runs/distill_optimized_v2")  

    image_size: int = 224
    max_images: int | None = None   
    per_dataset_caps: dict = field(default_factory=lambda: {"sam3d_gt_harmony4d": 300_000})
    val_split: float = 0.1
    num_workers: int = 4
    augment: bool = True

    geom_p: float = 0.8            
    geom_rot_deg: float = 30.0     
    geom_scale_range: float = 0.25  
    geom_trans: float = 0.08       
    geom_flip_p: float = 0.5       

    kp2d_bins: int = 96            
    kp2d_range: float = 1.5        

    backbone: str = "repvit_m2_3"
    backbone_feat_dim: int = 640    
    d_model: int = 512
    n_heads: int = 8
    n_decoder_layers: int = 4
    dropout: float = 0.1

    pose_dim: int = 136
    scale_dim: int = 68
    shape_dim: int = 45
    cam_dim: int = 3
    cliff_dim: int = 3
    num_joints: int = 70            

    @property
    def model_params_dim(self) -> int:
        return self.pose_dim + self.scale_dim 

    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 400
    warmup_epochs: int = 3
    grad_clip: float = 1.0          
    anomaly_loss_threshold: float = 100.0  
    use_amp: bool = True
    ema_decay: float = 0.9998       
    early_stop_patience: int = 150  
    resume: bool = True             

    w_pose: float = 1.0
    w_scale: float = 0.1
    w_shape: float = 1.0
    w_cam: float = 0.1
    w_keypoints3d: float = 10.0     
    w_keypoints2d: float = 10.0     
    w_3d_joints: float = 1e-3
    w_structural: float = 0.01
    w_reproj_3d: float = 0.01       
    w_reproj_mhr: float = 0.01      
    w_simcc: float = 1.0            
    w_bone_length: float = 0.5      # NEW: Soft Prior for bone length consistency

    w_feet = 0.01  

    native_left_foot  = [15, 16, 17]
    native_right_foot = [18, 19, 20]

    mhr_left_foot     = [6, 7, 8]
    mhr_right_foot    = [22, 23, 24]

    native_mapping_ids = [ 1, 2, 5, 6, 9, 10, 11, 12, 13, 14, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62 ]
    mhr_mapping_ids    = [ 125, 123, 75, 39, 2, 18, 3, 19, 5, 21, 64, 63, 62, 61, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 41, 100, 99, 98, 97, 95, 94, 93, 92, 91, 90, 89, 88, 87, 86, 85, 84, 83, 82, 81, 80, 77 ]

    # NEW: Major bone connections to enforce spatial rigidity on native 3D joints
    # Format: (parent_idx, child_idx) based on JOINT_NAMES
    major_bones = [
        (5, 7),   # Left Shoulder -> Left Elbow
        (7, 62),  # Left Elbow -> Left Wrist   (62 = left_wrist, NOT 41 = right_wrist)
        (6, 8),   # Right Shoulder -> Right Elbow
        (8, 41),  # Right Elbow -> Right Wrist  (41 = right_wrist, NOT 42 = left_thumb_tip)
        (9, 11),  # Left Hip -> Left Knee
        (11, 13), # Left Knee -> Left Ankle
        (10, 12), # Right Hip -> Right Knee
        (12, 14), # Right Knee -> Right Ankle
    ]

# ============================================================
# Cell 3 — Dataset and Data Augmentation
# ============================================================
class RandomPixelate:
    def __init__(self, p=0.2, min_res=32, max_res=112):
        self.p = p
        self.min_res = min_res
        self.max_res = max_res

    def __call__(self, img):
        if random.random() < self.p:
            orig_w, orig_h = img.size
            degraded_h = random.randint(self.min_res, self.max_res)
            degraded_w = int(orig_w * (degraded_h / orig_h))
            img = F_t.resize(img, [degraded_h, degraded_w], interpolation=InterpolationMode.NEAREST)
            img = F_t.resize(img, [orig_h, orig_w], interpolation=InterpolationMode.NEAREST)
        return img

class SAM3DStudentDataset(Dataset):
    def __init__(self, data_root: str,
                 image_size: int = 224, max_images: int | None = None,
                 augment: bool = True, per_dataset_caps: dict | None = None,
                 geom_p: float = 0.8, geom_rot_deg: float = 30.0,
                 geom_scale_range: float = 0.25, geom_trans: float = 0.08,
                 geom_flip_p: float = 0.5):
        super().__init__()
        self.image_size = image_size
        self.data_root = Path(data_root)
        self.augment = augment
        self.geom_p = geom_p
        self.geom_rot_deg = geom_rot_deg
        self.geom_scale_range = geom_scale_range
        self.geom_trans = geom_trans
        self.geom_flip_p = geom_flip_p

        tfm_list = [transforms.Resize((image_size, image_size))]

        if augment:
            tfm_list.extend([
                transforms.RandomApply([transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)], p=0.5),
                RandomPixelate(p=0.2, min_res=48, max_res=112),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=5)], p=0.2),
            ])

        tfm_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        if augment:
            tfm_list.append(transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)))

        self.transform = transforms.Compose(tfm_list)

        self.pairs = []
        dataset_dirs = self._find_dataset_dirs(self.data_root)
        if not dataset_dirs:
            raise FileNotFoundError(
                f"No annotations/images sub-folders found under '{self.data_root}'."
            )

        cap_rng = random.Random(42)
        caps = per_dataset_caps or {}

        for ann_dir, img_dir in dataset_dirs:
            sub_name = ann_dir.parent.name
            sub_pairs = []
            for npz_path in sorted(ann_dir.glob("*.npz")):
                stem = npz_path.stem
                img_path = img_dir / f"{stem}.jpg"
                if not img_path.exists():
                    img_path = img_dir / f"{stem}.png"

                if img_path.exists():
                    sub_pairs.append((img_path, npz_path))

            cap = caps.get(sub_name)
            if cap is not None and len(sub_pairs) > cap:
                n_before = len(sub_pairs)
                sub_pairs.sort(key=lambda p: (str(p[1]), str(p[0])))
                sub_pairs = cap_rng.sample(sub_pairs, cap)
                print(f"  [cap] {sub_name}: sampled {cap:,} of {n_before:,} crops")
            self.pairs.extend(sub_pairs)

        self.pairs.sort(key=lambda p: (str(p[1]), str(p[0])))

        if max_images is not None:
            self.pairs = self.pairs[:max_images]

        print(f"  Found {len(dataset_dirs)} sub-folder(s) under {self.data_root}; "
              f"{len(self.pairs)} valid (image, npz) pairs (augment={augment}).")

    @staticmethod
    def _find_dataset_dirs(root: Path):
        dirs = []
        if (root / "annotations").is_dir() and (root / "images").is_dir():
            dirs.append((root / "annotations", root / "images"))

        if root.is_dir():
            for sub in sorted(root.iterdir()):
                if not sub.is_dir():
                    continue
                ann, img = sub / "annotations", sub / "images"
                if ann.is_dir() and img.is_dir():
                    dirs.append((ann, img))
        return dirs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, npz_path = self.pairs[idx]
        W = H = self.image_size

        img_pil = Image.open(img_path).convert("RGB")
        ann = np.load(npz_path)
        orig_h, orig_w = ann["orig_shape"]

        sq_bbox = ann["bbox_square"]
        sq_x1, sq_y1 = sq_bbox[0], sq_bbox[1]
        orig_crop_size = sq_bbox[2] - sq_bbox[0]
        true_scale = self.image_size / max(orig_crop_size, 1.0)

        joints_2d = ann["joints_2d"].copy().astype(np.float32)
        joints_2d[:, 0] = (joints_2d[:, 0] - sq_x1) * true_scale
        joints_2d[:, 1] = (joints_2d[:, 1] - sq_y1) * true_scale
        joints_2d[:, 0] = (joints_2d[:, 0] / self.image_size) * 2.0 - 1.0
        joints_2d[:, 1] = (joints_2d[:, 1] / self.image_size) * 2.0 - 1.0

        joints_3d = ann["joints_3d"].astype(np.float32).copy()
        cam_trans = ann["cam_trans"].astype(np.float32).copy()

        tight_bbox = ann["bbox"]
        cx = (tight_bbox[0] + tight_bbox[2]) / 2.0
        cy = (tight_bbox[1] + tight_bbox[3]) / 2.0
        b_size = max(tight_bbox[2] - tight_bbox[0], tight_bbox[3] - tight_bbox[1])
        b_scale = b_size / max(orig_w, orig_h)

        aug_flip, cos_t, sin_t = 0.0, 1.0, 0.0
        M_total = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
        aug_active = 0.0
        if self.augment and random.random() < self.geom_p:
            aug_active = 1.0
            img_np = np.asarray(img_pil.resize((W, H)))

            if random.random() < self.geom_flip_p:
                aug_flip = 1.0
                M_flip = np.array([[-1.0, 0.0, float(W)], [0.0, 1.0, 0.0]])
            else:
                M_flip = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

            angle = random.uniform(-self.geom_rot_deg, self.geom_rot_deg)
            scale = random.uniform(1.0 - self.geom_scale_range, 1.0 + self.geom_scale_range)
            dx = random.uniform(-self.geom_trans, self.geom_trans)
            dy = random.uniform(-self.geom_trans, self.geom_trans)
            M_rst = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), angle, scale)
            M_rst[0, 2] += dx * (W / 2.0)
            M_rst[1, 2] += dy * (H / 2.0)

            A2, t2 = M_rst[:, :2], M_rst[:, 2]
            A1, t1 = M_flip[:, :2], M_flip[:, 2]
            M_total = np.hstack([A2 @ A1, (A2 @ t1 + t2)[:, None]])

            img_np = cv2.warpAffine(img_np, M_total, (W, H), flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
            img_pil = Image.fromarray(img_np)

            px = (joints_2d[:, 0] + 1.0) * 0.5 * W
            py = (joints_2d[:, 1] + 1.0) * 0.5 * H
            px2 = M_total[0, 0] * px + M_total[0, 1] * py + M_total[0, 2]
            py2 = M_total[1, 0] * px + M_total[1, 1] * py + M_total[1, 2]
            joints_2d[:, 0] = (px2 / (0.5 * W) - 1.0).astype(np.float32)
            joints_2d[:, 1] = (py2 / (0.5 * H) - 1.0).astype(np.float32)

            if aug_flip > 0.5:
                joints_3d[:, 0] = -joints_3d[:, 0]
                joints_3d = joints_3d[FLIP_PERM]
                # joints_2d was already mirrored geometrically by M_total, but a
                # mirrored image LOOKS like the opposite-handed person, so the
                # left/right labels must be swapped too - exactly as for joints_3d.
                # Without this the 2D head gets contradictory left/right supervision
                # and collapses symmetric joints toward the body midline.
                joints_2d = joints_2d[FLIP_PERM]
                cam_trans[0] = -cam_trans[0]
                cx = orig_w - cx                  
            rad = math.radians(angle)
            cos_t, sin_t = math.cos(rad), math.sin(rad)
            R_v = np.array([[cos_t, sin_t], [-sin_t, cos_t]], dtype=np.float32)
            joints_3d[:, :2] = joints_3d[:, :2] @ R_v.T
            cam_trans[:2] = R_v @ cam_trans[:2]
            
            dxy = np.array([cx - orig_w / 2.0, cy - orig_h / 2.0], dtype=np.float64)
            dxy = R_v.astype(np.float64) @ dxy
            cx, cy = orig_w / 2.0 + dxy[0], orig_h / 2.0 + dxy[1]

        image = self.transform(img_pil)  

        cx_norm = 2.0 * (cx / orig_w) - 1.0
        cy_norm = 2.0 * (cy / orig_h) - 1.0
        cliff_cond = torch.tensor([cx_norm, cy_norm, b_scale], dtype=torch.float32)

        return {
            "image": image,                                                          
            "cliff_cond": cliff_cond,                                                
            "mhr_model_params": torch.from_numpy(ann["mhr_model_params"]).float(),   
            "shape_params": torch.from_numpy(ann["shape_params"]).float(),           
            "cam_trans": torch.from_numpy(cam_trans).float(),                        
            "cam_focal": torch.from_numpy(ann["cam_focal_length"]).float(),          
            "joints_2d": torch.from_numpy(joints_2d).float(),                        
            "joints_3d": torch.from_numpy(joints_3d).float(),                        
            "orig_shape": torch.tensor([orig_h, orig_w], dtype=torch.float32),       
            "bbox_square": sq_bbox,                                                  
            "aug_active": torch.tensor(aug_active, dtype=torch.float32),             
            "aug_flip": torch.tensor(aug_flip, dtype=torch.float32),                 
            "aug_cos": torch.tensor(cos_t, dtype=torch.float32),
            "aug_sin": torch.tensor(sin_t, dtype=torch.float32),
            "aug_M": torch.from_numpy(M_total.astype(np.float32).reshape(-1)),       
        }

def build_dataloaders(cfg):
    geom = dict(geom_p=cfg.geom_p, geom_rot_deg=cfg.geom_rot_deg,
                geom_scale_range=cfg.geom_scale_range,
                geom_trans=cfg.geom_trans, geom_flip_p=cfg.geom_flip_p)
    full_dataset = SAM3DStudentDataset(
        cfg.data_root,
        augment=cfg.augment,
        max_images=cfg.max_images,
        per_dataset_caps=cfg.per_dataset_caps,
        **geom,
    )

    val_dataset_clean = SAM3DStudentDataset(
        cfg.data_root,
        augment=False,
        max_images=cfg.max_images,
        per_dataset_caps=cfg.per_dataset_caps,
        **geom,
    )

    n_val = int(len(full_dataset) * cfg.val_split)
    n_train = len(full_dataset) - n_val

    generator = torch.Generator().manual_seed(42)
    train_dataset, _ = random_split(full_dataset, [n_train, n_val], generator=generator)

    generator = torch.Generator().manual_seed(42)
    _, val_dataset = random_split(val_dataset_clean, [n_train, n_val], generator=generator)

    loader_kwargs = dict(num_workers=cfg.num_workers, pin_memory=True)
    if cfg.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        **loader_kwargs,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    print(f"Dataset Loaded! Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
    return train_loader, val_loader, full_dataset

# ============================================================
# Cell 7 — Student Architecture (Mini-SAM3D with PE)
# ============================================================
def get_2d_sincos_pos_embed(embed_dim, grid_size):
    grid_h, grid_w = grid_size, grid_size
    grid_y, grid_x = torch.meshgrid(torch.arange(grid_h), torch.arange(grid_w), indexing='ij')

    omega = torch.arange(embed_dim // 4).float() / (embed_dim // 4)
    omega = 1.0 / (10000 ** omega)

    out_y = grid_y.flatten().unsqueeze(1) * omega.unsqueeze(0)
    out_x = grid_x.flatten().unsqueeze(1) * omega.unsqueeze(0)

    pe_y = torch.cat([torch.sin(out_y), torch.cos(out_y)], dim=1)
    pe_x = torch.cat([torch.sin(out_x), torch.cos(out_x)], dim=1)

    pos_embed = torch.cat([pe_y, pe_x], dim=1)  
    return pos_embed.unsqueeze(0)  

class InstantHMRStudent(nn.Module):
    def __init__(self, cfg, pretrained=True):
        super().__init__()
        self.cfg = cfg

        self.backbone = timm.create_model(cfg.backbone, pretrained=pretrained, num_classes=0)
        embed_dim = self.backbone.num_features
        self.feat_proj = nn.Linear(embed_dim, cfg.d_model)

        self.grid_size = cfg.image_size // 32
        pos_embed = get_2d_sincos_pos_embed(cfg.d_model, self.grid_size)
        self.register_buffer("mem_pos_embed", pos_embed)

        self.num_global = 1
        self.num_2d = cfg.num_joints
        self.num_3d = cfg.num_joints
        self.total_queries = self.num_global + self.num_2d + self.num_3d

        self.query_embed = nn.Parameter(torch.randn(1, self.total_queries, cfg.d_model) * 0.02)

        self.cond_proj = nn.Sequential(
            nn.Linear(cfg.cliff_dim, 128),
            nn.GELU(),
            nn.Linear(128, cfg.d_model)
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_model * 4,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=cfg.n_decoder_layers)

        self.global_out_dim = cfg.model_params_dim + cfg.shape_dim + cfg.cam_dim
        self.head_global = nn.Linear(cfg.d_model, self.global_out_dim)

        self.kp2d_bins = getattr(cfg, 'kp2d_bins', 96)
        self.kp2d_range = getattr(cfg, 'kp2d_range', 1.5)
        self.head_2d_feat = nn.Sequential(
            nn.Linear(cfg.d_model, 256),
            nn.GELU()
        )
        self.head_2d_logits = nn.Linear(256, 2 * self.kp2d_bins)
        self.register_buffer(
            "kp2d_bin_centers",
            torch.linspace(-self.kp2d_range, self.kp2d_range, self.kp2d_bins))

        self.dim_3d = 3
        self.head_3d = nn.Sequential(
            nn.Linear(cfg.d_model, 256),
            nn.GELU(),
            nn.Linear(256, self.dim_3d)
        )

        nn.init.constant_(self.head_global.bias[-1], 2.0)
        nn.init.constant_(self.head_global.bias[-2], 0.0)
        nn.init.constant_(self.head_global.bias[-3], 0.0)

        nn.init.normal_(self.head_global.weight, mean=0.0, std=1e-4)
        nn.init.constant_(self.head_global.bias[:-3], 0.0)
        nn.init.normal_(self.head_2d_logits.weight, mean=0.0, std=1e-4)
        nn.init.constant_(self.head_2d_logits.bias, 0.0)
        nn.init.normal_(self.head_3d[-1].weight, mean=0.0, std=1e-4)
        nn.init.constant_(self.head_3d[-1].bias, 0.0)

    def forward(self, images, cliff_cond):
        B = images.shape[0]
        img_feats = self.backbone.forward_features(images)

        if img_feats.dim() == 4:
            img_feats = img_feats.flatten(2).transpose(1, 2)  

        memory = self.feat_proj(img_feats)
        memory = memory + self.mem_pos_embed
        queries = self.query_embed.expand(B, -1, -1)
        cond = self.cond_proj(cliff_cond).unsqueeze(1)
        queries = queries + cond
        shared_feats = self.transformer(tgt=queries, memory=memory)

        feat_global = shared_feats[:, 0, :]
        feat_2d     = shared_feats[:, 1 : 1 + self.num_2d, :]
        feat_3d     = shared_feats[:, 1 + self.num_2d :, :]

        global_preds = self.head_global(feat_global)

        idx_mhr = self.cfg.model_params_dim
        idx_shape = idx_mhr + self.cfg.shape_dim

        pred_mhr_params = global_preds[:, :idx_mhr]
        pred_shape_params = global_preds[:, idx_mhr : idx_shape]
        pred_cam_trans = global_preds[:, idx_shape :]

        feat_2d_processed = self.head_2d_feat(feat_2d)
        logits_2d = self.head_2d_logits(feat_2d_processed)
        logits_2d = logits_2d.unflatten(-1, (2, self.kp2d_bins))
        # softmax in fp32 (autocast already promotes it; explicit for no-autocast/bf16/export paths)
        prob_2d = torch.softmax(logits_2d.float(), dim=-1)
        pred_joints_2d = (prob_2d * self.kp2d_bin_centers.float()).sum(dim=-1)

        pred_joints_3d = self.head_3d(feat_3d)

        return {
            "mhr_params": pred_mhr_params,
            "shape_params": pred_shape_params,
            "cam_trans": pred_cam_trans,
            "joints_2d": pred_joints_2d,
            "joints_2d_logits": logits_2d,
            "joints_3d": pred_joints_3d
        }

# ============================================================
# Cell 8 — The Distillation Loss Module (Foolproof Edition)
# ============================================================
class MHRForwardPass:
    def __init__(self, mhr_path, device):
        self.device = device
        self.mhr = torch.jit.load(mhr_path, map_location=device).eval()
        for p in self.mhr.parameters():
            p.requires_grad_(False)

    def get_joints(self, model_params, shape_params):
        B = model_params.shape[0]
        dummy_identity = torch.zeros(B, 45, device=model_params.device, dtype=model_params.dtype)
        concat_params = torch.cat([model_params, dummy_identity], dim=1)
        joint_params = self.mhr.character_torch.model_parameters_to_joint_parameters(concat_params.to(self.device))
        skel_state = self.mhr.character_torch.joint_parameters_to_skeleton_state(joint_params)
        return skel_state.to(model_params.device)

class DistillationLoss(nn.Module):
    def __init__(self, cfg, mhr_module):
        super().__init__()
        self.cfg = cfg
        self.mhr_module = mhr_module
        self.mse = nn.MSELoss()
        self.l1 = nn.SmoothL1Loss()
        self.strict_l1 = nn.L1Loss()
        bin_w = 2.0 * cfg.kp2d_range / (cfg.kp2d_bins - 1)
        self.simcc_sigma = 2.0 * bin_w

    @staticmethod
    def _mmean(per_sample, mask):
        return (per_sample * mask).sum() / mask.sum().clamp_min(1.0)

    def m_smooth_l1(self, pred, tgt, mask):
        e = F.smooth_l1_loss(pred, tgt, reduction='none').flatten(1).mean(1)
        return self._mmean(e, mask)

    def m_mse(self, pred, tgt, mask):
        e = F.mse_loss(pred, tgt, reduction='none').flatten(1).mean(1)
        return self._mmean(e, mask)

    def m_l1(self, pred, tgt, mask):
        e = (pred - tgt).abs().flatten(1).mean(1)
        return self._mmean(e, mask)

    def _simcc_ce(self, logits, tgt_2d):
        bins = torch.linspace(-self.cfg.kp2d_range, self.cfg.kp2d_range,
                              self.cfg.kp2d_bins, device=logits.device)
        tgt = tgt_2d.clamp(-self.cfg.kp2d_range, self.cfg.kp2d_range)
        d = bins.view(1, 1, 1, -1) - tgt.unsqueeze(-1)              
        label = torch.exp(-(d ** 2) / (2.0 * self.simcc_sigma ** 2))
        label = label / label.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        logp = F.log_softmax(logits, dim=-1)
        return -(label * logp).sum(dim=-1).mean()

    def _project_3d_to_norm_2d(self, j3d, pred_cam_trans, targets):
        j3d_cam = j3d + pred_cam_trans.unsqueeze(1)
        z_cam = torch.clamp(j3d_cam[..., 2], min=0.4)

        fx = targets["cam_focal"][:, 0:1]        
        fy = targets["cam_focal"][:, 1:2]        
        cx = targets["orig_shape"][:, 1:2] / 2.0  
        cy = targets["orig_shape"][:, 0:1] / 2.0  

        u_full = (fx * j3d_cam[..., 0] / z_cam) + cx
        v_full = (fy * j3d_cam[..., 1] / z_cam) + cy

        bbox = targets["bbox_square"]
        sq_x1 = bbox[:, 0:1]  
        sq_y1 = bbox[:, 1:2]  
        crop_w = torch.clamp(bbox[:, 2:3] - bbox[:, 0:1], min=1.0)  

        true_scale = self.cfg.image_size / crop_w
        u_crop = (u_full - sq_x1) * true_scale
        v_crop = (v_full - sq_y1) * true_scale

        M = targets["aug_M"]                                          
        u_aug = M[:, 0:1] * u_crop + M[:, 1:2] * v_crop + M[:, 2:3]
        v_aug = M[:, 3:4] * u_crop + M[:, 4:5] * v_crop + M[:, 5:6]

        u_norm = (u_aug / self.cfg.image_size) * 2.0 - 1.0
        v_norm = (v_aug / self.cfg.image_size) * 2.0 - 1.0

        return torch.stack([u_norm, v_norm], dim=-1)  

    def forward(self, preds, targets):
        losses = {}
        idx_pose = self.cfg.pose_dim
        B = preds["mhr_params"].shape[0]

        ones = torch.ones(B, device=preds["mhr_params"].device)
        aug_active = targets.get("aug_active", 1.0 - ones)   
        aug_flip = targets.get("aug_flip", 0.0 * ones)
        m_ident = 1.0 - aug_active     
        m_noflip = 1.0 - aug_flip      

        pred_pose = preds["mhr_params"][:, :idx_pose]
        pred_scale = preds["mhr_params"][:, idx_pose:]
        tgt_pose = targets["mhr_model_params"][:, :idx_pose]
        tgt_scale = targets["mhr_model_params"][:, idx_pose:]

        losses['loss_pose'] = self.m_smooth_l1(pred_pose, tgt_pose, m_ident) * self.cfg.w_pose
        losses['loss_scale'] = self.m_mse(pred_scale, tgt_scale, m_noflip) * self.cfg.w_scale
        losses['loss_shape'] = self.m_mse(preds["shape_params"], targets["shape_params"], m_noflip) * self.cfg.w_shape
        losses['loss_cam'] = self.mse(preds["cam_trans"], targets["cam_trans"]) * self.cfg.w_cam

        pred_mhr_joints = self.mhr_module.get_joints(preds["mhr_params"], preds["shape_params"])[..., :3]
        with torch.no_grad():
            tgt_mhr_joints = self.mhr_module.get_joints(targets["mhr_model_params"], targets["shape_params"])[..., :3]
            c = targets["aug_cos"].view(-1, 1) if "aug_cos" in targets else torch.ones(B, 1, device=ones.device)
            s = targets["aug_sin"].view(-1, 1) if "aug_sin" in targets else torch.zeros(B, 1, device=ones.device)
            xm, ym = tgt_mhr_joints[..., 0], tgt_mhr_joints[..., 1]
            tgt_mhr_joints = torch.stack(
                [c * xm - s * ym, s * xm + c * ym, tgt_mhr_joints[..., 2]], dim=-1)

        losses['loss_mhr_joints'] = self.m_smooth_l1(
            pred_mhr_joints, tgt_mhr_joints.detach(), m_noflip) * self.cfg.w_3d_joints

        pred_mhr_vision = pred_mhr_joints.clone() / 100.0
        pred_mhr_vision[..., 1] = -pred_mhr_vision[..., 1]
        pred_mhr_vision[..., 2] = -pred_mhr_vision[..., 2]

        if hasattr(self.cfg, 'native_mapping_ids') and hasattr(self.cfg, 'mhr_mapping_ids'):
            pred_native_anchors = preds["joints_3d"][:, self.cfg.native_mapping_ids, :3]
            pred_mhr_anchors = pred_mhr_vision[:, self.cfg.mhr_mapping_ids, :3]
            losses['loss_structural'] = self.l1(pred_native_anchors, pred_mhr_anchors) * self.cfg.w_structural

        if hasattr(self.cfg, 'native_left_foot') and hasattr(self.cfg, 'w_feet'):
            nat_lf_pts = preds["joints_3d"][:, self.cfg.native_left_foot, :3]
            nat_rf_pts = preds["joints_3d"][:, self.cfg.native_right_foot, :3]

            mhr_lf_pts = pred_mhr_vision[:, self.cfg.mhr_left_foot, :3]
            mhr_rf_pts = pred_mhr_vision[:, self.cfg.mhr_right_foot, :3]

            nat_lf_bary = nat_lf_pts.mean(dim=1)
            nat_rf_bary = nat_rf_pts.mean(dim=1)
            mhr_lf_bary = mhr_lf_pts.mean(dim=1)
            mhr_rf_bary = mhr_rf_pts.mean(dim=1)

            losses['loss_feet'] = (self.strict_l1(nat_lf_bary, mhr_lf_bary) + self.strict_l1(nat_rf_bary, mhr_rf_bary)) * self.cfg.w_feet

        # ==============================================================================
        # NEW: Soft Bone-Length Consistency Loss
        # ==============================================================================
        if hasattr(self.cfg, 'major_bones') and hasattr(self.cfg, 'w_bone_length'):
            pred_3d = preds["joints_3d"]
            tgt_3d = targets["joints_3d"]
            
            p_idx = [b[0] for b in self.cfg.major_bones]
            c_idx = [b[1] for b in self.cfg.major_bones]
            
            pred_bones = pred_3d[:, c_idx, :3] - pred_3d[:, p_idx, :3]
            tgt_bones = tgt_3d[:, c_idx, :3] - tgt_3d[:, p_idx, :3]
            
            pred_bone_lens = torch.norm(pred_bones, dim=-1)
            tgt_bone_lens = torch.norm(tgt_bones, dim=-1)
            
            losses['loss_bone_length'] = self.l1(pred_bone_lens, tgt_bone_lens) * self.cfg.w_bone_length

        pred_2d = preds["joints_2d"]
        tgt_2d = targets["joints_2d"]
        losses['loss_2d_native'] = self.l1(pred_2d[..., :2], tgt_2d) * self.cfg.w_keypoints2d
        if "joints_2d_logits" in preds:
            losses['loss_2d_simcc'] = self._simcc_ce(preds["joints_2d_logits"], tgt_2d) * self.cfg.w_simcc

        pred_3d = preds["joints_3d"]
        tgt_3d = targets["joints_3d"]
        losses['loss_3d_native'] = self.l1(pred_3d[..., :3], tgt_3d) * self.cfg.w_keypoints3d

        cc = targets["aug_cos"].view(-1, 1)
        ss = targets["aug_sin"].view(-1, 1)

        def _counter_rot_pts(pts):                       
            x, y = pts[..., 0], pts[..., 1]
            return torch.stack([cc * x - ss * y, ss * x + cc * y, pts[..., 2]], dim=-1)

        def _counter_rot_cam(cam):                       
            x, y = cam[:, 0:1], cam[:, 1:2]
            return torch.cat([cc * x - ss * y, ss * x + cc * y, cam[:, 2:3]], dim=1)

        cam_cr = _counter_rot_cam(preds["cam_trans"])

        pred_reproj_native = self._project_3d_to_norm_2d(
            _counter_rot_pts(preds["joints_3d"][..., :3]), cam_cr, targets)
        losses['loss_reproj_native'] = self.m_l1(pred_reproj_native, tgt_2d, m_noflip) * self.cfg.w_reproj_3d

        if hasattr(self.cfg, 'native_mapping_ids') and hasattr(self.cfg, 'mhr_mapping_ids'):
            pred_mhr_anchors_vision = pred_mhr_vision[:, self.cfg.mhr_mapping_ids, :3]
            pred_reproj_mhr = self._project_3d_to_norm_2d(
                _counter_rot_pts(pred_mhr_anchors_vision), cam_cr, targets)
            tgt_2d_anchors = tgt_2d[:, self.cfg.native_mapping_ids, :]
            losses['loss_reproj_mhr'] = self.m_l1(pred_reproj_mhr, tgt_2d_anchors, m_noflip) * self.cfg.w_reproj_mhr

        losses['total_loss'] = sum(losses.values())
        return losses

# ============================================================
# Cell 10 — HMR Evaluation Metrics (MPJPE & PA-MPJPE)
# ============================================================
def batched_procrustes_alignment(pred_pts, gt_pts):
    B, N, D = pred_pts.shape
    mu_pred = pred_pts.mean(dim=1, keepdim=True)
    mu_gt = gt_pts.mean(dim=1, keepdim=True)
    pred_centered = pred_pts - mu_pred
    gt_centered = gt_pts - mu_gt
    norm_pred = torch.linalg.norm(pred_centered, dim=(1, 2), keepdim=True)
    norm_gt = torch.linalg.norm(gt_centered, dim=(1, 2), keepdim=True)
    pred_normalized = pred_centered / torch.clamp(norm_pred, min=1e-8)
    gt_normalized = gt_centered / torch.clamp(norm_gt, min=1e-8)
    H = torch.bmm(pred_normalized.transpose(1, 2), gt_normalized)
    U, S, Vh = torch.linalg.svd(H)
    R = torch.bmm(U, Vh)
    det = torch.linalg.det(R)
    det_sign = torch.where(det < 0, torch.tensor(-1.0, device=pred_pts.device), torch.tensor(1.0, device=pred_pts.device)).unsqueeze(-1)
    U_fixed = U.clone()
    U_fixed[:, :, 2] *= det_sign
    R_fixed = torch.bmm(U_fixed, Vh)
    S_fixed = S.clone()
    S_fixed[:, 2] *= det_sign.squeeze(-1)
    scale = S_fixed.sum(dim=-1, keepdim=True).unsqueeze(-1) * (norm_gt / torch.clamp(norm_pred, min=1e-8))
    pred_aligned = scale * torch.bmm(pred_centered, R_fixed) + mu_gt
    return pred_aligned

@torch.no_grad()
def evaluate_hmr_batch(preds, targets, cfg, mhr_module):
    metrics = {}
    pred_3d_native = preds["joints_3d"][..., :3]
    gt_3d_native = targets["joints_3d"][..., :3]
    pred_3d_root_rel = pred_3d_native - pred_3d_native[:, 0:1, :]
    gt_3d_root_rel = gt_3d_native - gt_3d_native[:, 0:1, :]
    err_native = torch.linalg.norm(pred_3d_root_rel - gt_3d_root_rel, dim=-1)
    metrics['Head_3D_MPJPE'] = err_native.mean().item() * 1000.0
    pred_3d_aligned = batched_procrustes_alignment(pred_3d_native, gt_3d_native)
    err_native_pa = torch.linalg.norm(pred_3d_aligned - gt_3d_native, dim=-1)
    metrics['Head_3D_PA_MPJPE'] = err_native_pa.mean().item() * 1000.0

    if hasattr(cfg, 'native_mapping_ids') and hasattr(cfg, 'mhr_mapping_ids'):
        pred_mhr_raw = mhr_module.get_joints(preds["mhr_params"], preds["shape_params"])[..., :3]
        pred_mhr_meters = pred_mhr_raw / 100.0
        pred_mhr_meters[..., 1] = -pred_mhr_meters[..., 1]
        pred_mhr_meters[..., 2] = -pred_mhr_meters[..., 2]
        pred_mesh_anchors = pred_mhr_meters[:, cfg.mhr_mapping_ids, :]
        gt_mesh_anchors = gt_3d_native[:, cfg.native_mapping_ids, :]
        pred_mesh_centered = pred_mesh_anchors - pred_mesh_anchors.mean(dim=1, keepdim=True)
        gt_mesh_centered = gt_mesh_anchors - gt_mesh_anchors.mean(dim=1, keepdim=True)
        err_mesh = torch.linalg.norm(pred_mesh_centered - gt_mesh_centered, dim=-1)
        metrics['Mesh_MPJPE'] = err_mesh.mean().item() * 1000.0
        pred_mesh_aligned = batched_procrustes_alignment(pred_mesh_anchors, gt_mesh_anchors)
        err_mesh_pa = torch.linalg.norm(pred_mesh_aligned - gt_mesh_anchors, dim=-1)
        metrics['Mesh_PA_MPJPE'] = err_mesh_pa.mean().item() * 1000.0
    return metrics

# ============================================================
# Cell 11 — Full Distillation Training Loop
# ============================================================
def make_ema_avg_fn(max_decay):
    @torch.no_grad()
    def avg_fn(ema_params, model_params, num_averaged):
        if not (torch.is_floating_point(ema_params[0]) or torch.is_complex(ema_params[0])):
            for e, m in zip(ema_params, model_params):
                e.copy_(m)
            return
        n = num_averaged.item() if torch.is_tensor(num_averaged) else float(num_averaged)
        decay = min(max_decay, (1.0 + n) / (10.0 + n))
        torch._foreach_lerp_(ema_params, model_params, 1.0 - decay)
    return avg_fn

def train_instant_hmr():
    EMA_DECAY = getattr(cfg, "ema_decay", 0.9998)
    EARLY_STOP_PATIENCE = getattr(cfg, "early_stop_patience", 50)  

    print(f"--- Training {cfg.backbone} | EMA decay={EMA_DECAY} | early-stop patience={EARLY_STOP_PATIENCE} ---")
    os.makedirs(cfg.log_dir, exist_ok=True)
    RESUME_FROM_CHECKPOINT = getattr(cfg, "resume", True)

    model = InstantHMRStudent(cfg, pretrained=True).to(device)
    criterion = DistillationLoss(cfg, mhr_module)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler(device='cuda', enabled=cfg.use_amp)

    steps_per_epoch = len(train_loader)
    scheduler = lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.lr, epochs=cfg.epochs,
        steps_per_epoch=steps_per_epoch, pct_start=0.1,
    )

    best_raw_path  = os.path.join(cfg.log_dir, "best_student_model_raw.pth")
    best_ema_path  = os.path.join(cfg.log_dir, "best_student_model_ema.pth")
    best_ckpt_path = os.path.join(cfg.log_dir, "best_student_model_v3.pth")  

    start_epoch = 0
    best_raw_pa = float('inf')
    best_ema_pa = float('inf')
    best_overall_pa = float('inf')
    epochs_no_improve = 0

    if RESUME_FROM_CHECKPOINT and os.path.exists(best_ckpt_path):
        print(f"🔄 Resuming from checkpoint: {best_ckpt_path}")
        ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt.get("epoch", -1) + 1
        best_overall_pa = ckpt.get("val_pa_mpjpe", float('inf'))
        best_raw_pa = best_overall_pa
        best_ema_pa = best_overall_pa
        print(f"✅ Resumed successfully from Epoch {start_epoch}. Best PA-MPJPE was {best_overall_pa:.1f}mm.")
    else:
        print("🚀 Starting training from scratch with fresh weights.")

    ema_model = AveragedModel(model, multi_avg_fn=make_ema_avg_fn(EMA_DECAY), use_buffers=True)

    @torch.no_grad()
    def run_validation(eval_model, tag, epoch):
        eval_model.eval()
        vloss = {'total': 0.0, '2d': 0.0, '3d': 0.0, 'mhr': 0.0, 'struct': 0.0, 'bone': 0.0}
        hmr   = {'Head_3D_MPJPE': 0.0, 'Head_3D_PA_MPJPE': 0.0, 'Mesh_MPJPE': 0.0, 'Mesh_PA_MPJPE': 0.0}
        n = 0
        vbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{cfg.epochs} [Val:{tag}]", leave=False)
        for batch in vbar:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=cfg.use_amp):
                preds_fp16 = eval_model(batch["image"], batch["cliff_cond"])
            preds = {k: v.float() for k, v in preds_fp16.items()}
            losses = criterion(preds, batch)
            if math.isnan(losses['total_loss'].item()) or math.isinf(losses['total_loss'].item()):
                continue
            vloss['total']  += losses['total_loss'].item()
            vloss['2d']     += losses.get('loss_2d_native', torch.tensor(0)).item()
            vloss['3d']     += losses.get('loss_3d_native', torch.tensor(0)).item()
            vloss['mhr']    += losses.get('loss_mhr_joints', torch.tensor(0)).item()
            vloss['struct'] += losses.get('loss_structural', torch.tensor(0)).item()
            vloss['bone']   += losses.get('loss_bone_length', torch.tensor(0)).item()
            bm = evaluate_hmr_batch(preds, batch, cfg, mhr_module)
            for k in hmr:
                hmr[k] += bm.get(k, 0.0)
            n += 1
            vbar.set_postfix({'PA-MPJPE': f"{hmr['Mesh_PA_MPJPE']/max(n,1):.1f}mm"})
        vbar.close()
        vloss = {k: v / max(n, 1) for k, v in vloss.items()}
        hmr   = {k: v / max(n, 1) for k, v in hmr.items()}
        return vloss, hmr, n

    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        tm = {'total': 0.0, '2d': 0.0, 'simcc': 0.0, '3d': 0.0, 'pose': 0.0, 'scale': 0.0, 'shape': 0.0, 'mhr': 0.0, 'struct': 0.0, 'bone': 0.0}
        nb = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs} [Train]")
        for batch in pbar:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=cfg.use_amp):
                preds_fp16 = model(batch["image"], batch["cliff_cond"])
            preds_fp32 = {k: v.float() for k, v in preds_fp16.items()}
            losses = criterion(preds_fp32, batch)
            loss = losses['total_loss']
            loss_val = loss.item()
            if math.isnan(loss_val) or math.isinf(loss_val):
                print("⚠️ WARNING: NaN/Inf loss detected! Skipping step.")
                continue
            if loss_val > cfg.anomaly_loss_threshold:
                print(f"⚠️ WARNING: anomalous loss {loss_val:.1f} > {cfg.anomaly_loss_threshold} — skipping step.")
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            scale_after = scaler.get_scale()
            if scale_before <= scale_after:
                scheduler.step()
                ema_model.update_parameters(model)
            tm['total']  += loss.item()
            tm['2d']     += losses.get('loss_2d_native', torch.tensor(0)).item()
            tm['simcc']  += losses.get('loss_2d_simcc', torch.tensor(0)).item()
            tm['3d']     += losses.get('loss_3d_native', torch.tensor(0)).item()
            tm['pose']   += losses.get('loss_pose', torch.tensor(0)).item()
            tm['scale']  += losses.get('loss_scale', torch.tensor(0)).item()
            tm['mhr']    += losses.get('loss_mhr_joints', torch.tensor(0)).item()
            tm['struct'] += losses.get('loss_structural', torch.tensor(0)).item()
            tm['bone']   += losses.get('loss_bone_length', torch.tensor(0)).item()
            nb += 1
            pbar.set_postfix({'Tot': f"{loss.item():.3f}",
                              'MHR': f"{losses.get('loss_mhr_joints', torch.tensor(0)).item():.3f}"})
        pbar.close()
        avg_train = {k: v / max(nb, 1) for k, v in tm.items()}

        raw_val, raw_hmr, n_raw = run_validation(model, "raw", epoch)
        ema_val, ema_hmr, n_ema = run_validation(ema_model, "ema", epoch)

        print(f"\n📈 Epoch {epoch+1} Summary | LR: {scheduler.get_last_lr()[0]:.2e}")
        print(f"   [Train]   Tot: {avg_train['total']:.4f} | MHR: {avg_train['mhr']:.4f} | 2D: {avg_train['2d']:.4f} "
              f"| 3D: {avg_train['3d']:.4f} | Bone: {avg_train['bone']:.4f} | simcc: {avg_train['simcc']:.3f}")
        print(f"   [Val RAW] Tot: {raw_val['total']:.4f} | Mesh PA-MPJPE: {raw_hmr['Mesh_PA_MPJPE']:.1f} | Mesh MPJPE: {raw_hmr['Mesh_MPJPE']:.1f} | Head PA: {raw_hmr['Head_3D_PA_MPJPE']:.1f} | Head MPJPE: {raw_hmr['Head_3D_MPJPE']:.1f} mm")
        print(f"   [Val EMA] Tot: {ema_val['total']:.4f} | Mesh PA-MPJPE: {ema_hmr['Mesh_PA_MPJPE']:.1f} | Mesh MPJPE: {ema_hmr['Mesh_MPJPE']:.1f} | Head PA: {ema_hmr['Head_3D_PA_MPJPE']:.1f} | Head MPJPE: {ema_hmr['Head_3D_MPJPE']:.1f} mm")
        delta = raw_hmr['Mesh_PA_MPJPE'] - ema_hmr['Mesh_PA_MPJPE']
        print(f"   🔬 EMA vs RAW (Mesh PA-MPJPE): {'EMA better' if delta > 0 else 'RAW better'} by {abs(delta):.1f} mm")

        def get_save_dict(model_to_save, val_metric, source_name):
            return {
                'epoch': epoch,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_pa_mpjpe': val_metric,
                'source': source_name
            }

        improved_raw = n_raw > 0 and raw_hmr['Mesh_PA_MPJPE'] < best_raw_pa
        improved_ema = n_ema > 0 and ema_hmr['Mesh_PA_MPJPE'] < best_ema_pa

        if improved_raw:
            best_raw_pa = raw_hmr['Mesh_PA_MPJPE']
            torch.save(get_save_dict(model, best_raw_pa, 'raw'), best_raw_path)
        if improved_ema:
            best_ema_pa = ema_hmr['Mesh_PA_MPJPE']
            torch.save(get_save_dict(ema_model.module, best_ema_pa, 'ema'), best_ema_path)

        current = min(raw_hmr['Mesh_PA_MPJPE'], ema_hmr['Mesh_PA_MPJPE'])
        if current < best_overall_pa - 1e-4:
            best_overall_pa = current
            use_ema = ema_hmr['Mesh_PA_MPJPE'] <= raw_hmr['Mesh_PA_MPJPE']
            better = ema_model.module if use_ema else model
            torch.save(get_save_dict(better, best_overall_pa, 'ema' if use_ema else 'raw'), best_ckpt_path)
            print(f"💾 New best overall: {best_overall_pa:.1f} mm ({'EMA' if use_ema else 'RAW'}) -> {os.path.basename(best_ckpt_path)}")

        if improved_raw or improved_ema:
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"   ⏳ Neither RAW nor EMA improved: {epochs_no_improve}/{EARLY_STOP_PATIENCE} (best raw {best_raw_pa:.1f} / ema {best_ema_pa:.1f} mm)")
            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                print(f"\n🛑 Early stopping at epoch {epoch+1}: neither RAW nor EMA improved for {EARLY_STOP_PATIENCE} epochs.")
                break

    print(f"\n🎉 Training complete! Best overall Mesh PA-MPJPE: {best_overall_pa:.1f} mm")
    print(f"   best RAW: {best_raw_pa:.1f} mm -> {best_raw_path}")
    print(f"   best EMA: {best_ema_pa:.1f} mm -> {best_ema_path}")

# ============================================================
# Cell 15 — Export & Static Quantization
# ============================================================
class HMRDeployWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image, cliff_cond):
        out = self.model(image, cliff_cond)
        return (out["mhr_params"], out["shape_params"], out["cam_trans"],
                out["joints_2d"], out["joints_3d"])

def topological_sort_onnx(model):
    available = set()
    for inp in model.graph.input:
        available.add(inp.name)
    for init in model.graph.initializer:
        available.add(init.name)

    remaining = list(model.graph.node)
    sorted_nodes = []

    while remaining:
        progress = False
        for node in remaining:
            if all(inp in available or inp == '' for inp in node.input):
                sorted_nodes.append(node)
                remaining.remove(node)
                for out in node.output:
                    available.add(out)
                progress = True
                break
        if not progress:
            sorted_nodes.extend(remaining)
            break

    del model.graph.node[:]
    model.graph.node.extend(sorted_nodes)
    return model

def export_and_quantize():
    print("--- Exporting and Quantizing Full Model for Deployment ---")
    try:
        from onnxruntime.quantization import (
            CalibrationDataReader, quantize_static,
            QuantType, QuantFormat, CalibrationMethod,
        )

        class HMRCalibrationDataReader(CalibrationDataReader):
            def __init__(self, dataloader, max_samples=100):
                self.data_list = []
                print(f"Extracting {max_samples} individual images for INT8 calibration...")
                for batch in dataloader:
                    images = batch["image"].numpy().astype(np.float32)
                    cliffs = batch["cliff_cond"].numpy().astype(np.float32)
                    for b_idx in range(images.shape[0]):
                        if len(self.data_list) >= max_samples:
                            break
                        self.data_list.append({
                            "image": images[b_idx:b_idx + 1],
                            "cliff_cond": cliffs[b_idx:b_idx + 1]
                        })
                    if len(self.data_list) >= max_samples:
                        break
                self.enum_data = iter(self.data_list)

            def get_next(self):
                return next(self.enum_data, None)

    except ImportError:
        print("⚠️ 'onnxruntime' is not installed. INT8 quantization will fail.")
        HMRCalibrationDataReader = None

    export_dir = Path(cfg.log_dir) / "export"
    export_dir.mkdir(parents=True, exist_ok=True)

    onnx_fp32_path = export_dir / "sam3d_full_fp32.onnx"
    onnx_fp16_path = export_dir / "sam3d_full_fp16.onnx"
    onnx_int8_path = export_dir / "sam3d_full_int8.onnx"

    output_keys = ["mhr_params", "shape_params", "cam_trans", "joints_2d", "joints_3d"]
    export_target_model = InstantHMRStudent(cfg, pretrained=False).to(device)
    best_ckpt_path = os.path.join(cfg.log_dir, "best_student_model_v3.pth")

    if os.path.exists(best_ckpt_path):
        ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=True)
        clean_state_dict = {k.replace('_orig_mod.', ''): v for k, v in ckpt['model_state_dict'].items()}
        export_target_model.load_state_dict(clean_state_dict, strict=False)
        print(f"✅ Loaded best model from epoch {ckpt.get('epoch', 0)+1}.\n")

        deploy_model = HMRDeployWrapper(export_target_model)
        deploy_model.eval()

        dummy_img = torch.randn(1, 3, cfg.image_size, cfg.image_size, device=device)
        dummy_cliff = torch.randn(1, 3, device=device)

        dynamic_axes = {"image": {0: "batch"}, "cliff_cond": {0: "batch"}}
        for k in output_keys:
            dynamic_axes[k] = {0: "batch"}

        try:
            import onnx
            torch.onnx.export(
                deploy_model, (dummy_img, dummy_cliff), str(onnx_fp32_path),
                input_names=["image", "cliff_cond"], output_names=output_keys,
                dynamic_axes=dynamic_axes, opset_version=17
            )
            print(f"✅ FP32 saved: {onnx_fp32_path.name} | Size: {onnx_fp32_path.stat().st_size / 1e6:.1f} MB")
        except Exception as e:
            print(f"❌ FP32 export failed: {e}")

        try:
            from onnxruntime.transformers.float16 import convert_float_to_float16
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model_fp32 = onnx.load(str(onnx_fp32_path))
                model_fp16 = convert_float_to_float16(model_fp32, keep_io_types=True)
                model_fp16 = topological_sort_onnx(model_fp16)
                onnx.save(model_fp16, str(onnx_fp16_path))
            onnx.checker.check_model(onnx.load(str(onnx_fp16_path)), full_check=True)
            print(f"✅ FP16 saved: {onnx_fp16_path.name} | Size: {onnx_fp16_path.stat().st_size / 1e6:.1f} MB")
        except ImportError:
            print("⚠️ FP16 skipped (onnxruntime.transformers not available)")
        except Exception as e:
            print(f"❌ FP16 conversion failed: {e}")

        try:
            calib_reader = HMRCalibrationDataReader(train_loader, max_samples=100)
            calib_reader.enum_data = iter(calib_reader.data_list)
            fp32_model = onnx.load(str(onnx_fp32_path))
            nodes_to_exclude = []
            for node in fp32_model.graph.node:
                name = node.name or ''
                if any(kw in name for kw in ['/model/transformer/', '/model/head_',
                                              '/model/feat_proj', '/model/cond_proj']):
                    nodes_to_exclude.append(name)
            del fp32_model
            print(f"  Excluding {len(nodes_to_exclude)} transformer/head nodes from quantization")

            quantize_static(
                model_input=str(onnx_fp32_path),
                model_output=str(onnx_int8_path),
                calibration_data_reader=calib_reader,
                quant_format=QuantFormat.QDQ,
                op_types_to_quantize=['Conv'],
                per_channel=True,
                reduce_range=False,
                activation_type=QuantType.QInt8,
                weight_type=QuantType.QInt8,
                nodes_to_exclude=nodes_to_exclude,
                calibrate_method=CalibrationMethod.Percentile,
                extra_options={'CalibTensorRangeSymmetric': True},
            )
            print(f"✅ INT8 saved: {onnx_int8_path.name} | Size: {onnx_int8_path.stat().st_size / 1e6:.1f} MB")
        except NameError:
            print("⚠️ INT8 skipped (train_loader not available)")
        except Exception as e:
            import traceback
            print(f"❌ INT8 conversion failed: {e}")
            traceback.print_exc()

        sd_path = export_dir / "sam3d_student_state_dict.pth"
        torch.save(export_target_model.state_dict(), sd_path)
        print(f"\n✅ PyTorch state dict saved: {sd_path.name}")
    else:
        print("⚠️ No checkpoint found at:", best_ckpt_path)

# ============================================================
# Self-tests
# ============================================================
def _mock_preds_from(batch):
    return {
        "mhr_params": batch["mhr_model_params"].clone(),
        "shape_params": batch["shape_params"].clone(),
        "cam_trans": batch["cam_trans"].clone(),
        "joints_2d": batch["joints_2d"].clone(),
        "joints_3d": batch["joints_3d"].clone(),
    }

def run_self_tests():
    print("--- Architecture Output Shapes ---")
    test_model = InstantHMRStudent(cfg, pretrained=False)
    out = test_model(torch.randn(2, 3, 224, 224), torch.randn(2, 3))
    for k, v in out.items():
        print(f"  {k}: {tuple(v.shape)}")

    criterion = DistillationLoss(cfg, mhr_module)
    print("\n--- Perfect Student (identity batch) ---")
    clean_ds = SAM3DStudentDataset(cfg.data_root, augment=False,
                                   max_images=256, per_dataset_caps=cfg.per_dataset_caps)
    clean_loader = DataLoader(clean_ds, batch_size=32, shuffle=False, num_workers=0)
    cb = {k: v.to(device) for k, v in next(iter(clean_loader)).items()}
    l_clean = criterion(_mock_preds_from(cb), cb)
    for k, v in l_clean.items():
        print(f"  {k:<20}: {v.item():.6f}")

    print("\n--- MHR consistency under geometric augmentation ---")
    aug_ds = SAM3DStudentDataset(cfg.data_root, augment=True,
                                 max_images=256, per_dataset_caps=cfg.per_dataset_caps,
                                 geom_p=1.0, geom_rot_deg=cfg.geom_rot_deg,
                                 geom_scale_range=cfg.geom_scale_range,
                                 geom_trans=cfg.geom_trans, geom_flip_p=0.0)
    aug_loader = DataLoader(aug_ds, batch_size=32, shuffle=False, num_workers=0)
    ab = {k: v.to(device) for k, v in next(iter(aug_loader)).items()}

    with torch.no_grad():
        tgt_mhr = mhr_module.get_joints(ab["mhr_model_params"], ab["shape_params"])[..., :3]
        cth = ab["aug_cos"].view(-1, 1)
        sth = ab["aug_sin"].view(-1, 1)
        xm, ym = tgt_mhr[..., 0], tgt_mhr[..., 1]
        tgt_mhr_rot = torch.stack([cth * xm - sth * ym, sth * xm + cth * ym, tgt_mhr[..., 2]], dim=-1)
        mhr_vis = tgt_mhr_rot / 100.0
        mhr_vis[..., 1] = -mhr_vis[..., 1]
        mhr_vis[..., 2] = -mhr_vis[..., 2]
        d_aug = (ab["joints_3d"][:, cfg.native_mapping_ids, :]
                 - mhr_vis[:, cfg.mhr_mapping_ids, :]).abs().mean().item()
        tgt_mhr_c = mhr_module.get_joints(cb["mhr_model_params"], cb["shape_params"])[..., :3]
        mhr_vis_c = tgt_mhr_c / 100.0
        mhr_vis_c[..., 1] = -mhr_vis_c[..., 1]
        mhr_vis_c[..., 2] = -mhr_vis_c[..., 2]
        d_clean = (cb["joints_3d"][:, cfg.native_mapping_ids, :]
                   - mhr_vis_c[:, cfg.mhr_mapping_ids, :]).abs().mean().item()
        print(f"  native<->MHR anchor residual: clean {d_clean*1000:.2f} mm | augmented {d_aug*1000:.2f} mm")
        ok_struct = d_aug < d_clean * 1.5 + 1e-3

        cc, ss = cth, sth
        def counter(pts):
            x, y = pts[..., 0], pts[..., 1]
            return torch.stack([cc * x - ss * y, ss * x + cc * y, pts[..., 2]], dim=-1)

        cam_cr = torch.cat([cc * ab["cam_trans"][:, 0:1] - ss * ab["cam_trans"][:, 1:2],
                            ss * ab["cam_trans"][:, 0:1] + cc * ab["cam_trans"][:, 1:2],
                            ab["cam_trans"][:, 2:3]], dim=1)
        reproj_aug = criterion._project_3d_to_norm_2d(counter(ab["joints_3d"]), cam_cr, ab)
        r_aug = (reproj_aug - ab["joints_2d"]).abs().mean().item()
        reproj_clean = criterion._project_3d_to_norm_2d(cb["joints_3d"], cb["cam_trans"], cb)
        r_clean = (reproj_clean - cb["joints_2d"]).abs().mean().item()
        print(f"  reproj(counter-rot + M) residual: clean {r_clean:.4f} | augmented {r_aug:.4f} (normalised units)")
        ok_reproj = r_aug < r_clean * 1.5 + 5e-3

    print("\n--- Perfect Student (augmented batch, masked losses) ---")
    l_aug = criterion(_mock_preds_from(ab), ab)
    for k, v in l_aug.items():
        print(f"  {k:<20}: {v.item():.6f}")
    keys = ['loss_2d_native', 'loss_3d_native', 'loss_cam', 'loss_reproj_native', 'loss_bone_length']
    ok_loss = all(l_aug[k].item() < max(5 * l_clean.get(k, torch.tensor(1e-3)).item(), 1e-3) for k in keys if k in l_aug)

    print()
    print(f"  structural-consistency : {'✅' if ok_struct else '❌'}")
    print(f"  reprojection-pathway   : {'✅' if ok_reproj else '❌'}")
    print(f"  masked perfect-student : {'✅' if ok_loss else '❌'}")
    if ok_struct and ok_reproj and ok_loss:
        print("\n✅ SELF-TEST PASSED: augmentation is consistent with the MHR pipeline.")
    else:
        print("\n❌ SELF-TEST FAILED — do not launch a long run.")

def run_overfit_test(steps=1000, subset=8, lr=5e-4):
    print(f"--- 1-Batch Overfit Test ({subset} images, {steps} steps) ---")
    print("Notice: Geometric augmentations are FORCED OFF to allow loss_pose to stabilize the mesh.")
    
    model = InstantHMRStudent(cfg, pretrained=True).to(device)
    criterion = DistillationLoss(cfg, mhr_module)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler(device='cuda', enabled=cfg.use_amp)

    # 1. SURGICAL INSERTION: Create a dedicated, non-augmented dataset for the overfit test
    overfit_dataset = SAM3DStudentDataset(
        cfg.data_root,
        augment=True, # Keep color/blur augs to prevent total collapse
        max_images=subset,
        per_dataset_caps=cfg.per_dataset_caps,
        geom_p=0.0,         # STRICTLY DISABLE SPATIAL AUGS
        geom_flip_p=0.0     # STRICTLY DISABLE FLIPS
    )
    overfit_loader = DataLoader(overfit_dataset, batch_size=subset, shuffle=False, num_workers=0)
    batch = next(iter(overfit_loader))
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    def reg(ls):
        return (ls['loss_2d_native'].item() + ls['loss_3d_native'].item()
                + ls['loss_cam'].item() + ls['loss_pose'].item() + ls.get('loss_bone_length', torch.tensor(0)).item())

    first_reg = last_reg = last_pa = None
    best_pa = float('inf')   # 8-sample overfit is noisy at constant LR; judge the best fit reached
    n_skipped = 0
    model.train()

    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=cfg.use_amp):
            preds = model(batch["image"], batch["cliff_cond"])
        preds = {k: v.float() for k, v in preds.items()}
        losses = criterion(preds, batch)
        
        lv = losses['total_loss'].item()
        if math.isnan(lv) or math.isinf(lv) or lv > cfg.anomaly_loss_threshold:
            n_skipped += 1
            continue
            
        scaler.scale(losses['total_loss']).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        
        if first_reg is None:
            first_reg = reg(losses)
        last_reg = reg(losses)
        
        if step % 50 == 0 or step == steps - 1:
            model.eval()
            with torch.no_grad():
                p = {k: v.float() for k, v in model(batch["image"], batch["cliff_cond"]).items()}
                m = evaluate_hmr_batch(p, batch, cfg, mhr_module)
            model.train()
            last_pa = m['Head_3D_PA_MPJPE']
            best_pa = min(best_pa, last_pa)

            print(f"  step {step:04d} | tot {losses['total_loss'].item():.3f} "
                  f"| 3d {losses['loss_3d_native'].item():.4f} "
                  f"| pose {losses['loss_pose'].item():.4f} "
                  f"| shape {losses.get('loss_shape', torch.tensor(0)).item():.4f} "
                  f"| mhr {losses['loss_mhr_joints'].item():.4f} "
                  f"| Head [MPJPE: {m['Head_3D_MPJPE']:.1f} | PA: {m['Head_3D_PA_MPJPE']:.1f}] mm "
                  f"| Mesh [MPJPE: {m['Mesh_MPJPE']:.1f} | PA: {m['Mesh_PA_MPJPE']:.1f}] mm")

    ok = (last_reg < first_reg * 0.10) and (best_pa < 40.0)
    print(f"\n{'✅ SUCCESS' if ok else '❌ FAIL'}: regression loss {first_reg:.4f} -> {last_reg:.5f} "
          f"| best Head PA-MPJPE {best_pa:.1f} mm (final {last_pa:.1f}) | anomalous steps skipped: {n_skipped}")
    return ok

def parse_args():
    p = argparse.ArgumentParser(description="Standalone InstantHMR distillation training.")
    p.add_argument("--data_root", type=str, default=None,
                   help="Entry-point folder containing sub-folders with annotations/ + images/.")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Where checkpoints/logs/exports are written (cfg.log_dir).")
    p.add_argument("--mhr_model_path", type=str, default=None,
                   help="Path to the TorchScript MHR model (mhr_model.pt).")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--max_images", type=int, default=None)
    p.add_argument("--harmony4d_cap", type=int, default=None,
                   help="Max Harmony4D crops kept (randomly sampled). 0 = uncapped/use all.")
    p.add_argument("--no-resume", dest="no_resume", action="store_true",
                   help="Ignore any existing checkpoint and train from scratch.")
    p.add_argument("--self-test", dest="self_test", action="store_true",
                   help="Run the geometry/MHR-consistency checks and exit.")
    p.add_argument("--overfit-test", dest="overfit_test", action="store_true",
                   help="Run the 1-batch overfit sanity check and exit.")
    p.add_argument("--no-export", dest="no_export", action="store_true",
                   help="Skip the ONNX export / quantization step after training.")
    p.add_argument("--gpu", type=int, default=None, help="GPU count (informational).")
    args, unknown = p.parse_known_args()
    if unknown:
        print(f"⚠️ Ignoring unrecognized arguments: {unknown}")
    return args

cfg = None
train_loader = None
val_loader = None
full_dataset = None
mhr_module = None

def main():
    global cfg, train_loader, val_loader, full_dataset, mhr_module

    args = parse_args()
    cfg = DistillConfig()
    if args.data_root is not None:      cfg.data_root = args.data_root
    if args.output_dir is not None:     cfg.log_dir = args.output_dir
    if args.mhr_model_path is not None: cfg.mhr_model_path = args.mhr_model_path
    if args.epochs is not None:         cfg.epochs = args.epochs
    if args.batch_size is not None:     cfg.batch_size = args.batch_size
    if args.lr is not None:             cfg.lr = args.lr
    if args.num_workers is not None:    cfg.num_workers = args.num_workers
    if args.max_images is not None:     cfg.max_images = args.max_images
    if args.harmony4d_cap is not None:
        if args.harmony4d_cap <= 0:
            cfg.per_dataset_caps.pop("sam3d_gt_harmony4d", None)
        else:
            cfg.per_dataset_caps["sam3d_gt_harmony4d"] = args.harmony4d_cap
    if args.no_resume:                  cfg.resume = False

    print("=" * 60)
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Data root    : {cfg.data_root}")
    print(f"Output dir   : {cfg.log_dir}")
    print(f"MHR model    : {cfg.mhr_model_path}")
    print(f"Backbone     : {cfg.backbone} | epochs={cfg.epochs} | batch={cfg.batch_size} | lr={cfg.lr}")
    print(f"max_images   : {cfg.max_images if cfg.max_images is not None else 'unlimited (all crops)'}")
    print(f"dataset caps : {cfg.per_dataset_caps if cfg.per_dataset_caps else 'none'}")
    print(f"geom aug     : p={cfg.geom_p} rot±{cfg.geom_rot_deg}° scale±{cfg.geom_scale_range} "
          f"trans±{cfg.geom_trans} flip={cfg.geom_flip_p}")
    print(f"2D head      : SimCC soft-argmax, {cfg.kp2d_bins} bins over ±{cfg.kp2d_range} (w_simcc={cfg.w_simcc})")
    print(f"Bone Length  : Soft Prior weight (w_bone_length={cfg.w_bone_length}) active on {len(cfg.major_bones)} links")
    print("=" * 60)

    if not Path(cfg.mhr_model_path).exists():
        raise FileNotFoundError(
            f"MHR model not found at '{cfg.mhr_model_path}'. "
            f"Place mhr_model.pt in the checkpoints/ folder or pass --mhr_model_path."
        )

    train_loader, val_loader, full_dataset = build_dataloaders(cfg)
    mhr_module = MHRForwardPass(cfg.mhr_model_path, device)

    if args.self_test:
        run_self_tests()
        return
    if args.overfit_test:
        run_overfit_test()
        return

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    train_instant_hmr()

    if not args.no_export:
        export_and_quantize()

if __name__ == "__main__":
    main()