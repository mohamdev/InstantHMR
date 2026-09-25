"""Usage: python tools/profiling/export_family.py <out_dir>  then  python tools/profiling/profile_family.py <out_dir>
Ticket 03: ONNX exports of the g8h architecture with the backbone swapped.
Random weights for B0/B1/B4, plus the trained g8h, all through one path."""
import io, sys, copy, json
from pathlib import Path
import torch, onnx, onnxsim
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "instanthmr_distill_train"))
import train_distill_mhr_only as T

CKPT = REPO / "instanthmr_distill_train/runs/g8h_s0/g8h_s0/best_student_model_ema.pth"
OUT = Path(sys.argv[1])
blob = torch.load(CKPT, map_location="cpu", weights_only=False)
state = {k.replace("_orig_mod.", ""): v for k, v in blob["model_state_dict"].items()}
cfg, _ = T.config_from_checkpoint(state, CKPT)
cfg.kp2d_bins = state["head_2d_logits.bias"].shape[0] // 2
cfg.kp2d_range = float(state["kp2d_bin_centers"][-1])

class Deploy(torch.nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, image, cliff_cond):
        o = self.m(image, cliff_cond)
        return o["mhr_params"], o["shape_params"], o["cam_trans"], o["joints_2d"]

def export(model, name):
    model.eval()
    buf = io.BytesIO()
    torch.onnx.export(Deploy(model), (torch.zeros(1, 3, 224, 224), torch.zeros(1, 3)), buf,
                      input_names=["image", "cliff_cond"],
                      output_names=["mhr_params", "shape_params", "cam_trans", "joints_2d"],
                      opset_version=17)
    m, ok = onnxsim.simplify(onnx.load_from_string(buf.getvalue()))
    assert ok
    p = OUT / f"{name}.onnx"; onnx.save(m, p)
    n = sum(x.numel() for x in model.parameters())
    nb = sum(x.numel() for x in model.backbone.parameters())
    print(json.dumps(dict(name=name, params_M=round(n/1e6, 2), backbone_M=round(nb/1e6, 2),
                          mb=round(p.stat().st_size/2**20, 1))), flush=True)

trained = T.InstantHMRStudent(copy.deepcopy(cfg), pretrained=False)
missing, unexpected = trained.load_state_dict(state, strict=False)
assert not missing and not unexpected, (missing[:5], unexpected[:5])
export(trained, "g8h_trained_b4")
for bb in ("hgnetv2_b4", "hgnetv2_b1", "hgnetv2_b0"):
    c = copy.deepcopy(cfg); c.backbone = bb
    torch.manual_seed(0)
    export(T.InstantHMRStudent(c, pretrained=False), f"random_{bb.split('_')[1]}")
