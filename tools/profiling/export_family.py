"""ONNX exports of the g8h architecture with the backbone and input size swapped.

    python tools/profiling/export_family.py --out /tmp/fam                       # ticket 03
    python tools/profiling/export_family.py --out /tmp/fam288 --image-size 288 \
        --backbones b4 --no-trained                                               # ticket 06

Random weights (latency does not depend on them; ticket 03 measured random B4
within 2-6% of the trained g8h on every phone), plus optionally the trained
g8h. Network only, no FK / mesh: that is what differs between members.
Then profile with tools/profiling/profile_family.py.
"""
import argparse, copy, io, json, sys
from pathlib import Path
import torch, onnx, onnxsim
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "instanthmr_distill_train"))
import train_distill_mhr_only as T

CKPT = REPO / "instanthmr_distill_train/runs/g8h_s0/g8h_s0/best_student_model_ema.pth"


class Deploy(torch.nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, image, cliff_cond):
        o = self.m(image, cliff_cond)
        return o["mhr_params"], o["shape_params"], o["cam_trans"], o["joints_2d"]


def export(model, path: Path, size: int):
    model.eval()
    buf = io.BytesIO()
    torch.onnx.export(Deploy(model), (torch.zeros(1, 3, size, size), torch.zeros(1, 3)), buf,
                      input_names=["image", "cliff_cond"],
                      output_names=["mhr_params", "shape_params", "cam_trans", "joints_2d"],
                      opset_version=17)
    m, ok = onnxsim.simplify(onnx.load_from_string(buf.getvalue()))
    assert ok
    onnx.save(m, path)
    n = sum(x.numel() for x in model.parameters())
    nb = sum(x.numel() for x in model.backbone.parameters())
    print(json.dumps(dict(name=path.stem, image_size=size, params_M=round(n / 1e6, 2),
                          backbone_M=round(nb / 1e6, 2))), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--backbones", default="b4,b1,b0", help="hgnetv2 sizes, comma-separated")
    p.add_argument("--no-trained", action="store_true", help="skip the trained g8h export")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    blob = torch.load(CKPT, map_location="cpu", weights_only=False)
    state = {k.replace("_orig_mod.", ""): v for k, v in blob["model_state_dict"].items()}
    cfg, _ = T.config_from_checkpoint(state, CKPT)
    cfg.kp2d_bins = state["head_2d_logits.bias"].shape[0] // 2
    cfg.kp2d_range = float(state["kp2d_bin_centers"][-1])

    if not args.no_trained:
        if args.image_size != cfg.image_size:
            sys.exit("the trained g8h is a 224 model; pass --no-trained with --image-size")
        trained = T.InstantHMRStudent(copy.deepcopy(cfg), pretrained=False)
        missing, unexpected = trained.load_state_dict(state, strict=False)
        assert not missing and not unexpected, (missing[:5], unexpected[:5])
        export(trained, args.out / "g8h_trained_b4.onnx", args.image_size)
    for b in args.backbones.split(","):
        c = copy.deepcopy(cfg)
        c.backbone, c.image_size = f"hgnetv2_{b}", args.image_size
        torch.manual_seed(0)
        export(T.InstantHMRStudent(c, pretrained=False),
               args.out / f"random_{b}_{args.image_size}.onnx", args.image_size)


if __name__ == "__main__":
    main()
