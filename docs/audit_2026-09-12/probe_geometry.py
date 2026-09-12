"""Read-only CPU audit; run from any directory in the existing sam3d_video_312 environment.
Requires the adjacent sam3-biomechanics checkout and its DINOv3 checkpoint.
"""
import torch,numpy as np,json
from pathlib import Path
import importlib.util

torch.set_num_threads(4)
root=Path(__file__).resolve().parents[2]
s=torch.load(root.parent/'sam3-biomechanics/video_to_pose_pipeline/checkpoints/sam-3d-body-dinov3/model.ckpt',map_location='cpu',weights_only=False,mmap=True)
rig=torch.jit.load(str(root/'checkpoints/mhr_model.pt'),map_location='cpu').eval()
for param in rig.parameters(): param.requires_grad_(False)
ct=rig.character_torch
print('limits repr',ct.parameter_limits)
print('limits attributes',ct.parameter_limits._c.dump_to_str(True,False,False)[:5000])
paths=sorted((root/'data/sam3d_gt_coco/annotations').glob('*.npz'))
rng=np.random.default_rng(20260912)
paths=[paths[i] for i in rng.choice(len(paths),6000,replace=False)[:256]]
r=[]
for path in paths:
 with np.load(path) as z:r.append((z['mhr_model_params'],z['shape_params']))
p,h=[torch.from_numpy(np.stack([a[i] for a in r])).float() for i in range(2)]
C=s['head_pose.scale_comps'].double(); mu=s['head_pose.scale_mean'].double()
coeff=(p[:,136:].double()-mu)@torch.linalg.pinv(C,rtol=1e-5)
print('stable_pca_max_residual',float((coeff@C+mu-p[:,136:]).abs().max()),'coeff_max',float(coeff.abs().max()))
W=torch.from_numpy(np.load(root/'instanthmr_distill_train/assets/mhr_j127_to_kp70.npy')).float()
K=s['head_pose.keypoint_mapping'][:70].float()
hand=torch.cat([s['head_pose.hand_joint_idxs_left'],s['head_pose.hand_joint_idxs_right']]).long()
body=list(range(21))+[41]+list(range(62,70))
err=[]; exacterr=[]; approxerr=[]; shapeerr=[]; shapekp=[]; pcerr=[]; pcent=[]
for start in range(0,len(p),16):
 pp=p[start:start+16]; hh=h[start:start+16]
 with torch.no_grad():
  vv,ss=rig(hh,pp,torch.zeros(len(pp),72)); ss=ss[...,:3]; kp=K@torch.cat([vv,ss],1); ka=W@ss
  err.append((ka-kp).norm(dim=-1)*10)
  ph=pp.clone(); ph[:,hand]=0
  vh,sh=rig(hh,ph,torch.zeros(len(pp),72)); kh=K@torch.cat([vh,sh[...,:3]],1)
  exacterr.append((kh-kp).norm(dim=-1)*10); approxerr.append(((W@sh[...,:3])-ka).norm(dim=-1)*10)
  v0,s0=rig(torch.zeros_like(hh),pp,torch.zeros(len(pp),72)); k0=K@torch.cat([v0,s0[...,:3]],1)
  shapeerr.append((v0-vv).norm(dim=-1)*10);shapekp.append((k0-kp).norm(dim=-1)*10)
  pc=pp.clone();pc[:,136:]=(coeff[start:start+16]@C+mu).float()
  vc,sc=rig(hh,pc,torch.zeros(len(pp),72));pcerr.append((vc-vv).norm(dim=-1)*10)
  # Rig origin is not hip-normalized: measure change of midhip induced by scale only.
  kc=K@torch.cat([vc,sc[...,:3]],1)
  pcent.append((((kp[:,9]+kp[:,10])-(kc[:,9]+kc[:,10]))/2).norm(dim=-1)*10)
for name,vals in [('regressor error',err),('hand zero exact',exacterr),('hand zero approx',approxerr),('shape zero exactkp',shapekp)]:
 e=torch.cat(vals);print(name,'body mean/max',e[:,body].mean().item(),e[:,body].max().item(),'top means',[(i,e[:,i].mean().item()) for i in e.mean(0).argsort(descending=True)[:12].tolist()])
print('pca24 PVE mean/max',torch.cat(pcerr).mean().item(),torch.cat(pcerr).max().item())
print('pca24 midhip movement max',torch.cat(pcent).max().item())
print('shapezero pve',torch.cat(shapeerr).mean().item())
print('student regressor row sum range',W.sum(1).min().item(),W.sum(1).max().item())
