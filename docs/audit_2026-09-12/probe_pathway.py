"""Read-only CPU audit; run from any directory in the existing sam3d_video_312 environment.
Requires the adjacent sam3-biomechanics checkout and its DINOv3 checkpoint.
"""
import importlib.util, json, sys
from pathlib import Path
import numpy as np
import torch
import roma
from PIL import Image

torch.set_num_threads(4)
root=Path(__file__).resolve().parents[2]
teacher=root.parent/'sam3-biomechanics/sam-3d-body'
modpath=teacher/'sam_3d_body/models/modules/mhr_utils.py'
spec=importlib.util.spec_from_file_location('mhr_utils',modpath)
u=importlib.util.module_from_spec(spec); spec.loader.exec_module(u)
s=torch.load(root.parent/'sam3-biomechanics/video_to_pose_pipeline/checkpoints/sam-3d-body-dinov3/model.ckpt',map_location='cpu',weights_only=False,mmap=True)
C=s['head_pose.scale_comps'].double(); mu=s['head_pose.scale_mean'].double(); pinv=torch.linalg.pinv(C)
print('scale singular values',torch.linalg.svdvals(C).tolist(),flush=True)
rig=torch.jit.load(str(root/'checkpoints/mhr_model.pt'),map_location='cpu').eval()
for p in rig.parameters(): p.requires_grad_(False)
W=torch.from_numpy(np.load(root/'instanthmr_distill_train/assets/mhr_j127_to_kp70.npy')).float()
K=s['head_pose.keypoint_mapping'][:70].float()
hand_idx=torch.cat([s['head_pose.hand_joint_idxs_left'],s['head_pose.hand_joint_idxs_right']]).long()
body_idx=[i for i in range(70) if i not in list(range(21,41))+list(range(42,62))]
print('hand rig idx',hand_idx.tolist(),flush=True)
print('keypoint vertex support',int((K[:,:18439].abs().sum(0)>0).sum()),flush=True)

def stat(x):
 x=torch.as_tensor(x).double().reshape(-1)
 return {k:float(v) for k,v in [('mean',x.mean()),('p95',torch.quantile(x,.95)),('max',x.max())]}

def skel(p,h):
 ct=rig.character_torch
 return ct.joint_parameters_to_skeleton_state(ct.model_parameters_to_joint_parameters(torch.cat([p,h],1)))[...,:3]

out={'seed':20260912,'scale_rank':int(torch.linalg.matrix_rank(C)), 'scale_singular_values':torch.linalg.svdvals(C).tolist(),'datasets':{}}
for split,n in [('coco',6000),('aic',1000),('mpii',1000),('harmony4d',1000),('3dpw',1000)]:
 paths=sorted((root/f'data/sam3d_gt_{split}/annotations').glob('*.npz'))
 rng=np.random.default_rng(20260912)
 chosen=[paths[i] for i in rng.choice(len(paths),min(n,len(paths)),replace=False)]
 records=[]
 for path in chosen:
  with np.load(path) as z: records.append([z[k].copy() for k in ['mhr_model_params','shape_params','joints_3d','joints_2d','bbox_square']])
 p,h,j,j2,bbox=[torch.from_numpy(np.stack([r[i] for r in records])).float() for i in range(5)]
 b=torch.cat([p[:,6:136],torch.zeros(len(p),3)],1)
 bc=u.compact_model_params_to_cont_body(b); br=u.compact_cont_to_model_params_body(bc)
 pr=p.clone(); pr[:,6:136]=br[:,:130]
 scales=((p[:,136:].double()-mu)@pinv)@C+mu
 rootR=roma.euler_to_rotmat('ZYX',p[:,3:6])
 root_e=roma.rotmat_to_euler('ZYX',rootR)
 root_rig=roma.euler_to_rotmat('xyz',p[:,3:6])
 round_rig=roma.euler_to_rotmat('xyz',root_e)
 naive_e=roma.rotmat_to_euler('ZYX',u.batch6DFromXYZ(p[:,3:6],True))
 d={'n':len(p),'root_fraction_abs_gt_2_8':float((p[:,3:6].abs()>2.8).float().mean()),'root_near_pi_sample_fraction':float(((p[:,3:6].abs()-torch.pi).abs()<.34).any(1).float().mean()),'root_near_gimbal_0_1_fraction':float(((p[:,4].abs()-torch.pi/2).abs()<.1).float().mean()),'body_roundtrip_param_abs':stat((br-b).abs()),'body_roundtrip_samples_gt_1e_4':int(((br-b).abs().amax(1)>1e-4).sum()),'scale_projection_abs':stat((scales-p[:,136:]).abs()),'root_roundtrip_rig_matrix_abs':stat((round_rig-root_rig).abs()),'root_naive_parameter_abs':stat((naive_e-p[:,3:6]).abs())}
 # Hand roundtrip through the distinct, joint-ordered conversion.
 hr=p.clone()
 for ids in [s['head_pose.hand_joint_idxs_left'].long(),s['head_pose.hand_joint_idxs_right'].long()]:
  hr[:,ids]=u.compact_cont_to_model_params_hand(u.compact_model_params_to_cont_hand(p[:,ids]))
 d['hand_roundtrip_param_abs']=stat((hr-p).abs())
 # Full FK probes on a fixed subset; cm -> mm. Keep runtime and memory bounded.
 errs={k:[] for k in ['body_roundtrip_skel_mm','scale_projection_skel_mm','scale_projection_pve_mm','hand_only_nonhand_kp_mm','skeleton_kp_vs_exact_mm','skeleton_body_kp_vs_exact_mm','exact_kp_vs_annotation_mm','skeleton_kp_vs_annotation_mm','pelvis_midhip_mm','root_canonical_skel_mm']}
 for start in range(0,min(256,len(p)),16):
  pp=p[start:start+16]; hh=h[start:start+16]
  with torch.no_grad():
   vv,ss=rig(hh,pp,torch.zeros(len(pp),72)); ss=ss[...,:3]
   kp=K@torch.cat([vv,ss],1)
   kp_s=W@ss
   errs['skeleton_kp_vs_exact_mm'].append((kp-kp_s).norm(dim=-1)*10)
   errs['skeleton_body_kp_vs_exact_mm'].append((kp[:,body_idx]-kp_s[:,body_idx]).norm(dim=-1)*10)
   sign=torch.tensor([1.,-1.,-1.])
   errs['exact_kp_vs_annotation_mm'].append((kp*sign/100-j[start:start+16]).norm(dim=-1)*1000)
   errs['skeleton_kp_vs_annotation_mm'].append((kp_s*sign/100-j[start:start+16]).norm(dim=-1)*1000)
   errs['pelvis_midhip_mm'].append(((kp[:,9]+kp[:,10])/2).norm(dim=-1)*10)
   errs['body_roundtrip_skel_mm'].append((skel(pr[start:start+16],hh)-ss).norm(dim=-1)*10)
   pc=pp.clone(); pc[:,136:]=scales[start:start+16].float()
   vc,sc=rig(hh,pc,torch.zeros(len(pp),72))
   errs['scale_projection_skel_mm'].append((sc[...,:3]-ss).norm(dim=-1)*10)
   errs['scale_projection_pve_mm'].append((vc-vv).norm(dim=-1)*10)
   ph=pp.clone(); ph[:,hand_idx]=0
   sh=skel(ph,hh)
   errs['hand_only_nonhand_kp_mm'].append(((W@sh)[:,body_idx]-kp_s[:,body_idx]).norm(dim=-1)*10)
   proot=pp.clone(); proot[:,3:6]=root_e[start:start+16]
   errs['root_canonical_skel_mm'].append((skel(proot,hh)-ss).norm(dim=-1)*10)
 d['geometry_n']=min(256,len(p))
 d.update({k:stat(torch.cat(v).flatten()) for k,v in errs.items()})
 # Visible hand extents within the stored 224px body crop, without claiming visibility labels exist.
 ext=[]
 for ids in [list(range(21,42)),list(range(42,63))]:
  xy=j2[:,ids]; side=(xy.amax(1)-xy.amin(1)).amax(1)
  ext.append(side/(bbox[:,2]-bbox[:,0])*224)
 d['hand_span_in_224_crop_px']=stat(torch.cat(ext))
 imgs=[]
 for path in chosen[:16]:
  im=root/f'data/sam3d_gt_{split}/images/{path.stem}.png'
  if not im.exists(): im=im.with_suffix('.jpg')
  if im.exists():
   with Image.open(im) as image: imgs.append(list(image.size))
 d['sampled_image_sizes']=sorted({tuple(i) for i in imgs})
 out['datasets'][split]=d
 print(split,json.dumps(d),flush=True)
Path(__file__).with_name('measurements.json').write_text(json.dumps(out,indent=2)+'\n')
