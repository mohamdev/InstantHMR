#!/bin/sh -e

echo "example: $0 --gpu 1"

# if `requirements.txt` is present, virtual environment is installed and used
EXEC="python3 train_distill_optimized_correctives_fine_tune.py"

GPU_REQUIRED=true

. /pfcalcul/tools/sbatchHelpers5.sh

# --data_root points at the dataset synced by datasynch_perso (see below).
# It is synced next to this project folder, i.e. ../instanthmr_data once we
# cd into instanthmr_distill_train.
#
# Fine-tune to repair the 2D keypoint head after the flip-label fix
# (joints_2d[FLIP_PERM] was missing from the geometric augmentation).
# --init_ckpt is the checkpoint we START FROM; results land in --output_dir.
# --reset_2d_head is important: the existing 2D head encodes the OLD left/right
# convention, so fine-tuning it in place has to unlearn that on ~40% of samples
# and thrashes. Resetting it makes the head learn the corrected convention from
# scratch on frozen-quality features, which converges cleanly.
# trunk lr 1e-5 / head lr 3e-4; --pa_tolerance rejects any checkpoint whose
# Mesh PA-MPJPE regresses more than 2 mm from the measured baseline.
EXEC="python3 -u train_distill_optimized_correctives_fine_tune.py \
  --data_root /datasets/instanthmr_data \
  --init_ckpt runs/distill_optimized_v2/best_student_model_ema.pth \
  --output_dir runs/distill_optimized_v2_ft \
  --num_workers 8 --batch_size 128 \
  --reset_2d_head \
  --epochs 15 --lr 1e-5 --head_2d_lr_mult 30 \
  --pa_tolerance 2.0 --early_stop_patience 4 $useropt"

# Zero-risk fallback if the PA guard blocks everything: freezes all but head_2d_*,
# so the 3D pose provably cannot change (measured: PA held to 0.0 mm over 8 epochs).
# Slower to converge — the decoder cannot adapt with it.
#   ... --train_2d_only --reset_2d_head --lr 3e-4 --head_2d_lr_mult 1 --epochs 30

# SBATCH --output/--error paths are relative to the submission directory, so
# create the log folder before submitting.
mkdir -p instanthmr_distill

jobMessage=$(
######################### SBATCH launcher
######################### #SBATCH --option # <= this is an enabled paramater
######################### ##SBATCH --option # <= this is a disabled parameter
sbatch ${sbatchopt} << eof
#!/bin/bash
#SBATCH --job-name="instanthmr_finetune"
#SBATCH --mail-type=ALL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=instanthmr_distill/stdout_finetune_dolcino.txt
#SBATCH --error=instanthmr_distill/stderr_finetune_dolcino.txt
#SBATCH --account=dept_rob
#SBATCH --partition=robgpu
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=50000
#SBATCH --time=2-00:00:00

. /pfcalcul/tools/sbatchHelpers5.sh

/pfcalcul/tools/datasync /datasets/instanthmr_data


time $EXEC

eof
)

showSubmitted
