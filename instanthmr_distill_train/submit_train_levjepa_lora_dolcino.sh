#!/bin/sh -e

echo "example: $0 --gpu 1"

# if `requirements.txt` is present, virtual environment is installed and used
EXEC="python3 train_levjepa_lora.py"

GPU_REQUIRED=true

. /pfcalcul/tools/sbatchHelpers5.sh

# --data_root points at the dataset synced by datasynch_perso (see below).
#
# batch_size is in CLIPS, not frames: 4 clips x 16 frames = 64 frames/step, and
# with --accum_steps 4 that is 256 frames per optimiser update. LoRA cannot use
# a feature cache (the encoder moves every step), so every clip is re-encoded
# every epoch -- this job is ~30x the wall-clock of the frozen probe on the same
# data. Raise --batch_size if the GPU has the memory; grad checkpointing is on
# by default and roughly halves activation memory for ~30% more compute.
#
# --max_hours 92 stops cleanly and checkpoints before SLURM's 4-day SIGKILL.
EXEC="python3 -u train_levjepa_lora.py --data_root /datasets/instanthmr_data --num_workers 8 --batch_size 4 --accum_steps 4 --max_hours 92 $useropt"

# SBATCH --output/--error paths are relative to the submission directory, so
# create the log folder before submitting.
mkdir -p instanthmr_levjepa_lora

jobMessage=$(
######################### SBATCH launcher
######################### #SBATCH --option # <= this is an enabled paramater
######################### ##SBATCH --option # <= this is a disabled parameter
sbatch ${sbatchopt} << eof
#!/bin/bash
#SBATCH --job-name="instanthmr_levjepa_lora"
#SBATCH --mail-type=ALL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=instanthmr_levjepa_lora/stdout_levjepa_dolcino.txt
#SBATCH --error=instanthmr_levjepa_lora/stderr_levjepa_dolcino.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=50000
#SBATCH --time=4-00:00:00

. /pfcalcul/tools/sbatchHelpers5.sh

/pfcalcul/tools/datasync /datasets/instanthmr_data


time $EXEC

eof
)

showSubmitted
