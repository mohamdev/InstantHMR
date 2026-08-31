#!/bin/sh -e

echo "example: $0 --gpu 1"

# if `requirements.txt` is present, virtual environment is installed and used
EXEC="python3 train_levjepa_cloud.py"

GPU_REQUIRED=true

. /pfcalcul/tools/sbatchHelpers5.sh

# ---------------------------------------------------------------------------
# BEFORE THE FIRST SUBMIT, stage the encoder ONCE from a machine with internet:
#
#   python3 -c "from huggingface_hub import snapshot_download; \
#       print(snapshot_download('galilai-group/LeVJEPA-VideoMix-Large'))"
#   rsync -avPL <printed path>/ <cluster>:<ENCODER_DIR>/
#
# -L (--copy-links) is MANDATORY. A HuggingFace snapshot dir is nothing but
# symlinks into ../../blobs/, so plain `rsync -av` copies six dangling links
# (~650 bytes) instead of the 1.2 GB of weights, and the job then fails to load
# the model. The trailing slash on the source flattens the snapshot into
# ENCODER_DIR rather than nesting it under a commit-hash folder.
#
# Compute nodes generally cannot reach huggingface.co, so --encoder_path points
# at that staged copy and the loader runs with local_files_only=True. Without
# it the job dies on the first AutoModel.from_pretrained call.
#
# requirements.txt must contain `transformers` + `safetensors` (the weights ship
# their own modeling code, hence trust_remote_code=True). A venv built before
# that line was added will NOT have it -- delete venv/ so it is rebuilt, or the
# job fails with "ModuleNotFoundError: No module named 'transformers'".
# ---------------------------------------------------------------------------
# NOTE the LEADING SLASH: without it this is a relative path and resolves
# against the submission directory, so from_pretrained gets a path that does
# not exist. Point it at the flattened snapshot (config.json + *.py +
# model.safetensors directly inside), not at a nested commit-hash folder.
ENCODER_DIR="/pfcalcul/work/kchalabi/envs/lstm/instanthmr_distill_train/levjepa_encoder"

# batch_size is in CLIPS (16 frames each). Stills are encoded as T=1 -- 197
# tokens instead of 3137 -- so they run at static_batch_mult x that batch.
# The encoder is FROZEN: no backward through 305 M params, no LoRA state.
# --max_hours 92 checkpoints and stops cleanly before SLURM's 4-day SIGKILL.
EXEC="python3 -u train_levjepa_cloud.py --data_root /datasets/instanthmr_data --encoder_path $ENCODER_DIR --num_workers 8 --batch_size 8 --static_batch_mult 8 --epochs 30 --max_hours 92 $useropt"

# SBATCH --output/--error paths are relative to the submission directory, so
# create the log folder before submitting.
mkdir -p instanthmr_levjepa_cloud

jobMessage=$(
######################### SBATCH launcher
######################### #SBATCH --option # <= this is an enabled paramater
######################### ##SBATCH --option # <= this is a disabled parameter
sbatch ${sbatchopt} << eof
#!/bin/bash
#SBATCH --job-name="instanthmr_levjepa_cloud"
#SBATCH --mail-type=ALL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=instanthmr_levjepa_cloud/stdout_perelha.txt
#SBATCH --error=instanthmr_levjepa_cloud/stderr_perelha.txt
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
