#!/bin/bash
#SBATCH --job-name=setup_castle_env
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd /scratch/$USER/castle_poc
mkdir -p logs

echo "HOSTNAME=$(hostname)"
echo "PWD=$(pwd)"
echo "SYSTEM PYTHON=$(command -v python3 || command -v python)"
python3 --version || python --version

rm -rf venv_compute

PYBIN=$(command -v python3 || command -v python)
"$PYBIN" -m venv --copies venv_compute

source /scratch/$USER/castle_poc/venv_compute/bin/activate
echo "VENV PYTHON=$(which python)"
python --version

python -m ensurepip --upgrade || true
python -m pip install --upgrade pip wheel setuptools
python -m pip install --no-cache-dir \
  huggingface_hub pandas pillow numpy tqdm imageio-ffmpeg requests

mkdir -p /scratch/$USER/castle_poc/bin
FFMPEG_PATH=$(python - <<'PY'
import imageio_ffmpeg
print(imageio_ffmpeg.get_ffmpeg_exe())
PY
)
ln -sf "$FFMPEG_PATH" /scratch/$USER/castle_poc/bin/ffmpeg
chmod +x /scratch/$USER/castle_poc/bin/ffmpeg

export PATH="/scratch/$USER/castle_poc/bin:$PATH"

python - <<'PY'
from huggingface_hub import hf_hub_url
import pandas, PIL, numpy, tqdm, imageio_ffmpeg, requests
print("dependency check ok")
print("ffmpeg:", imageio_ffmpeg.get_ffmpeg_exe())
PY
)
ln -sf "$FFMPEG_PATH" /scratch/$USER/castle_poc/bin/ffmpeg
chmod +x /scratch/$USER/castle_poc/bin/ffmpeg

export PATH="/scratch/$USER/castle_poc/bin:$PATH"

python - <<'PY'
from huggingface_hub import hf_hub_url
import pandas, PIL, numpy, tqdm, imageio_ffmpeg, requests
print("dependency check ok")
print("ffmpeg:", imageio_ffmpeg.get_ffmpeg_exe())
PY

which ffmpeg
ffmpeg -version | head -n 2

echo "SETUP_DONE"
