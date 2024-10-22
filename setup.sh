
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

source $SCRIPT_DIR/venv/bin/activate


export TF_CPP_MIN_LOG_LEVEL=2
export MKL_SERVICE_FORCE_INTEL=1
export MUJOCO_GL=egl
export HYDRA_FULL_ERROR=1
export PYTHONHASHSEED=0
export WANDB__SERVICE_WAIT=300
export DISPLAY=:0
