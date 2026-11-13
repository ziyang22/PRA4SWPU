#!/bin/bash
# 在项目根目录提交：sbatch scripts/baseline.slurm [evaluate|train|hip-check]
#SBATCH --partition=hx1hdnormal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=dcu:1
#SBATCH --mem=14G
#SBATCH --time=00:20:00
#SBATCH --job-name=swpu-baseline
#SBATCH --output=slurm-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?请通过sbatch提交作业}"
# 在加载工具链前锁定已激活环境的解释器，避免模块修改PATH。
PYTHON_BIN="$(command -v "${PRA4SWPU_PYTHON:-python3}")"
if [[ "$PYTHON_BIN" != /* ]]; then
    PYTHON_BIN="$PWD/$PYTHON_BIN"
fi
unset PYTHONHOME PYTHONPATH
"$PYTHON_BIN" -c 'import sys; print("Python:", sys.executable, sys.version); assert sys.version_info[:2] == (3, 11), "请激活Python 3.11环境，或设置PRA4SWPU_PYTHON为该环境解释器的绝对路径"'
module load compiler/gcc/12.2.0 compiler/dtk/25.04.2
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONDONTWRITEBYTECODE=1

# 工具链可能设置其他Python版本的库路径，保留已锁定的环境。
unset PYTHONHOME PYTHONPATH
ACTION="${1:-evaluate}"
RESULT_DIR="outputs/slurm-${SLURM_JOB_ID}"
mkdir -p "$RESULT_DIR"
"$PYTHON_BIN" -c 'import torch; assert torch.version.hip and torch.cuda.is_available(), "需要HIP版PyTorch及已分配的DCU"; print("PyTorch:", torch.__version__, "HIP:", torch.version.hip, "Device:", torch.cuda.get_device_name(0))'

case "$ACTION" in
    evaluate)
        srun "$PYTHON_BIN" src/main.py evaluate --checkpoint weights/baseline.pt \
            --data-dir data --backend torch --device cuda --output "$RESULT_DIR/evaluation"
        ;;
    train)
        srun "$PYTHON_BIN" src/main.py train --config configs/baseline.json \
            --device cuda --output-dir "$RESULT_DIR/training"
        ;;
    hip-check)
        export ROCM_HOME="$DTKROOT"
        export MAX_JOBS="$OMP_NUM_THREADS"
        export PYTORCH_ROCM_ARCH=$("$PYTHON_BIN" -c 'import torch; print(torch.cuda.get_device_properties(0).gcnArchName.split(":")[0])')
        srun "$PYTHON_BIN" scripts/check_hip.py --output "$RESULT_DIR/hip-check.json"
        ;;
    *)
        echo "用法：sbatch scripts/baseline.slurm [evaluate|train|hip-check]" >&2
        exit 2
        ;;
esac
echo "BASELINE_JOB=passed; action=$ACTION; results=$RESULT_DIR"
