set -e

PYTHONPATH="${PROJECT_ROOT}" \
OMP_NUM_THREADS=1 \
CUDA_VISIBLE_DEVICES=0 \
python "${PROJECT_ROOT}"/src/ensemble.py "/dumps/predictions/fold_*.pkl" --output=/dumps/predictions
