set -e

PYTHONPATH="${PROJECT_ROOT}" \
OMP_NUM_THREADS=1 \
CUDA_VISIBLE_DEVICES=0 \
python "${PROJECT_ROOT}"/src/submit.py /dumps/predictions/ensemble.pkl \
    --output=/dumps/fold8_sub.csv \
    --sample_submission=/data/stage_2_sample_submission.csv
