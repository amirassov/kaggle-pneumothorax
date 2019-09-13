set -e

for FOLD in {0..7}
do
PYTHONPATH="${PROJECT_ROOT}" \
OMP_NUM_THREADS=1 \
CUDA_VISIBLE_DEVICES=0 \
python "${PROJECT_ROOT}"/src/test.py "${PROJECT_ROOT}"/configs/resnet34_768_unet.yaml \
    --fold=${FOLD} \
    --output=/dumps/predictions \
    --path=/dumps/checkpoints
wait
done