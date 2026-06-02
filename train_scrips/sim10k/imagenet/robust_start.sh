CUDA_VISIBLE_DEVICES=0,1 python tools/train_net.py \
--config ./configs/sim10k/Base-RCNN-FPN-Sim10k_robust_start.yaml \
--num-gpus 2 \
--dist-url tcp://127.0.0.1:50155 \
SOLVER.BASE_LR 0.02 \
SOLVER.IMS_PER_BATCH 32 \
OUTPUT_DIR ./output/robust_start/sim10k/imagenet_fpn

