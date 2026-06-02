CUDA_VISIBLE_DEVICES=0,1 python tools/train_net.py \
--config ./configs/bdd100k/Base-RCNN-FPN-BDD100K_robust_start-RegionCLIP.yaml \
--num-gpus 2 \
--dist-url tcp://127.0.0.1:50155 \
SOLVER.BASE_LR 0.02 \
SOLVER.IMS_PER_BATCH 32 \
AUG.SP_PROB 0.5 \
OUTPUT_DIR ./output/robust_start/bdd100k/regionclip_fpn

