CUDA_VISIBLE_DEVICES=0,1 python tools/train_net.py \
--config ./configs/bdd100k/DCR-Cityscapes-to-BDD100K-RegionCLIP.yaml \
--num-gpus 2 \
--dist-url tcp://127.0.0.1:50155 \
MODEL.WEIGHTS ./output/robust_start/bdd100k/regionclip_fpn/bdd100k_val_daytime_model_best.pth \
SOLVER.BASE_LR 0.02 \
SOLVER.IMS_PER_BATCH 32 \
AUG.SP_PROB 0.5 \
AUG.SP_DYNAMIC_ADJUST True \
OUTPUT_DIR ./output/dcr/cityscapes_to_bdd100k/regionclip_fpn

