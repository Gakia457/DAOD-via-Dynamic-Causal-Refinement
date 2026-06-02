CUDA_VISIBLE_DEVICES=0,1 python tools/train_net.py \
--config ./configs/cityscapes/DCR-Cityscapes-to-FoggyCityscapes-RegionCLIP.yaml \
--num-gpus 2 \
--dist-url tcp://127.0.0.1:50155 \
MODEL.WEIGHTS ./output/robust_start/cityscapes/regionclip_fpn/cityscapes_foggy_val_model_best.pth \
SOLVER.BASE_LR 0.02 \
SOLVER.IMS_PER_BATCH 32 \
AUG.SP_PROB 0.5 \
AUG.SP_DYNAMIC_ADJUST True \
OUTPUT_DIR ./output/dcr/cityscapes_to_foggy_cityscapes/regionclip_fpn

# Tip: You can start by training with spc-only. 
# After a certain number of epochs, enable both modules (spc + dgcr) to continue training.

# CUDA_VISIBLE_DEVICES=0,1 python tools/train_net.py \
# --config ./configs/cityscapes/DCR-Cityscapes-to-FoggyCityscapes-RegionCLIP.yaml \
# --num-gpus 2 \
# --dist-url tcp://127.0.0.1:50155 \
# MODEL.WEIGHTS ./output/cityscapes/robust_start/regionclip_fpn_aug/cityscapes_foggy_val_model_best.pth \
# SOLVER.BASE_LR 0.02 \
# SOLVER.IMS_PER_BATCH 32 \
# OUTPUT_DIR ./output/C2F/regionclip_fpn/spc_only

# CUDA_VISIBLE_DEVICES=0,1 python tools/train_net.py \
# --config ./configs/cityscapes/DCR-Cityscapes-to-FoggyCityscapes-RegionCLIP.yaml \
# --num-gpus 2 \
# --dist-url tcp://127.0.0.1:50155 \
# MODEL.WEIGHTS ./output/C2F/regionclip_fpn/spc_only/model_00xx999.pth \
# SOLVER.BASE_LR 0.02 \
# SOLVER.IMS_PER_BATCH 32 \
# AUG.SP_PROB 0.5 \
# AUG.SP_DYNAMIC_ADJUST True \
# OUTPUT_DIR ./output/C2F/regionclip_fpn/spc+dgcr