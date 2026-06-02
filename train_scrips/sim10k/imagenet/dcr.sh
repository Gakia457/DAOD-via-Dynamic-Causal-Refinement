CUDA_VISIBLE_DEVICES=0,1 python tools/train_net.py \
--config ./configs/sim10k/DCR-Sim10k-to-CityscapesCars.yaml \
--num-gpus 2 \
--dist-url tcp://127.0.0.1:50155 \
MODEL.WEIGHTS ./output/robust_start/sim10k/imagenet_fpn/cityscapes_cars_val_model_best.pth \
SOLVER.BASE_LR 0.02 \
SOLVER.IMS_PER_BATCH 32 \
AUG.SP_PROB 0.5 \
AUG.SP_DYNAMIC_ADJUST True \
OUTPUT_DIR ./output/dcr/sim10k_to_cityscapes_cars/imagenet_fpn/

