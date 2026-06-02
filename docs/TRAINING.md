# Training

Training DCR has two stages:

1. **Robust Start** (source-only warm-up)
2. **DCR Adaptation** (domain adaptation with SPC and DGCR)

This assumes you already finished installation and dataset registration (see [`./README.md`](../README.md) and [`./docs/DATASETS.md`](./DATASETS.md)).

## Quick start (example: C2F + RegionCLIP)

We already provide runnable configs in [`./configs`](../configs) and convenient scripts in [`./train_scrips`](../train_scrips).  
For example, Cityscapes -> Foggy Cityscapes with RegionCLIP:

```bash
# Stage 1: Robust Start
bash ./train_scrips/cityscapes/regionclip/robust_start.sh

# Stage 2: DCR Adaptation
bash ./train_scrips/cityscapes/regionclip/dcr.sh
```

## Common config edits

We already provide out-of-the-box configs and scripts. In practice, this section is for common edits (especially if you use your own dataset).

### 1. Switch datasets

Example file: [`./configs/cityscapes/Base-RCNN-FPN-Cityscapes_robust_start.yaml`](../configs/cityscapes/Base-RCNN-FPN-Cityscapes_robust_start.yaml)

```yaml
DATASETS:
  TRAIN: ("cityscapes_train",)
  TEST: ("cityscapes_val", "cityscapes_foggy_val",)
  BATCH_CONTENTS: ("labeled_strong", )
```

Change dataset names here, and keep them consistent with registrations in [`./dcr/datasets.py`](../dcr/datasets.py).

### 2. Change number of classes

Edit:

```yaml
MODEL:
  ROI_HEADS:
    NUM_CLASSES: 8
```

For example, S2C uses `NUM_CLASSES: 1` (car-only).

### 3. Switch to RegionCLIP backbone/init

In RegionCLIP configs (for example [`./configs/cityscapes/Base-RCNN-FPN-Cityscapes_robust_start-RegionCLIP.yaml`](../configs/cityscapes/Base-RCNN-FPN-Cityscapes_robust_start-RegionCLIP.yaml)):

```yaml
MODEL:
  WEIGHTS: "./models/regionclip_finetuned-coco_rn50_fpn.pth"
  BACKBONE:
    NAME: "build_clip_resnet_fpn_backbone"
```

The `regionclip_finetuned-coco_rn50_fpn.pth` weight is converted by us. [`download link`](https://github.com/Gakia457/DAOD-via-Dynamic-Causal-Refinement/releases/download/RegionCLIP-init/regionclip_finetuned-coco_rn50_fpn.pth).

## Common knobs

You can edit the `.sh` scripts according to your hardware and training budget, especially:

- `CUDA_VISIBLE_DEVICES`
- `--num-gpus`
- `SOLVER.BASE_LR`
- `SOLVER.IMS_PER_BATCH`

For DGCR spectral perturbation (SP), use:

```bash
AUG.SP_PROB 0.5 \
AUG.SP_DYNAMIC_ADJUST True \
```

- `AUG.SP_PROB` controls SP application probability.
- `AUG.SP_DYNAMIC_ADJUST` controls dynamic SP parameter adjustment (discrepancy guided).

For PDC/SCC switches, see `DOMAIN_ADAPT.DISTILL` in adaptation configs (for example [`./configs/cityscapes/DCR-Cityscapes-to-FoggyCityscapes.yaml`](../configs/cityscapes/DCR-Cityscapes-to-FoggyCityscapes.yaml)):

- PDC-related items: `{STANDARD, DISTRIBUTION}_*_ENABLED`
- SCC-related item: `FEATURE_DISTILL_{ENABLED, WEIGHT, METHOD}`

For more detailed options, please refer to [`./dcr/config.py`](../dcr/config.py).
