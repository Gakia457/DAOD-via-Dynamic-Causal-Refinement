"""Configuration defaults for DCR training, distillation, and augmentation."""

from detectron2.config import CfgNode as CN


def add_dcr_config(cfg):
    _C = cfg

    # Datasets and sampling
    _C.DATASETS.UNLABELED = tuple()
    _C.DATASETS.BATCH_CONTENTS = ("labeled_weak",)  # one or more of: {"labeled_weak", "labeled_strong", "unlabeled_weak", "unlabeled_strong"}
    _C.DATASETS.BATCH_RATIOS = (1,)  # must match length of BATCH_CONTENTS

    # Strong augmentations
    _C.AUG = CN()
    _C.AUG.WEAK_INCLUDES_MULTISCALE = True
    _C.AUG.LABELED_INCLUDE_RANDOM_ERASING = True
    _C.AUG.UNLABELED_INCLUDE_RANDOM_ERASING = True
    _C.AUG.LABELED_MIC_AUG = False
    _C.AUG.UNLABELED_MIC_AUG = False
    _C.AUG.MIC_RATIO = 0.5
    _C.AUG.MIC_BLOCK_SIZE = 32
    # SP augmentation
    _C.AUG.SP_PROB = 0.0  # disabled by default
    _C.AUG.SP_V1_SCALE = 0.005
    _C.AUG.SP_V2_SCALE = 0.7
    _C.AUG.SP_DYNAMIC_ADJUST = False  # enable SP dynamic parameter updates
    _C.AUG.SP_EMA_DECAY = 0.996  
    _C.AUG.SP_SOURCE_ONLY = True  
    _C.AUG.SP_EXCLUDE_FEATURE_FB = False  
    _C.AUG.SP_RANDOM_ADJUST = False  # random adjustment mode for ablation

    # EMA of student weights
    _C.EMA = CN()
    _C.EMA.ENABLED = False
    _C.EMA.ALPHA = 0.9996
    # When loading at the start of training (not resume), if MODEL.WEIGHTS
    # contains both ["model", "ema"], initialize with EMA weights. This also
    # controls whether EMA is used during eval-only runs.
    _C.EMA.LOAD_FROM_EMA_ON_START = True
    _C.EMA.START_ITER = 0

    # Begin domain adaptation settings
    _C.DOMAIN_ADAPT = CN()

    # Source-target alignment
    _C.DOMAIN_ADAPT.ALIGN = CN()
    _C.DOMAIN_ADAPT.ALIGN.MIXIN_NAME = "AlignMixin"
    _C.DOMAIN_ADAPT.ALIGN.IMG_DA_ENABLED = False
    _C.DOMAIN_ADAPT.ALIGN.IMG_DA_LAYER = "p2"
    _C.DOMAIN_ADAPT.ALIGN.IMG_DA_WEIGHT = 0.01
    _C.DOMAIN_ADAPT.ALIGN.IMG_DA_INPUT_DIM = 256  # output channels of backbone
    _C.DOMAIN_ADAPT.ALIGN.IMG_DA_HIDDEN_DIMS = [256, ]
    _C.DOMAIN_ADAPT.ALIGN.INS_DA_ENABLED = False
    _C.DOMAIN_ADAPT.ALIGN.INS_DA_WEIGHT = 0.01
    _C.DOMAIN_ADAPT.ALIGN.INS_DA_INPUT_DIM = 1024  # output channels of box head
    _C.DOMAIN_ADAPT.ALIGN.INS_DA_HIDDEN_DIMS = [1024, ]

    # Self-distillation
    _C.DOMAIN_ADAPT.DISTILL = CN()
    _C.DOMAIN_ADAPT.DISTILL.DISTILLER_NAME = "DCRDistiller"
    _C.DOMAIN_ADAPT.DISTILL.MIXIN_NAME = "DistillMixin"
    # Standard distillation
    _C.DOMAIN_ADAPT.DISTILL.STANDARD_ROIH_CLS_ENABLED = False
    _C.DOMAIN_ADAPT.DISTILL.STANDARD_ROIH_REG_ENABLED = False
    _C.DOMAIN_ADAPT.DISTILL.STANDARD_OBJ_ENABLED = False
    _C.DOMAIN_ADAPT.DISTILL.STANDARD_RPN_REG_ENABLED = False
    # Distribution distillation (PDC)
    _C.DOMAIN_ADAPT.DISTILL.DISTRIBUTION_ROIH_CLS_ENABLED = False
    _C.DOMAIN_ADAPT.DISTILL.DISTRIBUTION_ROIH_REG_ENABLED = False
    _C.DOMAIN_ADAPT.DISTILL.DISTRIBUTION_OBJ_ENABLED = False
    _C.DOMAIN_ADAPT.DISTILL.DISTRIBUTION_RPN_REG_ENABLED = False
    _C.DOMAIN_ADAPT.DISTILL.CLS_TMP = 1.0
    _C.DOMAIN_ADAPT.DISTILL.OBJ_TMP = 1.0
    _C.DOMAIN_ADAPT.CLS_LOSS_TYPE = "CE"  # one of: {"CE", "KL"}
    # Feature distillation (SCC)
    _C.DOMAIN_ADAPT.DISTILL.FEATURE_DISTILL_ENABLED = False
    _C.DOMAIN_ADAPT.DISTILL.FEATURE_DISTILL_WEIGHT = 1.0
    _C.DOMAIN_ADAPT.DISTILL.FEATURE_DISTILL_METHOD = "attn"  # options: "gap", "pointwise", "attn"

    # Teacher model settings for pseudo labels
    _C.DOMAIN_ADAPT.TEACHER = CN()
    _C.DOMAIN_ADAPT.TEACHER.ENABLED = False
    _C.DOMAIN_ADAPT.TEACHER.THRESHOLD = 0.8
    _C.DOMAIN_ADAPT.TEACHER.PSEUDO_LABEL_METHOD = "thresholding"

    # Vision Transformer settings
    _C.VIT = CN()
    _C.VIT.USE_ACT_CHECKPOINT = True

    # We interpret SOLVER.IMS_PER_BATCH as total batch size across all GPUs for
    # experimental consistency. Gradient accumulation follows:
    # num_gradient_accum_steps = IMS_PER_BATCH / (NUM_GPUS * IMS_PER_GPU)
    _C.SOLVER.IMS_PER_GPU = 2

    # We use gradient accumulation to run the weak/strong/unlabeled data separately
    # Should we call backward intermittently during accumulation or at the end?
    # The former is slower but less memory usage
    _C.SOLVER.BACKWARD_AT_END = True

    # Enable use of different optimizers (necessary to match VitDet settings)
    _C.SOLVER.OPTIMIZER = "SGD"

    # Extra configs for ConvNeXt
    # Default is ConvNeXt-T (ResNet-50 equivalent).
    _C.MODEL.CONVNEXT = CN()
    _C.MODEL.CONVNEXT.DEPTHS= [3, 3, 9, 3]
    _C.MODEL.CONVNEXT.DIMS= [96, 192, 384, 768]
    _C.MODEL.CONVNEXT.DROP_PATH_RATE= 0.2
    _C.MODEL.CONVNEXT.LAYER_SCALE_INIT_VALUE= 1e-6
    _C.MODEL.CONVNEXT.OUT_FEATURES= [0, 1, 2, 3]
    _C.SOLVER.WEIGHT_DECAY_RATE= 0.95
