"""Model builder that composes DCR mixins with a Detectron2 meta-architecture."""

import torch
from typing import Dict, List

from detectron2.config import configurable
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.utils.logger import _log_api_usage

from dcr.align import ALIGN_MIXIN_REGISTRY
from dcr.distill import DISTILL_MIXIN_REGISTRY


def build_dcr(cfg):
    """
    Build the DCR meta-architecture by composing a base Detectron2 model with:
      - a distillation mixin (required): implements the distillation-based consistency in our paper;
      - an alignment mixin (optional): can be enabled as an extra component when needed.
    """
    base_cls = META_ARCH_REGISTRY.get(cfg.MODEL.META_ARCHITECTURE)
    align_mixin = ALIGN_MIXIN_REGISTRY.get(cfg.DOMAIN_ADAPT.ALIGN.MIXIN_NAME)
    distill_mixin = DISTILL_MIXIN_REGISTRY.get(cfg.DOMAIN_ADAPT.DISTILL.MIXIN_NAME)

    class DCR(align_mixin, distill_mixin, base_cls):
        @configurable
        def __init__(self, **kwargs):
            super(DCR, self).__init__(**kwargs)

        @classmethod
        def from_config(cls, cfg):
            return super(DCR, cls).from_config(cfg)

        def forward(self, batched_inputs: List[Dict[str, torch.Tensor]], 
                    labeled: bool = True, do_align: bool = False):
            return super(DCR, self).forward(batched_inputs, do_align=do_align, labeled=labeled)
        
    model = DCR(cfg)
    model.to(torch.device(cfg.MODEL.DEVICE))
    _log_api_usage("modeling.meta_arch." + cfg.MODEL.META_ARCHITECTURE)
    return model
