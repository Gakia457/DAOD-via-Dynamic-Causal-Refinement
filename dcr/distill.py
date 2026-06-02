"""Distiller registry and implementations for standard, distribution, and feature distillation."""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from detectron2.config import configurable
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
from detectron2.layers import cat
from detectron2.layers.wrappers import cross_entropy
from detectron2.modeling.sampling import subsample_labels
from detectron2.modeling.box_regression import _dense_box_regression_loss
from detectron2.utils.registry import Registry
from fvcore.nn import smooth_l1_loss

from dcr.helpers import SaveIO, ManualSeed, ReplaceProposalsOnce, set_attributes
from dcr.pseudolabeler import PseudoLabeler

logger = logging.getLogger(__name__)

DISTILLER_REGISTRY = Registry("DISTILLER")
DISTILLER_REGISTRY.__doc__ = """
"""

DISTILL_MIXIN_REGISTRY = Registry("DISTILL_MIXIN")
DISTILL_MIXIN_REGISTRY.__doc__ = """
"""


def build_distiller(cfg, teacher, student):
    name = cfg.DOMAIN_ADAPT.DISTILL.DISTILLER_NAME
    return DISTILLER_REGISTRY.get(name).from_config(cfg, teacher, student)


@DISTILLER_REGISTRY.register()
class Distiller(nn.Module):
    """This Distiller does nothing."""
    def __init__(self, teacher, student):
        super().__init__()
        pass

    @classmethod
    def from_config(cls, cfg, teacher, student):
        return Distiller(teacher, student)

    def __call__(self, teacher_batched_inputs, student_batched_inputs):
        return {}
    
    def distill_enabled(self):
        return False
    
    
@DISTILLER_REGISTRY.register()
class StandardDistiller(Distiller):
    """Just do standard pseudo-label self-distillation; should work with any kind of detector."""
    def __init__(self, teacher, student, do_standard_cls=False, do_standard_obj=False, do_standard_rpn_reg=False,
                 do_standard_roi_reg=False, pseudo_label_threshold=0.8):
        set_attributes(self, locals())
        self.pseudo_labeler = PseudoLabeler(teacher, pseudo_label_threshold)

    @classmethod
    def from_config(cls, cfg, teacher, student):
        return StandardDistiller(teacher, student,
                        do_standard_cls=cfg.DOMAIN_ADAPT.DISTILL.STANDARD_ROIH_CLS_ENABLED,
                        do_standard_obj=cfg.DOMAIN_ADAPT.DISTILL.STANDARD_OBJ_ENABLED,
                        do_standard_rpn_reg=cfg.DOMAIN_ADAPT.DISTILL.STANDARD_RPN_REG_ENABLED,
                        do_standard_roi_reg=cfg.DOMAIN_ADAPT.DISTILL.STANDARD_ROIH_REG_ENABLED,
                        pseudo_label_threshold=cfg.DOMAIN_ADAPT.TEACHER.THRESHOLD)

    def __call__(self, teacher_batched_inputs, student_batched_inputs):
        self.pseudo_labeler(teacher_batched_inputs, student_batched_inputs)
        standard_distill_losses = self.student(student_batched_inputs)
        return standard_distill_losses

    def distill_enabled(self):
        return any([self.do_standard_cls, self.do_standard_obj, self.do_standard_rpn_reg, self.do_standard_roi_reg])


class FeatureDistillEngine(nn.Module):
    """Feature distillation module used by DCRDistiller."""

    def __init__(
        self,
        do_feature_distill=False,
        feature_distill_weight=1.0,
        feature_distill_method="attn",
        feature_distill_channel=256,
        is_c4_backbone=False,
        device=None,
    ):
        super().__init__()
        self.do_feature_distill = do_feature_distill
        self.feature_distill_weight = feature_distill_weight
        self.feature_distill_method = feature_distill_method
        self.is_c4_backbone = is_c4_backbone
        self.device = device
        self._init_feature_distill_modules(feature_distill_channel)

    def _init_feature_distill_modules(self, feature_distill_channel):
        self.cross_attentions = nn.ModuleDict()
        if not self.do_feature_distill:
            return

        if self.is_c4_backbone:
            self.cross_attentions["res4"] = nn.MultiheadAttention(
                embed_dim=1024,
                num_heads=16,
                batch_first=True,
            ).to(self.device)
            return

        embed_dim = feature_distill_channel
        num_heads = 8 if embed_dim % 8 == 0 else 4
        for layer_name in ("p4", "p5"):
            self.cross_attentions[layer_name] = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                batch_first=True,
            ).to(self.device)

    @staticmethod
    def _to_feature_dict(features, default_key):
        if isinstance(features, dict):
            return features
        return {default_key: features}

    @staticmethod
    def _extract_c4_feature(features):
        if isinstance(features, dict):
            return features.get("res4", features)
        return features

    def _compute_c4_feature_distill_loss(self, teacher_features, student_features):
        teacher_feat = self._extract_c4_feature(teacher_features)
        student_feat = self._extract_c4_feature(student_features)
        loss_res4 = self._compute_layer_loss(
            teacher_feat,
            student_feat,
            layer_name="res4",
            temperature=1.0,
            margin=0.15,
        )
        return {"loss_feature_attn_res4": self.feature_distill_weight * loss_res4}

    def _compute_fpn_feature_distill_loss(self, teacher_features, student_features):
        teacher_feat = self._to_feature_dict(teacher_features, default_key="p5")
        student_feat = self._to_feature_dict(student_features, default_key="p5")

        if self.feature_distill_method == "gap":
            return self._get_gap_loss(teacher_feat["p5"], student_feat["p5"])
        if self.feature_distill_method == "pointwise":
            return self._get_pointwise_loss(teacher_feat["p5"], student_feat["p5"])
        if self.feature_distill_method == "attn":
            return self._get_attention_loss(teacher_feat, student_feat)
        raise ValueError(f"Unsupported feature_distill_method: {self.feature_distill_method}")

    def get_feature_distill_loss(self, teacher_features, student_features):
        if self.is_c4_backbone:
            return self._compute_c4_feature_distill_loss(teacher_features, student_features)
        return self._compute_fpn_feature_distill_loss(teacher_features, student_features)

    def _get_attention_loss(self, teacher_feat, student_feat):
        if isinstance(teacher_feat, dict):
            teacher_p4 = teacher_feat.get("p4", None)
            teacher_p5 = teacher_feat.get("p5", None)
        else:
            teacher_p4, teacher_p5 = None, teacher_feat

        if isinstance(student_feat, dict):
            student_p4 = student_feat.get("p4", None)
            student_p5 = student_feat.get("p5", None)
        else:
            student_p4, student_p5 = None, student_feat

        loss_dict = {}

        if teacher_p4 is not None and student_p4 is not None:
            loss_p4 = self._compute_layer_loss(
                teacher_p4,
                student_p4,
                layer_name="p4",
                temperature=1.75,
                margin=0.15,
            )
            loss_dict["loss_feature_attn_p4"] = self.feature_distill_weight * 0.7 * loss_p4

        if teacher_p5 is not None and student_p5 is not None:
            loss_p5 = self._compute_layer_loss(
                teacher_p5,
                student_p5,
                layer_name="p5",
                temperature=0.8,
                margin=0.15,
            )
            loss_dict["loss_feature_attn_p5"] = self.feature_distill_weight * 0.3 * loss_p5

        return loss_dict

    def _compute_layer_loss(self, teacher_feat, student_feat, layer_name="p5", temperature=1.0, margin=0.2):
        teacher_norm = F.instance_norm(teacher_feat)
        student_norm = F.instance_norm(student_feat)
        teacher_norm = teacher_norm.detach()

        if layer_name == "res4":
            scale = 0.125
        else:
            scale = 0.25 if layer_name == "p5" else 0.5
        teacher_norm = F.interpolate(teacher_norm, scale_factor=scale, mode="bilinear", align_corners=False)
        student_norm = F.interpolate(student_norm, scale_factor=scale, mode="bilinear", align_corners=False)

        batch_size, channels, h, w = teacher_norm.shape
        seq_len = h * w
        current_attention = self.cross_attentions[layer_name]

        teacher_flat = teacher_norm.view(batch_size, channels, seq_len).permute(0, 2, 1)
        student_flat = student_norm.view(batch_size, channels, seq_len).permute(0, 2, 1)

        attn_output, _ = current_attention(
            student_flat / temperature,
            teacher_flat / temperature,
            teacher_flat / temperature,
        )
        attn_output = attn_output + student_flat * 0.1

        student_pooled = torch.mean(attn_output, dim=1)
        teacher_pooled = torch.mean(teacher_flat, dim=1)
        student_pooled_norm = F.normalize(student_pooled, p=2, dim=1)
        teacher_pooled_norm = F.normalize(teacher_pooled, p=2, dim=1)
        cosine_sim = (student_pooled_norm * teacher_pooled_norm).sum(dim=1)
        return torch.clamp(1.0 - margin - cosine_sim, min=0.0).mean()

    def _get_mlp_loss(self, teacher_feat, student_feat): pass
    def _get_cosine_loss(self, teacher_feat, student_feat): pass
    def _get_gap_loss(self, teacher_feat, student_feat): pass
    def _get_pointwise_loss(self, teacher_feat, student_feat): pass

    def export_state_dict(self):
        state = {}
        for layer_name, module in self.cross_attentions.items():
            state[f"cross_attention_{layer_name}"] = module.state_dict()
        return state

    def load_feature_state_dict(self, state_dict):
        for key, param_dict in state_dict.items():
            if not key.startswith("cross_attention_"):
                continue

            layer_name = key.replace("cross_attention_", "")
            if layer_name not in self.cross_attentions and "in_proj_weight" in param_dict:
                embed_dim = param_dict["in_proj_weight"].shape[1]
                num_heads = 8 if embed_dim % 8 == 0 else 4
                logger.info("Reconstructing attention for %s (dim=%d)", layer_name, embed_dim)
                print("Reconstructing attention for %s (dim=%d)", layer_name, embed_dim)
                self.cross_attentions[layer_name] = nn.MultiheadAttention(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    batch_first=True,
                ).to(self.device)

            if layer_name in self.cross_attentions:
                self.cross_attentions[layer_name].load_state_dict(param_dict)


@DISTILLER_REGISTRY.register()
class DCRDistiller(Distiller):
    """Compute standard/distillation losses for Faster R-CNN based students/teachers.

    - distribution_distill corresponds to the PDC module
    - feature_distill corresponds to the SCC module
    """

    def __init__(self, teacher, student, do_standard_cls=False, do_standard_obj=False, do_standard_rpn_reg=False, do_standard_roi_reg=False,
                 do_distribution_cls=False, do_distribution_obj=False, do_distribution_rpn_reg=False, do_distribution_roih_reg=False,
                 cls_temperature=1.0, obj_temperature=1.0, cls_loss_type="CE", pseudo_label_threshold=0.8, pseudo_label_method="thresholding",
                 do_feature_distill=False, feature_distill_weight=1.0, feature_distill_method="attn", feature_distill_channel=256,
                 is_c4_backbone=False):
        super().__init__(teacher, student)
        set_attributes(self, locals())
        self.register_hooks()
        self.pseudo_labeler = PseudoLabeler(teacher, pseudo_label_threshold, method=pseudo_label_method)
        self.feature_distill_engine = FeatureDistillEngine(
            do_feature_distill=do_feature_distill,
            feature_distill_weight=feature_distill_weight,
            feature_distill_method=feature_distill_method,
            feature_distill_channel=feature_distill_channel,
            is_c4_backbone=is_c4_backbone,
            device=next(self.student.parameters()).device,
        )
        self.cross_attentions = self.feature_distill_engine.cross_attentions

    @classmethod
    def from_config(cls, cfg, teacher, student):
        return DCRDistiller(teacher, student,
                        do_standard_cls=cfg.DOMAIN_ADAPT.DISTILL.STANDARD_ROIH_CLS_ENABLED,
                        do_standard_obj=cfg.DOMAIN_ADAPT.DISTILL.STANDARD_OBJ_ENABLED,
                        do_standard_rpn_reg=cfg.DOMAIN_ADAPT.DISTILL.STANDARD_RPN_REG_ENABLED,
                        do_standard_roi_reg=cfg.DOMAIN_ADAPT.DISTILL.STANDARD_ROIH_REG_ENABLED,
                        do_distribution_cls=cfg.DOMAIN_ADAPT.DISTILL.DISTRIBUTION_ROIH_CLS_ENABLED,
                        do_distribution_obj=cfg.DOMAIN_ADAPT.DISTILL.DISTRIBUTION_OBJ_ENABLED,
                        do_distribution_rpn_reg=cfg.DOMAIN_ADAPT.DISTILL.DISTRIBUTION_RPN_REG_ENABLED,
                        do_distribution_roih_reg=cfg.DOMAIN_ADAPT.DISTILL.DISTRIBUTION_ROIH_REG_ENABLED,
                        cls_temperature=cfg.DOMAIN_ADAPT.DISTILL.CLS_TMP,
                        obj_temperature=cfg.DOMAIN_ADAPT.DISTILL.OBJ_TMP,
                        cls_loss_type=cfg.DOMAIN_ADAPT.CLS_LOSS_TYPE,
                        pseudo_label_threshold=cfg.DOMAIN_ADAPT.TEACHER.THRESHOLD,
                        pseudo_label_method=cfg.DOMAIN_ADAPT.TEACHER.PSEUDO_LABEL_METHOD,
                        do_feature_distill=cfg.DOMAIN_ADAPT.DISTILL.FEATURE_DISTILL_ENABLED,
                        feature_distill_weight=cfg.DOMAIN_ADAPT.DISTILL.FEATURE_DISTILL_WEIGHT,
                        feature_distill_method=cfg.DOMAIN_ADAPT.DISTILL.FEATURE_DISTILL_METHOD,
                        feature_distill_channel=cfg.MODEL.FPN.OUT_CHANNELS,
                        is_c4_backbone=cfg.MODEL.BACKBONE.NAME == "build_resnet_backbone")

    def register_hooks(self):
        self.student_rpn_io, self.student_rpn_head_io, self.student_boxpred_io = SaveIO(), SaveIO(), SaveIO()
        self.student_backbone_io = SaveIO()
        self.teacher_backbone_io, self.teacher_rpn_head_io, self.teacher_boxpred_io, self.teacher_anchor_io = SaveIO(), SaveIO(), SaveIO(), SaveIO()
        
        student_model = self.student.module if type(self.student) is DDP else self.student
        teacher_model = self.teacher.module if type(self.teacher) is DDP else self.teacher

        student_model.proposal_generator.register_forward_hook(self.student_rpn_io)
        student_model.proposal_generator.rpn_head.register_forward_hook(self.student_rpn_head_io)
        student_model.roi_heads.box_predictor.register_forward_hook(self.student_boxpred_io)
        student_model.backbone.register_forward_hook(self.student_backbone_io)

        teacher_model.backbone.register_forward_hook(self.teacher_backbone_io)
        teacher_model.proposal_generator.rpn_head.register_forward_hook(self.teacher_rpn_head_io)
        teacher_model.roi_heads.box_predictor.register_forward_hook(self.teacher_boxpred_io)
        teacher_model.proposal_generator.anchor_generator.register_forward_hook(self.teacher_anchor_io)

        # Make sure seeds are the same for proposal sampling in teacher/student
        self.seeder = ManualSeed()
        teacher_model.roi_heads.register_forward_pre_hook(self.seeder)
        student_model.roi_heads.register_forward_pre_hook(self.seeder)

        # Teacher and student second stage need to have the same input proposals in order to distill predictions on those proposals
        self.teacher_proposal_replacer = ReplaceProposalsOnce()
        teacher_model.roi_heads.register_forward_pre_hook(self.teacher_proposal_replacer)

    def distill_enabled(self):
        return any([self.do_standard_cls, self.do_standard_obj, self.do_standard_rpn_reg, self.do_standard_roi_reg,
                    self.do_distribution_cls, self.do_distribution_obj, self.do_distribution_rpn_reg, self.do_distribution_roih_reg,
                    self.do_feature_distill])

    def _distill_forward(self, teacher_batched_inputs, student_batched_inputs):
        # first, get standard pseudo labels -- this is done in place
        # even if not included in overall loss, we need them for RPN proposal sampling
        self.pseudo_labeler(teacher_batched_inputs, student_batched_inputs)
        
        self.seeder.reset_seed()

        # teacher might be in eval mode -- this is important for inputs/outputs aligning
        was_eval = not self.teacher.training
        if was_eval: 
            self.teacher.train()

        standard_distill_losses = self.student(student_batched_inputs)
        student_proposals, _ = self.student_rpn_io.output

        self.teacher_proposal_replacer.set_proposals(student_proposals)
        with torch.no_grad():
            self.teacher(teacher_batched_inputs)
        
        # return to eval mode if necessary
        if was_eval: 
            self.teacher.eval()

        return standard_distill_losses

    def __call__(self, teacher_batched_inputs, student_batched_inputs):
        losses = {}

        # Do a forward pass to get activations, and get standard pseudo-label losses if desired
        standard_distill_losses = self._distill_forward(teacher_batched_inputs, student_batched_inputs)
        loss_to_attr = {
            "loss_cls": self.do_standard_cls,
            "loss_rpn_cls": self.do_standard_obj,
            "loss_rpn_loc": self.do_standard_rpn_reg,
            "loss_box_reg": self.do_standard_roi_reg,
        }
        for k, v in standard_distill_losses.items():
            if loss_to_attr.get(k, False):
                losses[k] = v
            else:
                # Need to add to standard losses so that the optimizer can see it
                losses[k] = v * 0.0

        # distribution_distill (PDC)
        losses.update(self.get_rpn_losses(teacher_batched_inputs))
        losses.update(self.get_roih_losses())
        # feature_distill (SCC)
        if self.do_feature_distill:
            losses.update(self.get_feature_distill_loss())

        return losses

    def get_feature_distill_loss(self):
        """Compute feature distillation losses."""
        return self.feature_distill_engine.get_feature_distill_loss(
            teacher_features=self.teacher_backbone_io.output,
            student_features=self.student_backbone_io.output,
        )

    def state_dict(self):
        """Return trainable feature-distillation module states."""
        return self.feature_distill_engine.export_state_dict()

    def load_state_dict(self, state_dict):
        """Load trainable feature-distillation module states."""
        self.feature_distill_engine.load_feature_state_dict(state_dict)
    
    def get_rpn_losses(self, teacher_batched_inputs):
        losses = {}
        student_objectness_logits, student_proposal_deltas = self.student_rpn_head_io.output
        teacher_objectness_logits, teacher_proposal_deltas = self.teacher_rpn_head_io.output

        # the RPN samples proposals for loss computation *after* the RPN head
        # so we need to mimic this logic ourselves to match -- it's a bit complicated to reverse engineer
        rpn = (self.teacher.module if type(self.teacher) is DDP else self.teacher).proposal_generator
        pseudo_gt_labels = torch.stack(rpn.label_and_sample_anchors(self.teacher_anchor_io.output, 
                                                                       [i['instances'].to(self.teacher.device) for i in teacher_batched_inputs])[0])
        valid_mask = torch.flatten(pseudo_gt_labels >= 0) # the proposals we'll compute loss for
        fg_mask = pseudo_gt_labels == 1 # proposals matched to a pseudo GT box

        # Postprocessing -- for now just sharpening
        teacher_objectness_probs = torch.sigmoid(cat([torch.flatten(t) for t in teacher_objectness_logits]) / self.obj_temperature)

        # Objectness loss -- compute for all subsampled proposals (use valid_mask)
        if self.do_distribution_obj:
            objectness_loss = F.binary_cross_entropy_with_logits(
                cat([torch.flatten(t) for t in student_objectness_logits])[valid_mask],
                teacher_objectness_probs[valid_mask],
                reduction="mean"
            )
            losses["loss_obj_bce"] = objectness_loss

        # Regression loss -- compute only for positive proposals (use fg_mask)
        if self.do_distribution_rpn_reg:
            fg_mask = torch.repeat_interleave(fg_mask, repeats=4)
            loss_rpn_reg = smooth_l1_loss(
                cat([torch.flatten(t) for t in student_proposal_deltas])[fg_mask],
                cat([torch.flatten(t) for t in teacher_proposal_deltas])[fg_mask],
                beta=0.0, # default
                reduction="mean"
            )
            losses["loss_rpn_l1"] = loss_rpn_reg

        return losses
    
    def get_roih_losses(self):
        losses = {}
        student_cls_logits, student_proposal_deltas = self.student_boxpred_io.output
        teacher_cls_logits, teacher_proposal_deltas = self.teacher_boxpred_io.output

        # Postprocessing -- for now just sharpening
        teacher_cls_probs = F.softmax(teacher_cls_logits / self.cls_temperature, dim=1)

        # ROI heads classification loss
        if self.do_distribution_cls:
            if self.cls_loss_type == "CE":
                cls_dst_loss = cross_entropy(student_cls_logits, teacher_cls_probs)
            elif self.cls_loss_type == "KL":
                cls_dst_loss = F.kl_div(F.log_softmax(student_cls_logits, dim=1),
                                    F.log_softmax(teacher_cls_logits / self.cls_temperature, dim=1),
                                    reduction="batchmean",
                                    log_target=True)
            else:
                raise ValueError("cls_loss_type must be one of {CE, KL}")
            losses["loss_cls_ce"] = cls_dst_loss

        # ROI box loss
        if self.do_distribution_roih_reg:
            # get the regression targets for all pseudo-foreground proposals
            bg_idx = teacher_cls_logits.shape[1] - 1
            fg_cls = torch.argmax(teacher_cls_logits, dim=1)
            fg_mask = fg_cls != bg_idx
            
            fg_teacher_deltas = teacher_proposal_deltas.view(-1, bg_idx, 4)[
                fg_mask, fg_cls[fg_mask], :
            ]
            fg_student_deltas = student_proposal_deltas.view(-1, bg_idx, 4)[
                fg_mask, fg_cls[fg_mask], :
            ]

            loss_roih_reg = smooth_l1_loss(
                    fg_student_deltas,
                    fg_teacher_deltas,
                    beta=0.0, # default
                    reduction="sum"
                )
            
            # normalize by the total number of regions so that each proposal is given
            # equal weight; see detectron2.modeling.roi_heads.fast_rcnn.py:box_reg_loss
            normalizer = teacher_cls_logits.shape[0]
            losses["loss_roih_l1"] = loss_roih_reg / normalizer

        return losses


# Any modifications to the torch module itself go here and are mixed in
# See align.py for an example
# For now, no modifications are needed
@DISTILL_MIXIN_REGISTRY.register()
class DistillMixin(GeneralizedRCNN): pass
