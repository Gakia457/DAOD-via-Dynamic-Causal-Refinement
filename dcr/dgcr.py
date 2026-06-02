"""SP transform and discrepancy-guided dynamic controller."""

import logging
import random
from typing import Any, Dict

import cv2
import numpy as np
import torch
from fvcore.transforms.transform import NoOpTransform, Transform


class SPTransform(Transform):
    """Apply Spectral Perturbation in frequency domain."""

    _shared_v1_scale = 0.005
    _shared_v2_scale = 0.7
    _use_shared = False

    def __init__(self, v1_scale: float = 0.005, v2_scale: float = 0.7, use_shared: bool = False):
        super().__init__()
        self._set_attributes(locals())
        self.variant = random.randint(0, 9)

    def apply_sp(self, img: np.ndarray) -> np.ndarray:
        """Transform an RGB image with SP."""
        if self.use_shared or SPTransform._use_shared:
            v1_scale = SPTransform._shared_v1_scale
            v2_scale = SPTransform._shared_v2_scale
        else:
            v1_scale = self.v1_scale
            v2_scale = self.v2_scale

        try:
            h, w = img.shape[:2]
            if h % 2 != 0 or w % 2 != 0:
                img = cv2.resize(img, (w - w % 2, h - h % 2), interpolation=cv2.INTER_AREA)
                h, w = img.shape[:2]

            img_dct = np.zeros((h, w, 3), dtype=np.float32)
            for i in range(3):
                img_dct[:, :, i] = cv2.dct(img[:, :, i].astype(np.float32))

            mask = np.zeros((h, w, 3), dtype=np.float32)
            v1 = int(min(h, w) * v1_scale)
            v2 = int(min(h, w) * v2_scale)
            v3 = min(h, w)

            x_indices, y_indices = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
            max_xy = np.maximum(x_indices, y_indices)

            for c in range(3):
                cond1 = max_xy <= v1
                mask[:, :, c][cond1] = 1 - max_xy[cond1] / v1 * 0.95
                cond2 = (max_xy > v1) & (max_xy <= v2)
                mask[:, :, c][cond2] = 0.01
                cond3 = (max_xy > v2) & (max_xy <= v3)
                mask[:, :, c][cond3] = (max_xy[cond3] - v2) / (v3 - v2) * 0.3
                cond4 = max_xy > v3
                mask[:, :, c][cond4] = 0.5

            n_mask = 1 - mask
            non_img_dct = img_dct * mask
            cal_img_dct = img_dct * n_mask

            np.random.seed(self.variant)
            ref_dct = np.zeros_like(non_img_dct)
            ref_dct[:, :, 0] = non_img_dct[:, :, 0] * (1 + np.random.randn())
            ref_dct[:, :, 1] = non_img_dct[:, :, 1] * (1 + np.random.randn())
            ref_dct[:, :, 2] = non_img_dct[:, :, 2] * (1 + np.random.randn())

            img_fc = ref_dct + cal_img_dct
            img_out = np.zeros((h, w, 3), dtype=np.float32)
            for i in range(3):
                img_out[:, :, i] = cv2.idct(img_fc[:, :, i]).clip(0, 255)

            return img_out.astype(np.uint8)
        except Exception as e:
            logging.getLogger(__name__).warning(f"SP operation failed: {e}, returning original image")
            return img

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        return self.apply_sp(img)

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        return coords

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        return segmentation

    def inverse(self) -> Transform:
        return NoOpTransform()


class DiscrepancyGuider:
    """Dynamically adjust SP parameters from distillation signals."""

    def __init__(self, cfg):
        self.enabled = cfg.AUG.SP_DYNAMIC_ADJUST and cfg.AUG.SP_PROB > 0
        self.feature_distill_enabled = cfg.DOMAIN_ADAPT.DISTILL.FEATURE_DISTILL_ENABLED
        self.random_mode = cfg.AUG.SP_RANDOM_ADJUST
        self.exclude_feature_fb = cfg.AUG.SP_EXCLUDE_FEATURE_FB
        self.logger = logging.getLogger(__name__)

        if self.enabled:
            self.initial_v1_scale = cfg.AUG.SP_V1_SCALE
            self.initial_v2_scale = cfg.AUG.SP_V2_SCALE
            self.ema_decay = cfg.AUG.SP_EMA_DECAY

            SPTransform._use_shared = True
            SPTransform._shared_v1_scale = self.initial_v1_scale
            SPTransform._shared_v2_scale = self.initial_v2_scale

            self.loss_ema_distill = 1.0
            self.iteration = 0
            self.random_update_interval = 100
            self.last_random_update_iter = 0

            self.logger.info(
                "[SP] dynamic enabled (random=%s, exclude_feature_fb=%s)",
                self.random_mode,
                self.exclude_feature_fb,
            )
        elif cfg.AUG.SP_PROB > 0:
            self.logger.info("[SP] fixed parameters mode")
        else:
            self.logger.info("[SP] disabled (SP_PROB=0)")

    def update_parameters(self, loss_dict: Dict[str, Any], iteration: int) -> None:
        """Update shared SP parameters from current losses."""
        if not self.enabled:
            return

        if self.random_mode:
            self._random_adjust(iteration)
            return

        loss_distill = sum(v for k, v in loss_dict.items() if k.endswith("_distill"))
        if self.feature_distill_enabled and not self.exclude_feature_fb:
            loss_distill += sum(v for k, v in loss_dict.items() if k.startswith("loss_feature_"))

        if isinstance(loss_distill, torch.Tensor):
            loss_distill_value = loss_distill.detach().cpu().item()
        else:
            loss_distill_value = float(loss_distill)

        if loss_distill_value <= 0:
            return

        self.iteration = iteration
        alpha = self._compute_alpha(loss_distill_value)
        self._update_sp_transform_params(alpha)
        self._log_metrics(alpha, loss_distill_value)

    def _compute_alpha(self, loss_distill: float) -> float:
        """Compute multiplicative update factor for SP scales."""
        if self.iteration == 0:
            self.loss_ema_distill = loss_distill
            return 1.0

        self.loss_ema_distill = self.ema_decay * self.loss_ema_distill + (1 - self.ema_decay) * loss_distill
        ema = self.loss_ema_distill
        relative_level = np.clip((loss_distill / (ema + 1e-6)) - 1, -2.0, 2.0)
        delta = np.clip((loss_distill - ema) / (ema + 1e-6), -2.0, 2.0)
        score = 0.7 * relative_level + 0.3 * delta
        alpha = 1.0 + 0.03 * np.tanh(score * 1.5)
        return np.clip(alpha, 0.97, 1.03)

    def _update_sp_transform_params(self, alpha: float) -> None:
        """Update shared SP scale parameters."""
        SPTransform._shared_v1_scale = np.clip(SPTransform._shared_v1_scale / alpha, 0.003, 0.03)
        SPTransform._shared_v2_scale = np.clip(SPTransform._shared_v2_scale * alpha, 0.6, 0.8)

    def _log_metrics(self, alpha: float, loss_distill: float) -> None:
        """Log SP metrics to event storage."""
        from detectron2.utils.events import get_event_storage

        storage = get_event_storage()
        storage.put_scalar("sp/alpha", alpha)
        storage.put_scalar("sp/v1_scale", SPTransform._shared_v1_scale)
        storage.put_scalar("sp/v2_scale", SPTransform._shared_v2_scale)
        storage.put_scalar("sp/loss_distill", loss_distill)
        storage.put_scalar("sp/loss_distill_ema", self.loss_ema_distill)

    def _random_adjust(self, iteration: int) -> None:
        """Apply random SP updates for ablations."""
        self.iteration = iteration

        # Option A: sample every N iterations.
        # if iteration - self.last_random_update_iter >= self.random_update_interval:
        #     SPTransform._shared_v1_scale = np.random.uniform(0.003, 0.03)
        #     SPTransform._shared_v2_scale = np.random.uniform(0.6, 0.8)
        #     self.last_random_update_iter = iteration
        #     self._log_metrics_random()

        # Option B: perturb every iteration.
        alpha = np.random.uniform(0.97, 1.03)
        self._update_sp_transform_params(alpha)
        self._log_metrics_random()

    def _log_metrics_random(self) -> None:
        """Log SP scales in random-adjust mode."""
        from detectron2.utils.events import get_event_storage

        storage = get_event_storage()
        storage.put_scalar("sp/v1_scale", SPTransform._shared_v1_scale)
        storage.put_scalar("sp/v2_scale", SPTransform._shared_v2_scale)
