#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Copied directly from detectron2/tools/train_net.py except where noted.
"""
from datetime import timedelta

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.evaluation import verify_results

from dcr.checkpoint import DetectionCheckpointerWithEMA
from dcr.config import add_dcr_config
from dcr.ema import EMA
from dcr.trainer import DCRTrainer
import dcr.align # register align mixins with Detectron2
import dcr.datasets # register datasets with Detectron2
import dcr.distill # register distillers and distill mixins with Detectron2
import dcr.model # register DCR R-CNN model with Detectron2
import dcr.backbone # register ViT FPN backbone with Detectron2
import dcr.clip_backbone # register RegionCLIP backbone with Detectron2

def setup(args):
    """
    Copied directly from detectron2/tools/train_net.py
    """
    cfg = get_cfg()

    ## Change here
    add_dcr_config(cfg)
    ## End change

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def main(args):
    """
    Copied directly from detectron2/tools/train_net.py
    But replace Trainer with DATrainer and disable TTA.
    """
    cfg = setup(args)

    if args.eval_only:
        model = DCRTrainer.build_model(cfg)
        ## Change here
        ckpt = DetectionCheckpointerWithEMA(model, save_dir=cfg.OUTPUT_DIR)
        if cfg.EMA.ENABLED and cfg.EMA.LOAD_FROM_EMA_ON_START:
            ema = EMA(DCRTrainer.build_model(cfg), cfg.EMA.ALPHA, cfg.EMA.START_ITER)
            ckpt.add_checkpointable("ema", ema)
        ckpt.resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
        ## End change
        res = DCRTrainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            raise NotImplementedError("TTA not supported")
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = DCRTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        timeout=timedelta(minutes=1), # added for debugging
        args=(args,),
    )
