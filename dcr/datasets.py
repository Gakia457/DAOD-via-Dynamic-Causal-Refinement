"""Dataset registration entries and templates for DCR experiments."""

from detectron2.data.datasets import register_coco_instances

# for evaluating COCO-pretrained models: category IDs are remapped to match
# register_coco_instances("cityscapes_foggy_val_coco_ids", {},     "datasets/cityscapes/annotations/cityscapes_val_instances_foggyALL_coco.json",     "datasets/cityscapes/leftImg8bit_foggy/val/")


# Fill in `path_to_xxx_dataset/{train.json,val.json,train_image,val_image}` style
# paths and keep dataset names consistent with configs.
#
# # Cityscapes
# register_coco_instances("cityscapes_train", {}, "path_to_cityscapes_dataset/train.json", "path_to_cityscapes_dataset/train_image/")
# register_coco_instances("cityscapes_val",   {}, "path_to_cityscapes_dataset/val.json",   "path_to_cityscapes_dataset/val_image/")
#
# # Foggy Cityscapes beta0.02
# register_coco_instances("cityscapes_foggy_train", {}, "path_to_foggy_cityscapes_dataset/train_beta0.02.json", "path_to_foggy_cityscapes_dataset/train_image/")
# register_coco_instances("cityscapes_foggy_val",   {}, "path_to_foggy_cityscapes_dataset/val_beta0.02.json",   "path_to_foggy_cityscapes_dataset/val_image/")
#
# # BDD100K Daytime
# register_coco_instances("bdd100k_train_daytime", {}, "path_to_bdd100k_dataset/train.json", "path_to_bdd100k_dataset/train_image/")
# register_coco_instances("bdd100k_val_daytime",   {}, "path_to_bdd100k_dataset/val.json",   "path_to_bdd100k_dataset/val_image/")
#
# # Sim10k (car)
# register_coco_instances("sim10k_cars_train", {}, "path_to_sim10k_dataset/train.json", "path_to_sim10k_dataset/train_image/")
#
# # Cityscapes (car only)
# register_coco_instances("cityscapes_cars_train", {}, "path_to_cityscapes_car_dataset/train.json", "path_to_cityscapes_car_dataset/train_image/")
# register_coco_instances("cityscapes_cars_val",   {}, "path_to_cityscapes_car_dataset/val.json",   "path_to_cityscapes_car_dataset/val_image/")
