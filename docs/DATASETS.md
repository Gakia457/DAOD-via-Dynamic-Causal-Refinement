# Benchmark datasets setup

We use three adaptation benchmarks:

1. Cityscapes -> Foggy Cityscapes (C2F)
2. Cityscapes -> BDD100K-daytime (C2B)
3. Sim10k -> Cityscapes (S2C)

Please download the official images/annotations from the dataset websites:

- Cityscapes and Foggy Cityscapes: [link](https://www.cityscapes-dataset.com)
- BDD100K: [link](https://bair.berkeley.edu/blog/2018/05/30/bdd/)
- Sim10k: [link](https://deepblue.lib.umich.edu/data/downloads/ks65hc58r)

After downloading official files, prepare task-specific COCO-format JSON splits used in this repo:

1. **C2F:** build both `beta=0.02` and `ALL` splits from Foggy Cityscapes annotations (e.g., `cityscapes_{train,val}_instances_foggy_beta_0.02.json` and `cityscapes_{train,val}_instances_foggyALL.json`).
2. **C2B:** filter BDD100K annotations to daytime only, and keep 8 classes in the same order as Cityscapes/Foggy: `person`, `rider`, `car`, `truck`, `bus`, `train`, `motorcycle`, `bicycle`. Since `train` has very few images in BDD100K, we usually report the other 7 classes.
3. **S2C:** filter Cityscapes COCO-format JSON annotations for both train/val splits to keep only class `car` (e.g., `cityscapes_{train,val}_instances_cars.json`).

In order to work with our default config files, all data is expected to be in the `./datasets` directory of this repository in the following directory structure:

```text
datasets/
    cityscapes/
        leftImg8bit/
            train/...
            val/...
        annotations/
            cityscapes_train_instances.json
            cityscapes_val_instances.json
            cityscapes_{train,val}_instances_cars.json
            ...
    foggy_cityscapes/
        leftImg8bit_foggy/
            train/...
            val/...
        annotations/
            cityscapes_{train,val}_instances_foggy_beta_0.02.json
            cityscapes_{train,val}_instances_foggyALL.json
            ...
    bdd100k/
        images/
            train/...
            val/...
        annotations/
            bdd100k_daytime_train_coco.json
            bdd100k_daytime_val_coco.json
            ...
    sim10k/
        images/
        annotations/
            coco_car_annotations.json
            ...
    ...
```

Dataset registration is maintained in [`./dcr/datasets.py`](../dcr/datasets.py).

After organizing the dataset, you can preceed to configure the corresponding dictionary paths in the dataset.py like:

```python
from detectron2.data.datasets import register_coco_instances

# Example: register one dataset entry in ./dcr/datasets.py
register_coco_instances(
    "cityscapes_foggy_train", {},
    "/PATH/TO/datasets/foggy_cityscapes/annotations/cityscapes_train_instances_foggy_beta_0.02.json",
    "/PATH/TO/datasets/foggy_cityscapes/leftImg8bit_foggy/train/",
)
# ... configure other dataset entries following the same pattern.
```

## Custom dataset

The easiest way to use your own dataset is to create [COCO-formatted JSON files](https://docs.aws.amazon.com/rekognition/latest/customlabels-dg/md-coco-overview.html) and [register your datasets with Detectron2](https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html#register-a-coco-format-dataset).

You will register each set separately (usually in `./dcr/datasets.py`):

```python
from detectron2.data.datasets import register_coco_instances

register_coco_instances(
    "your_train_dataset_name", {},
    "path/to/your_train_coco_labels.json",
    "path/to/your/train/images/",
)
register_coco_instances(
    "your_unlabeled_dataset_name", {},
    "path/to/your_unlabeled_coco_labels.json",
    "path/to/your/unlabeled/images/",
)
register_coco_instances(
    "your_val_dataset_name", {},
    "path/to/your_val_coco_labels.json",
    "path/to/your/val/images/",
)
```

By default, Detectron2 assumes dataset paths are under `./datasets` (relative to your current working directory). You can change this location using:

```bash
export DETECTRON2_DATASETS=/path/to/datasets
```
