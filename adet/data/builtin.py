# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


"""
This file registers pre-defined datasets at hard-coded paths, and their metadata.

We hard-code metadata for common datasets. This will enable:
1. Consistency check when loading the datasets
2. Use models on these standard datasets directly and run demos,
   without having to download the dataset annotations

We hard-code some paths to the dataset that's assumed to
exist in "./datasets/".

Users SHOULD NOT use this file to create new dataset / metadata for new dataset.
To add new dataset, refer to the tutorial "docs/DATASETS.md".
"""

import os

from detectron2.data import MetadataCatalog, DatasetCatalog
from .builtin_meta import _get_builtin_metadata
from .register_uoais import register_uoais_instances
from .register_wisdom import register_wisdom_instances
from .register_cocoa import register_cocoa_instances
from .register_meta import register_meta_instances
from .register_instaorder import register_instaorder_instances




_PREDEFINED_SPLITS_INSTAORDER = {
    "insta_train": ("COCO/train2017/", "COCO/annotations/instances_train2017.json"),
    "insta_val": ("COCO/val2017/", "COCO/annotations/instances_val2017.json"),

}

def register_all_instaorder(root='./datasets'):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_INSTAORDER.items():
        register_instaorder_instances(
            key,
            {},
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )



_PREDEFINED_SPLITS_META = {
    "meta_real_train": ("MetaGraspNet/dataset_real/", "MetaGraspNet/Annotations/meta_real_train.json"),
    "meta_real_val": ("MetaGraspNet/dataset_real/", "MetaGraspNet/Annotations/meta_real_val.json"),

    "meta_sim_train": ("MetaGraspNet/dataset_sim/", "MetaGraspNet/Annotations/meta_sim_train.json"),
    "meta_sim_val": ("MetaGraspNet/dataset_sim/", "MetaGraspNet/Annotations/meta_sim_val.json"),

    "meta_sim_train_wo45": ("MetaGraspNet/dataset_sim/", "MetaGraspNet/Annotations/meta_sim_train_wo45.json"),
    "meta_sim_val_wo45": ("MetaGraspNet/dataset_sim/", "MetaGraspNet/Annotations/meta_sim_val_wo45.json"),

    "meta_sim_train_m2": ("MetaGraspNet/dataset_sim/", "MetaGraspNet/Annotations/meta_sim_train_m2.json"),
    "meta_sim_val_m2": ("MetaGraspNet/dataset_sim/", "MetaGraspNet/Annotations/meta_sim_val_m2.json"),
    # "meta_sim_train_m2": ("MetaGraspNet/dataset_sim/", "MetaGraspNet/Annotations/meta_sim_val_m2_10.json"),
    # "meta_sim_val_m2": ("MetaGraspNet/dataset_sim/", "MetaGraspNet/Annotations/meta_sim_val_m2_10.json"),

    "meta_sim_train_m4": ("MetaGraspNet/dataset_sim/", "MetaGraspNet/Annotations/meta_sim_train_m4.json"),
    "meta_sim_val_m4": ("MetaGraspNet/dataset_sim/", "MetaGraspNet/Annotations/meta_sim_val_m4.json"),
    
    "meta_sim_train_debug": ("MetaGraspNet/dataset_sim/", "MetaGraspNet/Annotations/meta_sim_train_debug.json"),
    "meta_sim_val_debug": ("MetaGraspNet/dataset_sim/", "MetaGraspNet/Annotations/meta_sim_val_debug.json"),
}



def register_all_meta(root="./datasets"):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_META.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_meta_instances(
            key,
            # _get_builtin_metadata("vmrd"),
            {},
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )



_PREDEFINED_SPLITS_COCOA = {
    "cocoa_train": ("COCO/train2014/", "COCO/annotations/cocoa_train.json"),
    "cocoa_val": ("COCO/val2014/", "COCO/annotations/cocoa_val.json"),
    "cocoa10_train": ("COCO/train2014/", "COCO/annotations/cocoa10_train.json"),
    "cocoa10_val": ("COCO/val2014/", "COCO/annotations/cocoa10_val.json"),
    "cocoa20_train": ("COCO/train2014/", "COCO/annotations/cocoa20_train.json"),
    "cocoa20_val": ("COCO/val2014/", "COCO/annotations/cocoa20_val.json"),
    "cocoa30_train": ("COCO/train2014/", "COCO/annotations/cocoa30_train.json"),
    "cocoa30_val": ("COCO/val2014/", "COCO/annotations/cocoa30_val.json"),
}

def register_all_cocoa(root='./datasets'):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_COCOA.items():
        register_cocoa_instances(
            key,
            {},
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )



_PREDEFINED_SPLITS_uoais = {
    
    "uoais_sim_train_amodal": ("UOAIS-Sim/train", "UOAIS-Sim/annotations/coco_anns_uoais_sim_train.json"),
    "uoais_sim_val_amodal": ("UOAIS-Sim/val", "UOAIS-Sim/annotations/coco_anns_uoais_sim_val.json"),
}


def register_all_uoais(root="./datasets"):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_uoais.items():
        # Assume pre-defined datasets live in `./datasets`.
        amodal = "amodal" in key
        if "occ" in key:
            md = "uoais_occ"
        else:
            md = "uoais"
        register_uoais_instances(
            key,
            _get_builtin_metadata(md),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            amodal=amodal
        )

_PREDEFINED_SPLITS_WISDOM = {
    "wisdom_real_train": ("wisdom/wisdom-real/high-res", "wisdom/wisdom-real/high-res/annotations/coco_anns_wisdom_train.json"),
    "wisdom_real_test": ("wisdom/wisdom-real/high-res", "wisdom/wisdom-real/high-res/annotations/coco_anns_wisdom_test.json"),
}


def register_all_wisdom(root="./datasets"):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_WISDOM.items():
        # Assume pre-defined datasets live in `./datasets`.
        if "occ" in key:
            md = "wisdom_occ"
        else:
            md = "wisdom"
        register_wisdom_instances(
            key,
            _get_builtin_metadata(md),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )



# Register them all under "./datasets"
register_all_uoais()
register_all_wisdom()
register_all_cocoa()
register_all_meta()
