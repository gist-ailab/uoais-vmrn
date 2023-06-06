# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
from detectron2.data import DatasetCatalog, MetadataCatalog

from .cocoa import load_cocoa_json

def register_cocoa_instances(name, metadata, json_file, image_root):
    """
    Register a dataset in uoais's json annotation format for
    instance detection
    Args:
        name (str): the name that identifies a dataset, e.g. "d2sa_train".
        metadata (dict): extra metadata associated with this dataset.  You can
            leave it as an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str): directory which contains all the images.
    """

    # 1. register a function which returns dicts
    DatasetCatalog.register(name, lambda: load_cocoa_json(json_file, image_root, name,))
    evaluator_type = "insta"

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type=evaluator_type, **metadata
    )

