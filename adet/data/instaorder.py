import io
import json
import logging
import contextlib
import numpy as np


from PIL import Image
from fvcore.common.file_io import PathManager
from pycocotools.coco import COCO

from detectron2.structures import BoxMode, PolygonMasks, Boxes
from detectron2.data import MetadataCatalog, DatasetCatalog
from pycocotools import mask as maskUtils


logger = logging.getLogger(__name__)

__all__ = ["load_instaorder_json"]


def load_instaorder_json(json_file, image_root, dataset_name=None, extra_annotation_keys=None):
    json_file = PathManager.get_local_path(json_file)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = COCO(json_file)

    img_ids = sorted(coco_api.imgs.keys())
    imgs = coco_api.loadImgs(img_ids)
    anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
    total_num_valid_anns = sum([len(x) for x in anns])
    total_num_anns = len(coco_api.anns)
    if total_num_valid_anns < total_num_anns:
        logger.warning(
            f"{json_file} contains {total_num_anns} annotations, but only "
            f"{total_num_valid_anns} of them match to images in the file."
        )

    imgs_anns = list(zip(imgs, anns))
    logger.info("Loaded {} images in COCO format from {}".format(len(imgs_anns), json_file))

    dataset_dicts = []
    ann_keys = ["bbox", "category_id", "visible_bbox", "area"] + (extra_annotation_keys or [])

    num_instances_without_valid_segmentation = 0
    for (img_dict, anno_dict_list) in imgs_anns:
        record = {}
        record["file_name"] = image_root + img_dict["file_name"]
        record["depth_file_name"] = image_root + img_dict["depth_file_name"]
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        record["rel_mat"] = img_dict["rel_mat"]
        image_id = record["image_id"] = img_dict["image_id"]

        objs = []
        for anno in anno_dict_list:
            assert anno["image_id"] == image_id

            obj = {key: anno[key] for key in ann_keys if key in anno}
            if anno.get("segmentation", None):  # either list[list[float]] or dict(RLE)
                obj["segmentation"] = anno.get("segmentation", None)
            if anno.get("visible_mask", None): 
                obj["visible_mask"] = anno.get("visible_mask", None)
            if anno.get("occluded_mask", None):
                obj["occluded_mask"] = anno.get("occluded_mask", None)
            obj["occluded_rate"] = anno.get("occluded_rate", None)

            obj["bbox_mode"] = BoxMode.XYWH_ABS     # (x0, y0, w, h)
        
            objs.append(obj)

        record["annotations"] = objs
        dataset_dicts.append(record)

    if num_instances_without_valid_segmentation > 0:
        logger.warning(
            "Filtered out {} instances without valid segmentation. ".format(
                num_instances_without_valid_segmentation
            )
            + "There might be issues in your dataset generation process. "
            "A valid polygon should be a list[float] with even length >= 6."
        )
    return dataset_dicts


