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
import cvbase as cvb

logger = logging.getLogger(__name__)

__all__ = ["load_instaorder_json"]


def load_instaorder_json(json_file, image_root, dataset_name=None, extra_annotation_keys=None):
    if 'train' in json_file:
        order_json_file = json_file.replace('instances_train2017', 'InstaOrder_train2017')
    elif 'val' in json_file:
        order_json_file = json_file.replace('instances_val2017', 'InstaOrder_val2017')
    order_json_file = PathManager.get_local_path(order_json_file)
    json_file = PathManager.get_local_path(json_file)

    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = COCO(json_file)
    data_order = cvb.load(order_json_file)['annotations']

    img_ids = sorted([x['image_id'] for x in data_order])

    # img_ids = sorted(coco_api.imgs.keys())
    imgs = coco_api.loadImgs(img_ids)
    anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
    total_num_valid_anns = sum([len(x) for x in anns])
    total_num_anns = len(coco_api.anns)
    if total_num_valid_anns < total_num_anns:
        logger.warning(
            f"{json_file} contains {total_num_anns} annotations, but only "
            f"{total_num_valid_anns} of them match to images in the file."
        )

    imgs_anns = list(zip(imgs, anns, data_order))
    logger.info("Loaded {} images in COCO format from {}".format(len(imgs_anns), json_file))

    dataset_dicts = []
    ann_keys = ["bbox", "category_id", "area"] + (extra_annotation_keys or [])

    num_instances_without_valid_segmentation = 0
    for (img_dict, anno_dict_list, order_dict) in imgs_anns:
        record = {}
        record["file_name"] = image_root + img_dict["file_name"]
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        image_id = record["image_id"] = img_dict["id"]
        record["rel_mat"] = np.zeros((len(order_dict['instance_ids']), len(order_dict['instance_ids'])))
        for i, data in enumerate(order_dict['occlusion']):
            occ_str = data['order']
            if '&' in occ_str:
                idx1, idx2 = list(map(int, occ_str.split(' & ')[0].split('<')))
                record['rel_mat'][idx2][idx1] = -1
                record['rel_mat'][idx1][idx2] = -1
            else:
                idx1, idx2 = list(map(int, occ_str.split('<')))
                record['rel_mat'][idx2][idx1] = -1


        objs = []
        for anno in anno_dict_list:
            assert anno["image_id"] == image_id

            obj = {key: anno[key] for key in ann_keys if key in anno}
            if anno.get("segmentation", None):  # either list[list[float]] or dict(RLE)
                obj["segmentation"] = anno.get("segmentation", None)

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




if __name__ == "__main__":
    from detectron2.utils.logger import setup_logger
    from detectron2.utils.visualizer import Visualizer
    import os, sys
    import tqdm
    

    logger = setup_logger(name=__name__)

    dicts = load_instaorder_json("/ailab_mat/dataset/InstaOrder/data/COCO/annotations/instances_train2017.json",
                           "datasets/COCO/train2017/")
    logger.info("Done loading {} samples.".format(len(dicts)))
    print("Done loading {} samples.".format(len(dicts)))
    dirname = "insta-data-vis"

    os.makedirs(dirname, exist_ok=True)
    i = 0
    for d in (dicts):
        print(d["file_name"])
        img = Image.open(d["file_name"])
        visualizer = Visualizer(img)
        vis = visualizer.draw_dataset_dict(d)
        fpath = os.path.join(dirname, os.path.basename(d["file_name"]))
        vis.save(fpath)
        i += 1
        if i==5: break
