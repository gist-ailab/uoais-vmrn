# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Modified from https://github.com/YutingXiao/Amodal-Segmentation-Based-on-Visible-Region-Segmentation-and-Shape-Prior
import contextlib
import copy
import io
import itertools
import json
import logging
from math import pi
import numpy as np
import os
import pickle
from PIL import Image
from collections import OrderedDict
import pycocotools.mask as mask_util
import torch
from fvcore.common.file_io import PathManager

from adet.data.amodal_datasets.pycocotools.coco import COCO
from adet.data.amodal_datasets.pycocotools.cocoeval import COCOeval

from tabulate import tabulate

import detectron2.utils.comm as comm
from detectron2.data import MetadataCatalog
from detectron2.data.datasets.coco import convert_to_coco_json
from detectron2.structures import Boxes, BoxMode, pairwise_iou
from detectron2.utils.logger import create_small_table
from detectron2.utils.events import get_event_storage
from detectron2.utils.visualizer import Visualizer

from .evaluator import DatasetEvaluator
from adet.utils.post_process import detector_postprocess
from sklearn.metrics import precision_score, recall_score, f1_score



def get_bbox(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return [0, 0, 0, 0]
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    return [int(x_min), int(y_min), int(x_max-x_min), int(y_max-y_min)]

class OrderEvaluator(DatasetEvaluator):
    """
    Evaluate object proposal, instance detection/segmentation, keypoint detection
    outputs using COCO's metrics and APIs.
    """

    def __init__(self, dataset_name, cfg,  distributed=True, output_dir=None):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
                It must have either the following corresponding metadata:
                    "json_file": the path to the COCO format annotation
                Or it must be in detectron2's standard dataset format
                so it can be converted to COCO format automatically.
            cfg (CfgNode): config instance
            distributed (True): if True, will collect results from all ranks for evaluation.
                Otherwise, will evaluate the results in the current process.
            output_dir (str): optional, an output directory to dump results.
        """
        self._cfg = cfg
        self._tasks = self._tasks_from_config(cfg)
        self._distributed = distributed
        self._output_dir = output_dir

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

        self._metadata = MetadataCatalog.get(dataset_name)
        if not hasattr(self._metadata, "json_file"):
            self._logger.warning(f"json_file was not found in MetaDataCatalog for '{dataset_name}'")

            cache_path = convert_to_coco_json(dataset_name, output_dir)
            self._metadata.json_file = cache_path

        json_file = PathManager.get_local_path(self._metadata.json_file)
        with contextlib.redirect_stdout(io.StringIO()):
            self._coco_api = COCO(json_file)

        self._kpt_oks_sigmas = cfg.TEST.KEYPOINT_OKS_SIGMAS
        # Test set json files do not contain annotations (evaluation must be
        # performed using the COCO evaluation server).
        self._do_evaluation = "annotations" in self._coco_api.dataset

    def reset(self):
        self._amodal_predictions = []
        self._visible_predictions = []
        self._occlusion_predictions = []
        self._order_predictions = []

        self._amodal_results = []
        self._visible_results = []
        self._occlusion_results = []
        self._order_results = []


    def _tasks_from_config(self, cfg):
        """
        Returns:
            tuple[str]: tasks that can be evaluated under the given configuration.
        """
        tasks = ("bbox",)
        tasks = tasks + ("amodal_segm",)
        tasks = tasks + ("visible_segm",)
        tasks = tasks + ("occlusion_segm",)

        return tasks

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a COCO model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a COCO model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        for i in range(len(inputs)):
            input, output, pred_rel_mat = inputs[i], outputs[2*i][0], outputs[2*i+1]
            amodal_prediction = {"image_id": input["image_id"]}
            visible_prediction = {"image_id": input["image_id"]}
            occlusion_prediction = {"image_id": input["image_id"]}
            order_prediction = {"image_id": input["image_id"]}

            if "instances" in output:
                instances = detector_postprocess(output["instances"], input["height"], input["width"])
                amodal_prediction["instances"], visible_prediction["instances"], occlusion_prediction["instances"] =\
                    amodal_instances_to_coco_json(instances, input["image_id"], type="amodal")

            if "proposals" in output:
                amodal_prediction["proposals"] = output["proposals"].to(self._cpu_device)
                visible_prediction["proposals"] = output["proposals"].to(self._cpu_device)
                occlusion_prediction["proposals"] = output["proposals"].to(self._cpu_device)
            
            order_prediction['pred_rel_mat'] = pred_rel_mat
            order_prediction['gt_rel_mat'] = np.array(input['rel_mat'])

            self._amodal_predictions.append(amodal_prediction)
            self._visible_predictions.append(visible_prediction)
            self._occlusion_predictions.append(occlusion_prediction)
            self._order_predictions.append(order_prediction)


    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            self._amodal_predictions = comm.gather(self._amodal_predictions, dst=0)
            self._amodal_predictions = list(itertools.chain(*self._amodal_predictions))

            self._visible_predictions = comm.gather(self._visible_predictions, dst=0)
            self._visible_predictions = list(itertools.chain(*self._visible_predictions))

            self._occlusion_predictions = comm.gather(self._occlusion_predictions, dst=0)
            self._occlusion_predictions = list(itertools.chain(*self._occlusion_predictions))
            
            self._order_predictions = comm.gather(self._order_predictions, dst=0)
            self._order_predictions = list(itertools.chain(*self._order_predictions))

            if not comm.is_main_process():
                return {}
        if len(self._amodal_predictions) == 0 or len(self._visible_predictions) == 0 or len(self._occlusion_predictions) == 0:
            self._logger.warning("[Amodal_VisibleEvaluator] 1st Did not receive valid predictions.")
            return {}

        PathManager.mkdirs(self._output_dir)
        file_path = os.path.join(self._output_dir, "instances_amodal_predictions.pth")
        with PathManager.open(file_path, "wb") as f:
            torch.save(self._amodal_predictions, f)

        file_path = os.path.join(self._output_dir, "instances_visible_predictions.pth")
        with PathManager.open(file_path, "wb") as f:
            torch.save(self._visible_predictions, f)

        file_path = os.path.join(self._output_dir, "instances_occlusion_predictions.pth")
        with PathManager.open(file_path, "wb") as f:
            torch.save(self._occlusion_predictions, f)

        self._results = OrderedDict()
        if "proposals" in self._amodal_predictions[0]:
            self._eval_box_proposals()
        if "instances" in self._amodal_predictions[0]:
            self._eval_predictions(set(self._tasks))
        # Copy so the caller can do whatever with results
        self._eval_order_predictions()
        return copy.deepcopy(self._results)
    
    def _eval_order_predictions(self):
        
        ## evaluate GT vs pred relationship matrix
        total_conf = np.array([[0,0,0],[0,0,0],[0,0,0]])
        for i, prediction in enumerate(self._order_predictions):
            print(prediction['pred_rel_mat'], prediction['gt_rel_mat'])
            pred = prediction['pred_rel_mat']
            gt = prediction['gt_rel_mat']

            if len(pred) != len(gt):
                print(f'Error: {len(pred)} != {len(gt)}')
                continue

            np.fill_diagonal(gt, -2)
            np.fill_diagonal(pred, -2)

            gt_order = gt[gt != -2].reshape(-1)
            pred_order = pred[pred != -2].reshape(-1)

            gt_order[gt_order == -1] = 1
            pred_order[pred_order == -1] = 1

            print('gt ', gt_order)
            print('pr ', pred_order)

            recall = recall_score(gt_order, pred_order, average='binary', zero_division=1)
            precision = precision_score(gt_order, pred_order, average='binary', zero_division=1)
            f1 = f1_score(gt_order, pred_order, average='binary', zero_division=1)

        #     PP, PC, PN = 0, 0, 0
        #     CP, CC, CN = 0, 0, 0
        #     NP, NC, NN = 0, 0, 0

        #     for i in range(len(gt)):
        #         for j in range(len(gt)):
        #             if gt[j][i] == -1:          ## i is child of j
        #                 if pred[j][i] == -1:        ## i is predicted as child of j
        #                     CC += 1
        #                 elif pred[i][j] == -1:      ## i is predicted as parent of j
        #                     CP += 1
        #                 elif pred[i][j] == 0:       ## i is predicted as no relation with j
        #                     CN += 1
        #             elif gt[i][j] == -1:        ## i is parent of j
        #                 if pred[j][i] == -1:        ## i is predicted as child of j
        #                     PC += 1
        #                 elif pred[i][j] == -1:      ## i is predicted as parent of j
        #                     PP += 1
        #                 elif pred[i][j] == 0:       ## i is predicted as no relation with j
        #                     PN += 1
        #             elif gt[i][j] == 0:         ## i is no relation with j
        #                 if pred[j][i] == -1:        ## i is predicted as child of j
        #                     NC += 1
        #                 elif pred[i][j] == -1:      ## i is predicted as parent of j
        #                     NP += 1
        #                 elif pred[i][j] == 0:       ## i is predicted as no relation with j
        #                     NN += 1
        #     conf = np.array([[PP, PC, PN],[CP, CC, CN],[NP, NC, NN]])
        #     total_conf += conf
        #     print(conf)
        
        # print('-----------------')
        # print(total_conf)
        # PP, PC, PN = total_conf[0][0], total_conf[0][1], total_conf[0][2]
        # CP, CC, CN = total_conf[1][0], total_conf[1][1], total_conf[1][2]
        # NP, NC, NN = total_conf[2][0], total_conf[2][1], total_conf[2][2]


        # parent_precision = PP / (PP + CP + NP) if (PP + CP + NP) else 0
        # parent_recall = PP / (PP + PC + PN) if (PP + PC + PN) else 0
        # child_precision = CC / (PC + CC + NC) if (PC + CC + NC) else 0
        # child_recall = CC / (CP + CC + CN) if (CP + CC + CN) else 0
        # none_precision = NN / (PN + CN + NN) if (PN + CN + NN) else 0
        # none_recall = NN / (NP + NC + NN) if (NP + NC + NN) else 0
        # print('[total] acc', (PP+CC+NN)/(PP+PC+PN+CP+CC+CN+NP+NC+NN), \
        #     '\n[Parent] precision', parent_precision, '\trecall', parent_recall, \
        #     '\n[Child] precision', child_precision, '\trecall', child_recall, \
        #     '\n[None] precision', none_precision, '\trecall', none_recall)
        print('recall: ', recall*100, 'precision: ', precision*100, 'f1: ', f1*100)
        
        ## return res
        res = {
            # 'accuracy': (PP+CC+NN)/(PP+PC+PN+CP+CC+CN+NP+NC+NN),
            # 'parent_precision': parent_precision, 
            # 'parent_recall': parent_recall,
            # 'child_precision': child_precision,
            # 'child_recall': child_recall,
            # 'none_precision': none_precision,
            # 'none_recall': none_recall,
            'recall': recall*100,
            'precision': precision*100,
            'f1': f1*100,
        }

        self._results["order_recovery"] = res


    def _eval_predictions(self, tasks):
        """
        Evaluate self._predictions on the given tasks.
        Fill self._results with the metrics of the tasks.
        """
        self._logger.info("Preparing results for COCO format ...")
        self._amodal_results = list(itertools.chain(*[x["instances"] for x in self._amodal_predictions]))
        self._visible_results = list(itertools.chain(*[x["instances"] for x in self._visible_predictions]))
        self._occlusion_results = list(itertools.chain(*[x["instances"] for x in self._occlusion_predictions]))

        # unmap the category ids for COCO
        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            reverse_id_mapping = {
                v: k for k, v in self._metadata.thing_dataset_id_to_contiguous_id.items()
            }
            for result in self._amodal_results:
                category_id = result["category_id"]
                assert (
                    category_id in reverse_id_mapping
                ), "A prediction has category_id={}, which is not available in the dataset.".format(
                    category_id
                )
                result["category_id"] = reverse_id_mapping[category_id]
            for result in self._visible_results:
                category_id = result["category_id"]
                assert (
                    category_id in reverse_id_mapping
                ), "A prediction has category_id={}, which is not available in the dataset.".format(
                    category_id
                )
                result["category_id"] = reverse_id_mapping[category_id]
            for result in self._occlusion_results:
                category_id = result["category_id"]
                assert (
                    category_id in reverse_id_mapping
                ), "A prediction has category_id={}, which is not available in the dataset.".format(
                    category_id
                )
                result["category_id"] = reverse_id_mapping[category_id]

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "coco_instances_amodal_results.json")
            self._logger.info("Saving results to {}".format(file_path))
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(self._amodal_results))
                f.flush()

            file_path = os.path.join(self._output_dir, "coco_instances_visible_results.json")
            self._logger.info("Saving results to {}".format(file_path))
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(self._visible_results))
                f.flush()
            file_path = os.path.join(self._output_dir, "coco_instances_occlusion_results.json")
            self._logger.info("Saving results to {}".format(file_path))
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(self._occlusion_results))
                f.flush()

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return
        coco_api_eval = copy.deepcopy(self._coco_api)
        coco_api_eval_visible = copy.deepcopy(self._coco_api)
        coco_api_eval_occlusion = copy.deepcopy(self._coco_api)

        # visible_name = "visible_mask" if "visible_mask" in coco_api_eval_visible.dataset['annotations'][0].keys() else "inmodal_seg"
        for key, ann in coco_api_eval_visible.anns.items():
            coco_api_eval_visible.anns[key]["segmentation"] = coco_api_eval_visible.anns[key]["visible_mask"]
        for key, ann in coco_api_eval_visible.anns.items():
            if coco_api_eval_visible.anns[key]["occluded_rate"] > 0.05:
                coco_api_eval_occlusion.anns[key]["segmentation"] = coco_api_eval_visible.anns[key]["occluded_mask"]
            else:
                del coco_api_eval_occlusion.anns[key]
        self._logger.info("Evaluating predictions ...")
        for task_name in sorted(tasks):
            self._logger.info("Evaluation task_name : {}".format(task_name))
            if task_name.startswith("visible"):
                coco_api = coco_api_eval_visible
            elif task_name.startswith("occlusion"):
                coco_api = coco_api_eval_occlusion
            else:
                coco_api = coco_api_eval

            _coco_results = self._amodal_results
            if task_name == "amodal_segm":
                task = "segm"
            elif task_name == "visible_segm":
                task = "segm"
                # for key, ann in coco_api_eval.anns.items():
                #     coco_api_eval.anns[key]["segmentation"] = coco_api_eval.anns[key][visible_name]
                _coco_results = self._visible_results
            elif task_name == "occlusion_segm":
                task = "segm"
                _coco_results = self._occlusion_results
            elif task_name == "bbox":
                task = "bbox"

            coco_eval = (
                _evaluate_predictions_on_coco(
                    coco_api, _coco_results, task, kpt_oks_sigmas=self._kpt_oks_sigmas
                )
                if len(self._amodal_results) > 0
                else None  # cocoapi does not handle empty results very well
            )

            res = self._derive_coco_results(
                coco_eval, task, class_names=self._metadata.get("thing_classes")
            )

            self._results[task_name] = res

    def _eval_box_proposals(self):
        """
        Evaluate the box proposals in self._predictions.
        Fill self._results with the metrics for "box_proposals" task.
        """
        if self._output_dir:
            # Saving generated box proposals to file.
            # Predicted box_proposals are in XYXY_ABS mode.
            bbox_mode = BoxMode.XYXY_ABS.value
            ids, boxes, objectness_logits = [], [], []
            for prediction in self._amodal_predictions:
                ids.append(prediction["image_id"])
                boxes.append(prediction["proposals"].proposal_boxes.tensor.numpy())
                objectness_logits.append(prediction["proposals"].objectness_logits.numpy())

            proposal_data = {
                "boxes": boxes,
                "objectness_logits": objectness_logits,
                "ids": ids,
                "bbox_mode": bbox_mode,
            }
            with PathManager.open(os.path.join(self._output_dir, "box_proposals.pkl"), "wb") as f:
                pickle.dump(proposal_data, f)

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        self._logger.info("Evaluating bbox proposals ...")
        res = {}
        areas = {"all": "", "small": "s", "medium": "m", "large": "l"}
        for limit in [100, 1000]:
            for area, suffix in areas.items():
                stats = _evaluate_box_proposals(
                    self._amodal_predictions, self._coco_api, area=area, limit=limit
                )
                key = "AR{}@{:d}".format(suffix, limit)
                res[key] = float(stats["ar"].item() * 100)
        self._logger.info("Proposal metrics: \n" + create_small_table(res))
        self._results["box_proposals"] = res

    def _derive_coco_results(self, coco_eval, iou_type, class_names=None):
        """
        Derive the desired score numbers from summarized COCOeval.
        Args:
            coco_eval (None or COCOEval): None represents no predictions from model.
            iou_type (str):
            class_names (None or list[str]): if provided, will use it to predict
                per-category AP.
        Returns:
            a dict of {metric name: score}
        """

        metrics = {
            "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
            "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl", "AR1", "AR10", "AR100"],
            "keypoints": ["AP", "AP50", "AP75", "APm", "APl"],
        }[iou_type]

        if coco_eval is None:
            self._logger.warn("No predictions from the model! Set scores to -1")
            return {metric: -1 for metric in metrics}

        # the standard metrics
        results = {metric: float(coco_eval.stats[idx] * 100) for idx, metric in enumerate(metrics)}
        self._logger.info(
            "Evaluation results for {}: \n".format(iou_type) + create_small_table(results)
        )

        if class_names is None or len(class_names) <= 1:
            return results
        # Compute per-category AP
        # from https://github.com/facebookresearch/Detectron/blob/a6a835f5b8208c45d0dce217ce9bbda915f44df7/detectron/datasets/json_dataset_evaluator.py#L222-L252 # noqa
        precisions = coco_eval.eval["precision"]
        # precision has dims (iou, recall, cls, area range, max dets)
        assert len(class_names) == precisions.shape[2]

        results_per_category = []
        for idx, name in enumerate(class_names):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]
            ap = np.mean(precision) if precision.size else float("nan")
            results_per_category.append(("{}".format(name), float(ap * 100)))

        # tabulate it
        N_COLS = min(6, len(results_per_category) * 2)
        results_flatten = list(itertools.chain(*results_per_category))
        results_2d = itertools.zip_longest(*[results_flatten[i::N_COLS] for i in range(N_COLS)])
        table = tabulate(
            results_2d,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=["category", "AP"] * (N_COLS // 2),
            numalign="left",
        )
        self._logger.info("Per-category {} AP: \n".format(iou_type) + table)

        results.update({"AP-" + name: ap for name, ap in results_per_category})
        return results


def amodal_instances_to_coco_json(instances, img_id, type="amodal"):
    """
    Dump an "Instances" object to a COCO-format json that's used for evaluation.
    Args:
        instances (Instances):
        img_id (int): the image id
    Returns:
        list[dict]: list of json annotations in COCO format.
    """
    num_instance = len(instances)
    if num_instance == 0:
        return [], [], []

    boxes = instances.pred_boxes.tensor.cpu().numpy()
    boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
    boxes = boxes.tolist()
    scores = instances.scores.tolist()
    classes = instances.pred_classes.tolist()
    # occlusion classification results
    if instances.has("pred_occlusions"):
        occ_inst = [i for i, occ_cls in enumerate(instances.pred_occlusions) if occ_cls == 1]        
    else:
        occ_inst = []
        for i, (amodal_mask, visible_mask) in enumerate(zip(instances.pred_masks, instances.pred_visible_masks)):
            # print(amodal_mask, visible_mask)
            ratio = torch.sum(visible_mask) / torch.sum(amodal_mask)
            # print(ratio)
            if ratio < 0.95:
                occ_inst.append(i)
                

        # use RLE to encode the masks, because they are too large and takes memory
        # since this evaluator stores outputs of the entire dataset
    if type == "amodal":
        amodal_rles = [
            mask_util.encode(np.array(mask[:, :, None].cpu(), order="F", dtype="uint8"))[0]
            for mask in instances.pred_masks
        ]
        amodal_area = [
            float(torch.sum(mask))
            for mask in instances.pred_masks
        ]
        boxes = instances.pred_boxes.tensor.cpu().numpy()
        boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
        amodal_boxes = boxes.tolist()
        
        visible_rles = [
            mask_util.encode(np.array(mask[:, :, None].cpu(), order="F", dtype="uint8"))[0]
            for mask in instances.pred_visible_masks
        ]
        
        visible_area = [
            float(torch.sum(mask))
            for mask in instances.pred_visible_masks
        ]
        visible_boxes = [
            get_bbox(np.array(mask[:, :].cpu()))
            for mask in instances.pred_visible_masks
        ]

        occlusion_area = [
            float(torch.sum(mask))
            for mask in instances.pred_occluded_masks
        ]
        occlusion_rles = [
            mask_util.encode(np.array(mask[:, :, None].cpu(), order="F", dtype="uint8"))[0]
            for mask in instances.pred_occluded_masks
        ]
        #TODO: OCCUSION BBOX => instances.pred_occluded_masks
        occlusion_boxes = [
            get_bbox(np.array(mask[:, :].cpu()))
            for mask in instances.pred_masks
        ]

        area = [
            (torch.sum(amodal_mask * visible_mask).float() / torch.sum(amodal_mask).float()).item()
            for amodal_mask, visible_mask in zip(instances.pred_masks, instances.pred_visible_masks)
        ]

    else:
        raise ValueError("type == {} is not available")

    for amodal_rle, visible_rle, occlusion_rle in zip(amodal_rles, visible_rles, occlusion_rles):
        # "counts" is an array encoded by mask_util as a byte-stream. Python3's
        # json writer which always produces strings cannot serialize a bytestream
        # unless you decode it. Thankfully, utf-8 works out (which is also what
        # the pycocotools/_mask.pyx does).
        amodal_rle["counts"] = amodal_rle["counts"].decode("utf-8")
        visible_rle["counts"] = visible_rle["counts"].decode("utf-8")
        occlusion_rle["counts"] = occlusion_rle["counts"].decode("utf-8")


    amodal_results = []
    visible_results = []
    occlusion_results = []
    for k in range(num_instance):
        result = {
            "image_id": img_id,
            "category_id": classes[k],
            "bbox": amodal_boxes[k],
            "score": scores[k],
            "segmentation": amodal_rles[k],
            "area": amodal_area[k]
        }
        amodal_results.append(result)
    for k in range(num_instance):
        result = {
            "image_id": img_id,
            "category_id": classes[k],
            "bbox": visible_boxes[k],
            "score": scores[k],
            "segmentation": visible_rles[k],
            "area": visible_area[k]
        }
        visible_results.append(result)
    for k in range(num_instance):
        # add occlusion result only if the classification result is True
        if k not in occ_inst: continue
        result = {
            "image_id": img_id,
            "category_id": classes[k],
            "bbox": occlusion_boxes[k],
            "score": scores[k],
            "segmentation": occlusion_rles[k],
            "area": occlusion_area[k]
        }
        occlusion_results.append(result)

    return amodal_results, visible_results, occlusion_results


# inspired from Detectron:
# https://github.com/facebookresearch/Detectron/blob/a6a835f5b8208c45d0dce217ce9bbda915f44df7/detectron/datasets/json_dataset_evaluator.py#L255 # noqa
def _evaluate_box_proposals(dataset_predictions, coco_api, thresholds=None, area="all", limit=None):
    """
    Evaluate detection proposal recall metrics. This function is a much
    faster alternative to the official COCO API recall evaluation code. However,
    it produces slightly different results.
    """
    # Record max overlap value for each gt box
    # Return vector of overlap values
    areas = {
        "all": 0,
        "small": 1,
        "medium": 2,
        "large": 3,
        "96-128": 4,
        "128-256": 5,
        "256-512": 6,
        "512-inf": 7,
    }
    area_ranges = [
        [0 ** 2, 1e5 ** 2],  # all
        [0 ** 2, 32 ** 2],  # small
        [32 ** 2, 96 ** 2],  # medium
        [96 ** 2, 1e5 ** 2],  # large
        [96 ** 2, 128 ** 2],  # 96-128
        [128 ** 2, 256 ** 2],  # 128-256
        [256 ** 2, 512 ** 2],  # 256-512
        [512 ** 2, 1e5 ** 2],
    ]  # 512-inf
    assert area in areas, "Unknown area range: {}".format(area)
    area_range = area_ranges[areas[area]]
    gt_overlaps = []
    num_pos = 0
    for prediction_dict in dataset_predictions:
        predictions = prediction_dict["proposals"]

        # sort predictions in descending order
        # TODO maybe remove this and make it explicit in the documentation
        inds = predictions.objectness_logits.sort(descending=True)[1]
        predictions = predictions[inds]

        ann_ids = coco_api.getAnnIds(imgIds=prediction_dict["image_id"])
        anno = coco_api.loadAnns(ann_ids)
        gt_boxes = [
            BoxMode.convert(obj["bbox"], BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
            for obj in anno
            if obj["iscrowd"] == 0
        ]
        gt_boxes = torch.as_tensor(gt_boxes).reshape(-1, 4)  # guard against no boxes
        gt_boxes = Boxes(gt_boxes)
        gt_areas = torch.as_tensor([obj["area"] for obj in anno if obj["iscrowd"] == 0])

        if len(gt_boxes) == 0 or len(predictions) == 0:
            continue

        valid_gt_inds = (gt_areas >= area_range[0]) & (gt_areas <= area_range[1])
        gt_boxes = gt_boxes[valid_gt_inds]

        num_pos += len(gt_boxes)

        if len(gt_boxes) == 0:
            continue

        if limit is not None and len(predictions) > limit:
            predictions = predictions[:limit]

        overlaps = pairwise_iou(predictions.proposal_boxes, gt_boxes)

        _gt_overlaps = torch.zeros(len(gt_boxes))
        for j in range(min(len(predictions), len(gt_boxes))):
            # find which proposal box maximally covers each gt box
            # and get the iou amount of coverage for each gt box
            max_overlaps, argmax_overlaps = overlaps.max(dim=0)

            # find which gt box is 'best' covered (i.e. 'best' = most iou)
            gt_ovr, gt_ind = max_overlaps.max(dim=0)
            assert gt_ovr >= 0
            # find the proposal box that covers the best covered gt box
            box_ind = argmax_overlaps[gt_ind]
            # record the iou coverage of this gt box
            _gt_overlaps[j] = overlaps[box_ind, gt_ind]
            assert _gt_overlaps[j] == gt_ovr
            # mark the proposal box and the gt box as used
            overlaps[box_ind, :] = -1
            overlaps[:, gt_ind] = -1

        # append recorded iou coverage level
        gt_overlaps.append(_gt_overlaps)
    gt_overlaps = torch.cat(gt_overlaps, dim=0)
    gt_overlaps, _ = torch.sort(gt_overlaps)

    if thresholds is None:
        step = 0.05
        thresholds = torch.arange(0.5, 0.95 + 1e-5, step, dtype=torch.float32)
    recalls = torch.zeros_like(thresholds)
    # compute recall for each iou threshold
    for i, t in enumerate(thresholds):
        recalls[i] = (gt_overlaps >= t).float().sum() / float(num_pos)
    # ar = 2 * np.trapz(recalls, thresholds)
    ar = recalls.mean()
    return {
        "ar": ar,
        "recalls": recalls,
        "thresholds": thresholds,
        "gt_overlaps": gt_overlaps,
        "num_pos": num_pos,
    }


def _evaluate_predictions_on_coco(coco_gt, coco_results, iou_type, kpt_oks_sigmas=None):
    """
    Evaluate the coco results using COCOEval API.
    """
    assert len(coco_results) > 0
    if iou_type == "segm":
        coco_results = copy.deepcopy(coco_results)
        # When evaluating mask AP, if the results contain bbox, cocoapi will
        # use the box area as the area of the instance, instead of the mask area.
        # This leads to a different definition of small/medium/large.
        # We remove the bbox field to let mask AP use mask area.
        for c in coco_results:
            c.pop("bbox", None)
    coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    # Use the COCO default keypoint OKS sigmas unless overrides are specified
    if kpt_oks_sigmas:
        coco_eval.params.kpt_oks_sigmas = np.array(kpt_oks_sigmas)

    if iou_type == "keypoints":
        num_keypoints = len(coco_results[0]["keypoints"]) // 3
        assert len(coco_eval.params.kpt_oks_sigmas) == num_keypoints, (
            "[COCOEvaluator] The length of cfg.TEST.KEYPOINT_OKS_SIGMAS (default: 17) "
            "must be equal to the number of keypoints. However the prediction has {} "
            "keypoints! For more information please refer to "
            "http://cocodataset.org/#keypoints-eval.".format(num_keypoints)
        )

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval