# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Modified by Seunghyeok Back (GIST), 2021. 

import torch
from torch import nn
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from detectron2.modeling.roi_heads import (
    ROI_HEADS_REGISTRY,
)
from detectron2.structures import Boxes, Instances, pairwise_iou, ImageList
from detectron2.utils.events import get_event_storage
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.sampling import subsample_labels
from detectron2.layers import ShapeSpec, Conv2d, ConvTranspose2d
from detectron2.modeling.proposal_generator.proposal_utils import add_ground_truth_to_proposals
from detectron2.modeling.box_regression import Box2BoxTransform
import fvcore.nn.weight_init as weight_init

from torch.nn import functional as F

from detectron2.layers import paste_masks_in_image
from .mask_heads import *
from .faster_rcnn import FastRCNNOutputLayers, FastRCNNOutputs
from .bbox_pooler import BBOXROIPooler
from .pooler import ROIPooler
from .box_head import build_box_head
from .order_head import build_order_recovery_head
from detectron2.utils.events import get_event_storage

def select_foreground_proposals(proposals, bg_label):
    
    """
    Given a list of N Instances (for N images), each containing a `gt_classes` field,
    return a list of Instances that contain only instances with `gt_classes != -1 &&
    gt_classes != bg_label`.
    Args:
        proposals (list[Instances]): A list of N Instances, where N is the number of
            images in the batch.
        bg_label: label index of background class.
    Returns:
        list[Instances]: N Instances, each contains only the selected foreground instances.
        list[Tensor]: N boolean vector, correspond to the selection mask of
            each Instances object. True for selected instances.
    """
    assert isinstance(proposals, (list, tuple))
    assert isinstance(proposals[0], Instances)
    assert proposals[0].has("gt_classes")
    fg_proposals = []
    fg_selection_masks = []
    fg_indexes = []
    for proposals_per_image in proposals:
        gt_classes = proposals_per_image.gt_classes
        fg_selection_mask = (gt_classes != -1) & (gt_classes != bg_label)
        fg_idxs = fg_selection_mask.nonzero().squeeze(1)
        fg_proposals.append(proposals_per_image[fg_idxs])
        fg_selection_masks.append(fg_selection_mask)
        fg_indexes.append(fg_idxs)

    return fg_proposals, fg_selection_masks, fg_indexes


class ROIHeads(nn.Module):
    """
    ROIHeads perform all per-region computation in an R-CNN.
    It contains logic of cropping the regions, extract per-region features,
    and make per-region predictions.
    It can have many variants, implemented as subclasses of this class.
    """

    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super(ROIHeads, self).__init__()

        # fmt: off
        self.batch_size_per_image     = cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE
        self.positive_sample_fraction = cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION
        self.test_score_thresh        = cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST
        self.test_nms_thresh          = cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST
        self.test_detections_per_img  = cfg.TEST.DETECTIONS_PER_IMAGE
        self.in_features              = cfg.MODEL.ROI_HEADS.IN_FEATURES
        self.num_classes              = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.proposal_append_gt       = cfg.MODEL.ROI_HEADS.PROPOSAL_APPEND_GT
        self.feature_strides          = {k: v.stride for k, v in input_shape.items()}
        self.feature_channels         = {k: v.channels for k, v in input_shape.items()}
        
        # fmt: on

        # Matcher to assign box proposals to gt boxes
        self.proposal_matcher = Matcher(
            cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS,
            cfg.MODEL.ROI_HEADS.IOU_LABELS,
            allow_low_quality_matches=False,
        )

    def _sample_proposals(self, matched_idxs, matched_labels, gt_classes):
        """
        Based on the matching between N proposals and M groundtruth,
        sample the proposals and set their classification labels.
        Args:
            matched_idxs (Tensor): a vector of length N, each is the best-matched
                gt index in [0, M) for each proposal.
            matched_labels (Tensor): a vector of length N, the matcher's label
                (one of cfg.MODEL.ROI_HEADS.IOU_LABELS) for each proposal.
            gt_classes (Tensor): a vector of length M.
        Returns:
            Tensor: a vector of indices of sampled proposals. Each is in [0, N).
            Tensor: a vector of the same length, the classification label for
                each sampled proposal. Each sample is labeled as either a category in
                [0, num_classes) or the background (num_classes).
        """
        has_gt = gt_classes.numel() > 0
        # Get the corresponding GT for each proposal
        if has_gt:
            gt_classes = gt_classes[matched_idxs]
            # Label unmatched proposals (0 label from matcher) as background (label=num_classes)
            gt_classes[matched_labels == 0] = self.num_classes
            # Label ignore proposals (-1 label)
            gt_classes[matched_labels == -1] = -1
        else:
            gt_classes = torch.zeros_like(matched_idxs) + self.num_classes

        sampled_fg_idxs, sampled_bg_idxs = subsample_labels(
            gt_classes, self.batch_size_per_image, self.positive_sample_fraction, self.num_classes
        )

        sampled_idxs = torch.cat([sampled_fg_idxs, sampled_bg_idxs], dim=0)
        return sampled_idxs, gt_classes[sampled_idxs]

    @torch.no_grad()
    def label_and_sample_proposals(self, proposals, targets):
        """
        Prepare some proposals to be used to train the ROI heads.
        It performs box matching between `proposals` and `targets`, and assigns
        training labels to the proposals.
        It returns ``self.batch_size_per_image`` random samples from proposals and groundtruth
        boxes, with a fraction of positives that is no larger than
        ``self.positive_sample_fraction``.
        Args:
            See :meth:`ROIHeads.forward`
        Returns:
            list[Instances]:
                length `N` list of `Instances`s containing the proposals
                sampled for training. Each `Instances` has the following fields:
                - proposal_boxes: the proposal boxes
                - gt_boxes: the ground-truth box that the proposal is assigned to
                  (this is only meaningful if the proposal has a label > 0; if label = 0
                  then the ground-truth box is random)
                Other fields such as "gt_classes", "gt_masks", that's included in `targets`.
        """
        # ywlee for using targets.gt_classes
        # in add_ground_truth_to_proposal()
        gt_boxes = [x.gt_boxes for x in targets]

        # Augment proposals with ground-truth boxes.
        # In the case of learned proposals (e.g., RPN), when training starts
        # the proposals will be low quality due to random initialization.
        # It's possible that none of these initial
        # proposals have high enough overlap with the gt objects to be used
        # as positive examples for the second stage components (box head,
        # cls head, mask head). Adding the gt boxes to the set of proposals
        # ensures that the second stage components will have some positive
        # examples from the start of training. For RPN, this augmentation improves
        # convergence and empirically improves box AP on COCO by about 0.5
        # points (under one tested configuration).
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(gt_boxes, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            # Set target attributes of the sampled proposals:
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            # We index all the attributes of targets that start with "gt_"
            # and have not been added to proposals yet (="gt_classes").
            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                # NOTE: here the indexing waste some compute, because heads
                # like masks, keypoints, etc, will filter the proposals again,
                # (by foreground/background, or number of keypoints in the image, etc)
                # so we essentially index the data twice.
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            else:
                
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros((len(sampled_idxs), 4))
                )
                proposals_per_image.gt_boxes = gt_boxes

            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        # Log the number of fg/bg samples that are selected for training ROI heads
        storage = get_event_storage()
        storage.put_scalar("roi_head/num_fg_samples", np.mean(num_fg_samples))
        storage.put_scalar("roi_head/num_bg_samples", np.mean(num_bg_samples))

        return proposals_with_gt

    def forward(self, images, features, proposals, targets=None):
        """
        Args:
            images (ImageList):
            features (dict[str: Tensor]): input data as a mapping from feature
                map name to tensor. Axis 0 represents the number of images `N` in
                the input data; axes 1-3 are channels, height, and width, which may
                vary between feature maps (e.g., if a feature pyramid is used).
            proposals (list[Instances]): length `N` list of `Instances`s. The i-th
                `Instances` contains object proposals for the i-th input image,
                with fields "proposal_boxes" and "objectness_logits".
            targets (list[Instances], optional): length `N` list of `Instances`s. The i-th
                `Instances` contains the ground-truth per-instance annotations
                for the i-th input image.  Specify `targets` during training only.
                It may have the following fields:
                - gt_boxes: the bounding box of each instance.
                - gt_classes: the label for each instance with a category ranging in [0, #class].
                - gt_masks: PolygonMasks or BitMasks, the ground-truth masks of each instance.
                - gt_keypoints: NxKx3, the groud-truth keypoints for each instance.
        Returns:
            results (list[Instances]): length `N` list of `Instances`s containing the
            detected instances. Returned during inference only; may be [] during training.
            losses (dict[str->Tensor]):
            mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        raise NotImplementedError()


@ROI_HEADS_REGISTRY.register()
class ORCNNROIHeads(ROIHeads):
    """
    It's "standard" in a sense that there is no ROI transform sharing
    or feature sharing between tasks.
    The cropped rois go to separate branches  masks directly.
    This way, it is easier to make separate abstractions for different branches.
    This class is used by most models, such as FPN and C5.
    To implement more models, you can subclass it and implement a different
    :meth:`forward()` or a head.
    """

    def __init__(self, cfg, input_shape):
        super(ORCNNROIHeads, self).__init__(cfg, input_shape)

        self.amodal = cfg.INPUT.AMODAL
        # ORCNN
        self.cross_check = cfg.MODEL.CROSS_CHECK
        # ASN
        self.MLC = cfg.MODEL.MULTI_LEVEL_CODING
        self.occ_cls_at_box = cfg.MODEL.OCC_CLS_AT_BOX
        # UOAIS-Net 
        self.occ_cls_at_mask = cfg.MODEL.OCC_CLS_AT_MASK
        self.hom = cfg.MODEL.HIERARCHCIAL_OCCLUSION_MODELING
        self.guidance_type = cfg.MODEL.GUIDANCE_TYPE
        self.prediction_order = cfg.MODEL.PREDICTION_ORDER
        self.no_dense_guidance = cfg.MODEL.NO_DENSE_GUIDANCE
                
        self._init_box_head(cfg)
        self._init_amodal_mask_head(cfg)
        self._init_visible_mask_head(cfg)

        if self.MLC:
            self._init_mlc_layer(cfg)
        self.edge_detection = cfg.MODEL.EDGE_DETECTION 
        if self.edge_detection:
            self._init_edge_detection_head(cfg)
        if self.occ_cls_at_mask:
            self._init_occ_cls_mask_head(cfg)

        self.order_recovery = cfg.MODEL.ORDER_RECOVERY_ON
        if self.order_recovery:
            self._init_order_recovery_head(cfg)
            self.order_criterion = nn.MSELoss(reduction='none')
            # self.order_criterion = nn.BCELoss()

    def _init_box_head(self, cfg):
        # fmt: off
        pooler_resolution        = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales            = tuple(1.0 / self.feature_strides[k] for k in self.in_features)
        sampling_ratio           = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type              = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        self.train_on_pred_boxes = cfg.MODEL.ROI_BOX_HEAD.TRAIN_ON_PRED_BOXES
        self.cls_agnostic_bbox_reg = cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG

        # fmt: on

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [self.feature_channels[f] for f in self.in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        self.box_pooler = BBOXROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
        # They are used together so the "box predictor" layers should be part of the "box head".
        # New subclasses of ROIHeads do not need "box predictor"s.
        self.box_head = build_box_head(
            cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
        )
        self.box_predictor = FastRCNNOutputLayers(
            self.box_head._output_size, self.num_classes, self.cls_agnostic_bbox_reg

        )
        if self.occ_cls_at_box:
            self.occ_cls_box_head = build_box_head(
                cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
            )
            self.occ_cls_box_predictor = FastRCNNOutputLayers(
                self.occ_cls_box_head._output_size, 1, True
            )
        self.box2box_transform = Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS)
        self.smooth_l1_beta           = cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA

    def _init_order_recovery_head(self, cfg):

        in_channels = 5
        # in_channels = [self.feature_channels[f] for f in self.in_features][0]
        pooler_resolution = cfg.MODEL.ROI_ORDER_HEAD.POOLER_RESOLUTION

        self.order_recovery_head = build_order_recovery_head(
            cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
        )


    def _init_amodal_mask_head(self, cfg):
        # fmt: off

        pooler_resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / self.feature_strides[k] for k in self.in_features)
        sampling_ratio    = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_MASK_HEAD.POOLER_TYPE

        # fmt: on
        in_channels = [self.feature_channels[f] for f in self.in_features][0]

        self.mask_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        self.amodal_mask_head = build_amodal_mask_head(
            cfg, ShapeSpec(channels=in_channels, width=pooler_resolution, height=pooler_resolution)
        )

    def _init_visible_mask_head(self, cfg):
        # fmt: off
        pooler_resolution = cfg.MODEL.ROI_VISIBLE_MASK_HEAD.POOLER_RESOLUTION
        in_channels = [self.feature_channels[f] for f in self.in_features][0]
        self.visible_mask_head = build_visible_mask_head(
            cfg, ShapeSpec(channels=in_channels, width=pooler_resolution, height=pooler_resolution)
        )
        
    def _init_occ_cls_mask_head(self, cfg):
        # fmt: off
        pooler_resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        in_channels = [self.feature_channels[f] for f in self.in_features][0]
        self.occ_cls_mask_head = OCCCLSMaskHead(
            cfg, ShapeSpec(channels=in_channels, width=pooler_resolution, height=pooler_resolution)
        )
    
        
    def _init_mlc_layer(self, cfg):
        
        self.mlc_layers = []
        in_channels = [self.feature_channels[f] for f in self.in_features][0]
        device = self.box_head.conv1.weight.device
        n_feature = 2 if self.occ_cls_at_box else 1
        self.mlc_layers.append(ConvTranspose2d(n_feature*in_channels, in_channels, 2, 2, 0).to(device=device))
        # if self.occ_cls_at_box:
        self.mlc_layers.append(Conv2d(in_channels, in_channels, 3, 1, 1, activation=nn.ReLU()).to(device=device))
        self.mlc_layers.append(Conv2d(in_channels, in_channels, 3, 1, 1, activation=nn.ReLU()).to(device=device))
        for i, layer in enumerate(self.mlc_layers):
            weight_init.c2_msra_fill(layer)
            self.add_module("extraction_mlc_layer{}".format(i), layer)


    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
        gt_rel_mat: List[List] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        See :class:`ROIHeads.forward`.
        """
        if self.training:
                proposals = self.label_and_sample_proposals(proposals, targets)
        # del targets

        if self.training:
            features_list = [features[f] for f in self.in_features]
            losses, box_head_features = self._forward_box(features_list, proposals)
            if self.order_recovery:
                amodal_vis_occ_losses, pred_mask_logits, pred_gt_boxes = self._forward_masks(features, proposals, box_head_features)
                order_recovery_losses = self._forward_order(images, pred_mask_logits, pred_gt_boxes, targets, gt_rel_mat)
                del gt_rel_mat, pred_mask_logits, pred_gt_boxes, targets
                torch.cuda.empty_cache()
                if order_recovery_losses:
                    losses.update({'order_recovery_loss': order_recovery_losses})
            else:
                amodal_vis_occ_losses = self._forward_masks(features, proposals, box_head_features)
            losses.update(amodal_vis_occ_losses)
            
            return proposals, losses

        else:
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            features_list = [features[f] for f in self.in_features]
            pred_instances, box_head_features = self._forward_box(features_list, proposals)
            if self.order_recovery:
                pred_instances, pred_mask_logits, _ = self._forward_masks(features, pred_instances, box_head_features)
                pred_rel_mat = self._inference_order(images, pred_instances, targets, gt_rel_mat)
            else:
                pred_instances = self._forward_masks(features, pred_instances, box_head_features)
            return pred_instances, pred_rel_mat
        
    def _inference_order(self, images, pred_instances, targets, gt_rel_mat):
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        image = images[0]
        draw_image = image.permute(1,2,0).cpu().numpy()
        pred_boxes = pred_instances[0].pred_boxes.tensor            # 4, 4
        gt_boxes = targets[0].gt_boxes.tensor                       # 3, 4
        pred_masks = pred_instances[0].pred_visible_masks           # 4,1,28,28
        gt_masks = targets[0].gt_visible_masks.tensor               # 3,800,800

        ## make pred_mask to image size
        paste_masks = []
        for i, mask in enumerate(pred_masks):
            mask = mask.cpu()
            box = pred_boxes[i].cpu()
            paste_masks.append(paste_masks_in_image(mask, box.unsqueeze(0), image.shape[-2:])[0])
        paste_masks = torch.stack(paste_masks, dim=0)

        ## match pred-gt masks
        matched_pred_idxes_of_gt_object = [None for _ in range(len(gt_masks))]
        matched_pred_ious_of_gt_object = [-1 for _ in range(len(gt_masks))]
        for pred_idx, pred in enumerate(paste_masks):
            pred = pred.cpu().numpy()
            max_iou, max_iou_gt_idx = -1, -1
            for gt_idx, gt in enumerate(gt_masks):
                gt = gt.cpu().numpy()
                intersection = np.logical_and(pred, gt)
                union = np.logical_or(pred, gt)
                iou_score = np.sum(intersection) / np.sum(union)
                # print(iou_score)
                if iou_score > max_iou:
                    max_iou = iou_score
                    max_iou_gt_idx = gt_idx
            if matched_pred_idxes_of_gt_object[max_iou_gt_idx] is None:
                matched_pred_idxes_of_gt_object[max_iou_gt_idx] = pred_idx
                matched_pred_ious_of_gt_object[max_iou_gt_idx] = max_iou
            elif max_iou > matched_pred_ious_of_gt_object[max_iou_gt_idx]:
                matched_pred_idxes_of_gt_object[max_iou_gt_idx] = pred_idx
                matched_pred_ious_of_gt_object[max_iou_gt_idx] = max_iou

        
        ## ordering recovery with pred masks
        pred_rel_mat = np.zeros((len(gt_masks), len(gt_masks)))
        for i in range(len(gt_masks)):
            for j in range(len(gt_masks)):
                if i == j: continue
                if matched_pred_idxes_of_gt_object[i] is None or matched_pred_idxes_of_gt_object[j] is None:
                    continue
                
                inputs1 = torch.cat([paste_masks[matched_pred_idxes_of_gt_object[i]].to(image.get_device()).unsqueeze(0),  \
                                     paste_masks[matched_pred_idxes_of_gt_object[j]].to(image.get_device()).unsqueeze(0),  \
                                     image], dim=0)
                inputs2 = torch.cat([paste_masks[matched_pred_idxes_of_gt_object[j]].to(image.get_device()).unsqueeze(0),  \
                                     paste_masks[matched_pred_idxes_of_gt_object[i]].to(image.get_device()).unsqueeze(0),  \
                                     image], dim=0)
                
                output1 = torch.sigmoid(self.order_recovery_head(inputs1.unsqueeze(0)))
                output2 = torch.sigmoid(self.order_recovery_head(inputs2.unsqueeze(0)))

                if output1[0][1]: pred_rel_mat[i][j] = -1
                if output2[0][1]: pred_rel_mat[j][i] = -1
        # print(np.sum(pred_rel_mat))
        # if np.sum(pred_rel_mat):
        #     print(matched_pred_idxes_of_gt_object)
        #     print('pred_rel_mat', pred_rel_mat)
        # if len(gt_masks)!= 3:
        #     print('gt_rel_mat', gt_rel_mat[0])
        # print()
        return pred_rel_mat
    

        
    def _forward_order(self, images: ImageList, pred_mask_logits: torch.Tensor, pred_gt_boxes: torch.Tensor, \
                        targets:List[Instances], gt_rel_mat: List[List]) -> torch.Tensor:
        from detectron2.layers import paste_masks_in_image
        """
        Forward logic of the order recovery branch.
        Args:
            images (ImageList): input images
            pred_visible_mask_logits (Tensor): A tensor of shape (R, H, W) or (R, 1, H, W)
                or (R, 2, H, W) for class-specific or class-agnostic, where R is the total number
                of predicted boxes for all images, H and W are height and width of the feature maps.
        Returns:
            Order Recovery Loss
        """
        mask_idx = 0
        loss = 0.
        ## pred-gt matching
        ## mean of pred masks
        for B, image in enumerate(images):
            # target = targets[B]
            # pred_logits_per_image = pred_mask_logits[mask_idx:mask_idx+len(pred_gt_boxes[B]['pred'])]

            # pred_boxes_per_image = pred_gt_boxes[B]['pred']
            # gt_boxes_per_image = pred_gt_boxes[B]['gt']

            # target_gt_boxes = target.gt_boxes.tensor
            # pred_indexes = [[] for _ in range(len(target_gt_boxes))]
            # for i, gt_box in enumerate(gt_boxes_per_image):
            #     for j, target_gt in enumerate(target_gt_boxes):
            #         if torch.all(torch.eq(gt_box, target_gt)):
            #             pred_indexes[j].append(i)
            #             break
            #     else:
            #         print('no gt box matched')

            # pred_masks_per_gt = []
            # for i, indexes in enumerate(pred_indexes):
            #     if len(indexes) == 0:
            #         pred_masks_per_gt.append(None)
            #         continue

            #     ## mean of image sized pred masks
            #     # pred_logits_per_gt = pred_logits_per_image[indexes].mean(dim=0)
            #     # pred_masks_per_gt.append(torch.sigmoid(pred_logits_per_gt) > 0.5)

            #     pred_logits_per_gt = pred_logits_per_image[indexes]
            #     pred_boxes_per_gt = pred_boxes_per_image[indexes]
            #     ## paste pred logit masks to image
            #     paste_mask_per_gt = paste_masks_in_image(pred_logits_per_gt, pred_boxes_per_gt, image.shape[-2:]).sum(dim=0)
            #     # so = nn.Sigmoid(dim=None)
            #     # paste_mask_per_gt = so((paste_mask_per_gt-0.5)*1000)
            #     paste_mask_per_gt = (paste_mask_per_gt >= len(indexes)/2)
                
            #     # pred_masks_per_gt.append(paste_masks_in_image(pred_logits_per_gt, pred_boxes_per_gt, image.shape[-2:]))
            #     pred_masks_per_gt.append(paste_mask_per_gt)

            #     # paste_masks_in_image()
            
            inputs1, inputs2, gt_order1, gt_order2 = [], [], [], []
            # for i, pred_mask_i in enumerate(pred_masks_per_gt):
            #     for j, pred_mask_j in enumerate(pred_masks_per_gt):
            for i, pred_mask_i in enumerate(targets[B].gt_masks):
                for j, pred_mask_j in enumerate(targets[B].gt_masks):
                    if i == j: continue
                    if pred_mask_i is None or pred_mask_j is None: continue

                    img, mask_i, mask_j = self._preprocess_order(image, pred_mask_i.unsqueeze(0), pred_mask_j.unsqueeze(0))
                    inputs1.append(torch.cat([mask_i, mask_j, img], dim=0).unsqueeze(0))
                    inputs2.append(torch.cat([mask_j, mask_i, img], dim=0).unsqueeze(0))


                    if gt_rel_mat[B][i][j] == -1:       ## i is occluded by j
                        gt_order1.append(torch.FloatTensor([[1., 0.]]))    ## does i occlude j? -> no -> 0
                        gt_order2.append(torch.FloatTensor([[0., 1.]]))    ## does j occlude i? -> yes -> 1
                    elif gt_rel_mat[B][j][i] == -1:     ## j is occluded by i
                        gt_order1.append(torch.FloatTensor([[0., 1.]]))    ## does i occlude j? -> yes -> 1
                        gt_order2.append(torch.FloatTensor([[1., 0.]]))    ## does j occlude i? -> no -> 0
                    elif gt_rel_mat[B][i][j] == 0:      ## none
                        gt_order1.append(torch.FloatTensor([[1., 0.]]))    ## does i occlude j? -> no -> 0
                        gt_order2.append(torch.FloatTensor([[1., 0.]]))    ## does j occlude i? -> no -> 0
                    
                    if len(gt_order1) >= 30:
                        gt_order1 = torch.cat(gt_order1, dim=0).to(device=images.device)
                        gt_order2 = torch.cat(gt_order2, dim=0).to(device=images.device)
                        # output1 = torch.sigmoid(self.order_recovery_head(torch.cat(inputs1, dim=0)))
                        # output2 = torch.sigmoid(self.order_recovery_head(torch.cat(inputs2, dim=0)))
                        so = nn.Softmax(dim=-1)
                        output1 = so(self.order_recovery_head(torch.cat(inputs1, dim=0)))
                        output2 = so(self.order_recovery_head(torch.cat(inputs2, dim=0)))

                        # # ## weighted loss
                        # mask1 = mask2 = torch.ones(gt_order1.shape[0]).to(device=images.device)
                        # for k in range(gt_order1.shape[0]):
                        #     if gt_order1[k, 1] > 0:
                        #         mask1[k] = 5
                        #     if gt_order2[k, 1] > 0:
                        #         mask2[k] = 5
                        # loss += torch.mean(mask1 * self.order_criterion(output1, gt_order1).mean() + \
                        #                    mask2 * self.order_criterion(output2, gt_order2).mean())
                        
                        # ## unweighted loss
                        loss += (self.order_criterion(output1, gt_order1).mean() + self.order_criterion(output2, gt_order2).mean())
            
                        inputs1, gt_order1 = [], []
                        inputs2, gt_order2 = [], []

            if len(gt_order1):
                gt_order1 = torch.cat(gt_order1, dim=0).to(device=images.device)
                gt_order2 = torch.cat(gt_order2, dim=0).to(device=images.device)
                output1 = torch.sigmoid(self.order_recovery_head(torch.cat(inputs1, dim=0)))
                output2 = torch.sigmoid(self.order_recovery_head(torch.cat(inputs2, dim=0)))

                # # ## weighted loss
                # mask1 = mask2 = torch.ones(gt_order1.shape[0]).to(device=images.device)
                # for k in range(gt_order1.shape[0]):
                #     if gt_order1[k, 1] > 0:
                #         mask1[k] = 5
                #     if gt_order2[k, 1] > 0:
                #         mask2[k] = 5

                # loss += torch.mean(mask1 * self.order_criterion(output1, gt_order1).mean() + \
                #                     mask2 * self.order_criterion(output2, gt_order2).mean())
                
                # ## unweighted loss
                loss += (self.order_criterion(output1, gt_order1).mean() + self.order_criterion(output2, gt_order2).mean())

        # del pred_logits_per_image, pred_boxes_per_image, gt_boxes_per_image, target_gt_boxes, \
        #     pred_indexes, pred_logits_per_gt, pred_masks_per_gt, gt_order1, gt_order2
        del gt_order1, gt_order2

        return loss

    def _preprocess_order(self, image, mask_i, mask_j):
        import torchvision.transforms as T
        from torchvision.transforms.functional import pad
        from detectron2.layers import paste_masks_in_image

        ## pad image/mask    (480,640) -> (640,640)
        # # COCOA   (480,640)
        # image = pad(image, (0, 80, 0, 80), 0, 'constant')
        # # mask_i = pad(mask_i, (0, 80, 0, 80), 0, 'constant')
        # # mask_j = pad(mask_j, (0, 80, 0, 80), 0, 'constant')
        
        ## no padding with MetaGraspNet ##      (800,800)

        ## resize image/mask (640,640) 
        image = T.Resize((64,64))(image)
        mask_i = T.Resize((64,64))(mask_i)
        mask_j = T.Resize((64,64))(mask_j)



        return image, mask_i, mask_j


    def _forward_box(self, features: List[torch.Tensor], proposals: List[Instances]
                    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        """
        Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
            the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.
        Args:
            features (list[Tensor]): #level input features for box prediction
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".
        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_head_features, first_head_features = self.box_head(box_features)
        pred_class_logits, pred_proposal_deltas = self.box_predictor(box_head_features)

        if self.occ_cls_at_box:
            occ_cls_head_features, occ_cls_first_head_features = self.occ_cls_box_head(box_features)
            pred_occ_cls_logits, _ = self.occ_cls_box_predictor(occ_cls_head_features)
        else:
            pred_occ_cls_logits = None

        del box_features

        outputs = FastRCNNOutputs(
            self.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            self.smooth_l1_beta,
            pred_occ_cls_logits, 
        )

        if self.MLC:
            if self.occ_cls_at_box:
                box_head_features = torch.cat([first_head_features, occ_cls_first_head_features], 1)
            else:
                box_head_features = first_head_features
            for layer in self.mlc_layers:
                box_head_features = layer(box_head_features)
        else:
            box_head_features = None

        if self.training:
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = outputs.predict_boxes_for_gt_classes()
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            return outputs.losses(), box_head_features
        else:
            pred_instances, index = outputs.inference(
                self.test_score_thresh, self.test_nms_thresh, self.test_detections_per_img
            )
            if box_head_features is not None:
                box_head_features = box_head_features[index]
            return pred_instances, box_head_features

    def _forward_single_mask(self, pred_target, features, mask_features_list, 
                                    proposals, box_head_features, losses, logits):

        if self.hom and self.guidance_type == "concat":
            if self.no_dense_guidance:
                input_features = torch.cat([features] + mask_features_list[-1:], 1)
            else:
                input_features = torch.cat([features] + mask_features_list, 1)
        elif self.hom and self.guidance_type == "add":
            input_features = features
            if self.no_dense_guidance:
                input_features = input_features + mask_features_list[-1]
            else:
                for mask_features in mask_features_list:
                    input_features = input_features + mask_features
        else:
            input_features = features

        if pred_target == "V":
            mask_logits, output_features  = self.visible_mask_head(input_features, proposals, box_head_features)
            loss, pred_mask_logits = mask_rcnn_loss(mask_logits, proposals, "gt_visible_masks")
            losses["loss_visible_mask"] = loss
            logits["visible"] = mask_logits
            
        elif pred_target == "A":
            mask_logits, output_features = self.amodal_mask_head(input_features, proposals, box_head_features)
            loss = mask_rcnn_loss(mask_logits, proposals, "gt_masks")
            losses["loss_amodal_mask"] = loss
            logits["amodal"] = mask_logits

        elif pred_target == "O":
            occ_cls_logits, output_features = self.occ_cls_mask_head(input_features, proposals, box_head_features)
            gt_occludeds = []
            for p in proposals:
                if p.has("gt_occluded_rate"):
                    gt_occludeds.append(p.gt_occluded_rate >= 0.05)
            gt_occludeds = cat(gt_occludeds, dim=0).to(torch.int64)
            n_occ, n_gt = torch.sum(gt_occludeds), gt_occludeds.shape[0]
            n_noocc = n_gt - n_occ
            loss = F.cross_entropy(occ_cls_logits, gt_occludeds, reduction="mean", 
                                weight=torch.Tensor([1, n_noocc/n_occ]).to(device=gt_occludeds.device))
            losses["loss_occ_cls"] = loss
        
        if self.order_recovery and pred_target == "V":
            pred_gt_boxes = []
            for p in proposals:
                pred_gt_boxes.append({'pred': p.proposal_boxes.tensor,
                                      'gt': p.gt_boxes.tensor})
            return losses, output_features, logits, pred_mask_logits, pred_gt_boxes
            
        return losses, output_features, logits

    def _inference_single_mask(self, pred_target, features, mask_features_list, 
                                    instances, box_head_features, logits):

        if self.hom and self.guidance_type == "concat":
            if self.no_dense_guidance:
                input_features = torch.cat([features] + mask_features_list[-1:], 1)
            else:
                input_features = torch.cat([features] + mask_features_list, 1)
        elif self.hom and self.guidance_type == "add":
            input_features = features
            if self.no_dense_guidance:
                input_features = input_features + mask_features_list[-1]
            else:
                for mask_features in mask_features_list:
                    input_features = input_features + mask_features
        else:
            input_features = features

        if pred_target == "V":
            mask_logits, output_features = self.visible_mask_head(input_features, instances, box_head_features)
            mask_rcnn_inference(mask_logits, instances, "pred_visible_masks")
            logits["visible"] = mask_logits
            
        elif pred_target == "A":
            mask_logits, output_features = self.amodal_mask_head(input_features, instances, box_head_features)
            mask_rcnn_inference(mask_logits, instances, "pred_masks")
            logits["amodal"] = mask_logits

        elif pred_target == "O":
            occ_cls_logits, output_features = self.occ_cls_mask_head(input_features, instances, box_head_features)
            occ_probs = F.softmax(occ_cls_logits, dim=-1)
            # occ_probs = occ_probs.split([len(p) for p in instances], dim=0)
            if len(occ_probs) > 0:
                occ_scores, occ_preds = occ_probs.max(-1)
                occ_scores = occ_scores.unsqueeze(0)
                occ_preds = occ_preds.unsqueeze(0)
                for occ_score, occ_pred, instance in zip(occ_scores, occ_preds, instances):
                    instance.set("pred_occlusions", occ_pred)
                    instance.set("pred_occlusion_scores", occ_score)
            
        return instances, output_features, logits

    def _forward_masks(self, features, instances, box_head_features):

        features = [features[f] for f in self.in_features]
#
        if self.training:
            # get MLC features from box branch
            proposals, _, fg_indexes = select_foreground_proposals(instances, self.num_classes)
            if self.MLC:
                fg_box_head_features = []
                for fg_idx in fg_indexes:
                    fg_box_head_features.append(box_head_features[fg_idx])
                box_head_features = torch.cat(fg_box_head_features, 0)
            else:
                box_head_features = None
            features = self.mask_pooler(features, proposals, self.training)
            
            losses = {}
            logits = {}
            mask_features_list = []
            for pred_target in self.prediction_order:
                if self.order_recovery and pred_target == "V":
                    losses, mask_features, logits, pred_mask_logits, pred_gt_boxes = \
                        self._forward_single_mask(pred_target, features, mask_features_list,\
                                                proposals, box_head_features, losses, logits)
                else:
                    losses, mask_features, logits = \
                        self._forward_single_mask(pred_target, features, mask_features_list,\
                                                proposals, box_head_features, losses, logits)
                mask_features_list.append(mask_features)
            
            # ORCNN loss
            if self.cross_check:
                occ_mask_logits = logits["amodal"] - F.relu(logits["visible"])
                occ_mask_loss = occlusion_mask_rcnn_loss(occ_mask_logits, proposals, False)
                losses["loss_occlusion_mask"] = occ_mask_loss

            if self.order_recovery:
                return losses, pred_mask_logits, pred_gt_boxes
            return losses
        else:
            # pred_boxes = [x.pred_boxes for x in instances]
            features = self.mask_pooler(features, instances)
            logits = {}
            mask_features_list = []
            for pred_target in self.prediction_order:
                losses, mask_features, logits = \
                    self._inference_single_mask(pred_target, features, mask_features_list,\
                                            instances, box_head_features, logits)
                mask_features_list.append(mask_features)

            if self.cross_check :
                occ_mask_logits = logits["amodal"] - F.relu(logits["visible"])
                occlusion_mask_rcnn_inference(occ_mask_logits, instances)

            if self.order_recovery:
                return instances, logits['visible'], {}
                return instances, pred_mask_logits, pred_gt_boxes
            return instances
