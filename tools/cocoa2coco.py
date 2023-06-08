import os
import cv2
import json
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt

from pycocotools.coco import COCO
from pycocotools import mask as maskUtils


def MaskToRle(mask):
    """
    input:
        mask format : [H, W]
    output: 
        rle
    """
    rle = maskUtils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle['counts'] = rle['counts'].decode('ascii')
    return rle

def RleToMask(rle):
    """
    input:
        rle
    output: 
        mask : [H, W]
    """
    m = maskUtils.decode(rle)
    # print(m.shape)
    # print(np.unique(m))
    # plt.imshow(m)
    # plt.show()
    # plt.savefig('vis/cocoa/mask.png')
    return m

def makeAmodalMask(visible, invisible):
    """
    input:
        visible : [H, W]
        invisible : [H, W]
    output: 
        amodal : [H, W]
    """
    amodal = np.zeros_like(visible)
    amodal[visible > 0] = 1
    amodal[invisible > 0] = 1
    return amodal


def mask_to_bbox_one_instance(mask):
    # Bounding box.
    horizontal_indicies = np.where(np.any(mask, axis=0))[0]
    vertical_indicies = np.where(np.any(mask, axis=1))[0]
    if horizontal_indicies.shape[0]:
        x1, x2 = horizontal_indicies[[0, -1]]
        y1, y2 = vertical_indicies[[0, -1]]
        # x2 and y2 should not be part of the box. Increment by 1.
        x2 += 1
        y2 += 1
    else:
        # No mask for this instance. Might happen due to
        # resizing or cropping. Set bbox to zeros
        x1, x2, y1, y2 = 0, 0, 0, 0
    bbox = np.array([x1, y1, x2-x1, y2-y1])     ## XYWH_ABS
    return bbox.astype(np.int16)


def load_segm(segm):
    # segm = anno.get(type, None)
    if isinstance(segm, dict):
        if isinstance(segm["counts"], list):
            # convert to compressed RLE
            segm = maskUtils.frPyobjs(segm, *segm["size"])
    else:
        # filter out invalid polygons (< 3 points)
        segm = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
        if len(segm) == 0:
            num_instances_without_valid_segmentation += 1
            segm = None
    return segm


class COCOA2COCO():
    def __init__(self, file_path):

        with open(file_path, 'r') as f:
            self.json_file = json.load(f)

        self.image_list = self.json_file['images']
        self.annot_list = self.json_file['annotations']
        
        self.image_dict_list = []
        self.annot_dict_list = []
        self.categ_dict_list = [
            { 'id': 0, 'name': '__background__' },
            { 'id': 1, 'name': '__foreground__' },
        ]


    def make_annot(self):
        annot_idx = 0
        max_obj = 0

        for image_idx, (image, annot) in enumerate(zip(self.image_list, self.annot_list)):
            
            # objs_num = annot['size']
            objs_num = len(annot['regions'])
            if annot['size'] != len(annot['regions']):
                print('size error', annot['size'], len(annot['regions']))
            if objs_num >= 50:
                print(objs_num)
            if objs_num > max_obj:
                max_obj = objs_num
            rel_mat = np.zeros((objs_num, objs_num))
            depth_constraint = annot['depth_constraint']
            constraint = depth_constraint.split(',')

            ## calculate relationship matrix
            if len(depth_constraint):
                for c in constraint:
                    occluder, occludee = c.split('-')
                    occluder, occludee = int(occluder)-1, int(occludee)-1
                    # rel_mat[occluder][occludee] = -1
                    # rel_mat[occludee][occluder] = 1
                    rel_mat[occludee][occluder] = -1
                    rel_mat[occluder][occludee] = 1

            # for obj_idx, obj in enumerate(annot['regions']):
            for obj_idx in range(objs_num):
                obj = annot['regions'][obj_idx]
                segm = [obj['segmentation']]

                if 'visible_mask' in obj:
                    visible_rle = obj['visible_mask']
                    visible_mask = RleToMask(visible_rle)

                    invisible_rle = obj['invisible_mask']
                    invisible_mask = RleToMask(invisible_rle)

                    amodal_mask = makeAmodalMask(visible_mask, invisible_mask)
                    amodal_rle = MaskToRle(amodal_mask)
                #     print('v-box', visible_bbox)
                #     print('a-box', amodal_bbox)
                else:
                    rles = maskUtils.frPyObjects(segm, image['height'], image['width'])
                    amodal_rle = maskUtils.merge(rles)
                    amodal_rle['counts'] = amodal_rle['counts'].decode('ascii')
                    amodal_mask = maskUtils.decode(amodal_rle)

                    visible_rle = amodal_rle
                    visible_mask = amodal_mask

                    invisible_mask = np.zeros_like(amodal_mask)
                    invisible_rle = maskUtils.encode(invisible_mask)
                    invisible_rle['counts'] = invisible_rle['counts'].decode('ascii')

                    # plt.imshow(amodal_mask)
                    # plt.show()
                
                amodal_bbox = mask_to_bbox_one_instance(amodal_mask)
                visible_bbox = mask_to_bbox_one_instance(visible_mask)
                
                self.annot_dict_list.append({
                    'id': annot_idx,
                    'image_id': image_idx,
                    'category_id': 0,
                    'iscrowd': 0,
                    'area': obj['area'],
                    'segmentation': amodal_rle,
                    'bbox': amodal_bbox.astype(np.int16).tolist(),
                    'visible_mask': visible_rle,
                    'visible_bbox': visible_bbox.astype(np.int16).tolist(),
                    'occluded_mask': invisible_rle,
                    'occluded_rate': obj['occlude_rate'],
                })

            self.image_dict_list.append({
                'id': image_idx,
                'image_id': image_idx,
                'width': image['width'],
                'height': image['height'],
                'file_name': image['file_name'],
                'rel_mat': rel_mat.tolist(),
            })
            # print(rel_mat)
            # print()

            # if image_idx == 30: break
        print('max: ', max_obj)

    def combine_annot(self):
        self.make_annot()
        print(len(self.image_dict_list))

        return {
            'images': self.image_dict_list,
            'annotations': self.annot_dict_list,
            'categories': self.categ_dict_list
        }


if __name__ == '__main__':
    val_path = '/ailab_mat/dataset/InstaOrder/data/COCO/annotations/COCO_amodal_val2014.json'
    train_path = '/ailab_mat/dataset/InstaOrder/data/COCO/annotations/COCO_amodal_train2014.json'
    
    print('train')
    train_converter = COCOA2COCO(train_path)
    train_annot = train_converter.combine_annot()
    with open('Annotations/cocoa_train.json', 'w') as f:
        json.dump(train_annot, f)

    print('val')
    val_converter = COCOA2COCO(val_path)
    val_annot = val_converter.combine_annot()
    with open('Annotations/cocoa_val.json', 'w') as f:
        json.dump(val_annot, f)


