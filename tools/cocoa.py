import json
import cv2
import matplotlib.pyplot as plt
import numpy as np



val_path = '/ailab_mat/dataset/InstaOrder/data/COCO/annotations/COCO_amodal_val2014.json'

with open(val_path, 'r') as f:
    val_file = json.load(f)

# val_file


# print(val_file)
image_dict = val_file['images']
# file_name, height, width, id (==image_id)
anno_dict_list = val_file['annotations']
# image_id, size (=num_obj)
#  regions
##  segmentation
##  name
##  occlude_rate

# depth_constraint


regions = anno_dict_list[0]['regions']
img_path = '/ailab_mat/dataset/InstaOrder/data/COCO/val2014/' + image_dict[0]['file_name']
print(img_path)
# print(regions[0]['segmentation'])
im = cv2.imread(img_path)
print(im.shape)
print(np.unique(im))
plt.imshow(im)
plt.savefig('/home/heeseon_rho/src/uoais-vmrn/vis/cocoa/image.png')
segm = regions[0]['segmentation']