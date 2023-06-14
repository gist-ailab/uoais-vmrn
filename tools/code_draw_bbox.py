
"""
import matplotlib.pyplot as plt
import cv2

im = batched_inputs[0]['image'].permute(1,2,0).cpu().numpy()
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
plt.axis('off')
plt.imshow(im); plt.savefig(f'vis/graduation/{img_idx}_img_org.png', bbox_inches='tight',transparent=True, pad_inches=0);


"""

img_idx = 'train1'


"""
plt.axis('off')
plt.imshow(draw_image); plt.savefig(f'vis/graduation/{img_idx}_img.png', bbox_inches='tight',transparent=True, pad_inches=0);

"""



"""
fig, ax = plt.subplots()
ax.imshow(draw_image)
plt.axis('off')
for box in pred_boxes:
    print(box)
    box = box.cpu()
    rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
plt.show(); plt.savefig(f'vis/graduation/{img_idx}_val_pred_box.png', bbox_inches='tight',transparent=True, pad_inches=0);


"""


"""
fig, ax = plt.subplots()
ax.imshow(draw_image)
plt.axis('off')
for i, mask in enumerate(pred_masks):
    mask = mask.cpu()
    box = pred_boxes[i].cpu()
    paste_mask = paste_masks_in_image(mask, box.unsqueeze(0), image.shape[-2:])[0]
    img = image.cpu()
    print(img.shape, paste_mask.shape)
    img[:,paste_mask==0] = 0
    plt.imshow(img.permute(1,2,0)); plt.savefig(f'vis/graduation/{img_idx}_val_paste_mask_mask{i}.png', bbox_inches='tight',transparent=True, pad_inches=0)

"""


## validation on train image
"""
import cv2
import matplotlib.pyplot as plt
import torchvision.transforms as T


image = cv2.imread('/ailab_mat/dataset/MetaGraspNet/dataset_sim/scene14/25_rgb.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = image.permute(2,0,1)
image = T.Resize((800,800))(image)


im = image.permute(1,2,0).cpu().numpy()
plt.axis('off')
plt.imshow(im); plt.savefig(f'vis/graduation/{img_idx}_img_org.png', bbox_inches='tight',transparent=True, pad_inches=0);



"""





