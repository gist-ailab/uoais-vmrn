


"""
fig, ax = plt.subplots()
ax.imshow(im)
plt.axis('off')
for box in pred_boxes:
    print(box)
    box = box.cpu()
    rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
plt.show(); plt.savefig('vis/graduation/val_pred_box.png', bbox_inches='tight',transparent=True, pad_inches=0);


"""


"""
for i, mask in enumerate(pred_masks):
    mask = mask.cpu()
    box = pred_boxes[i].cpu()
    paste_mask = paste_masks_in_image(mask, box.unsqueeze(0), image.shape[-2:])[0]
    img = image.cpu()
    print(img.shape, paste_mask.shape)
    img[:,paste_mask==0] = 0
    plt.imshow(img.permute(1,2,0)); plt.savefig(f'vis/graduation/val_paste_mask_mask{i}.png', bbox_inches='tight',transparent=True, pad_inches=0)

"""







