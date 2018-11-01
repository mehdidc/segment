import numpy as np
import os
import cv2
from glob import glob
from clize import run
from data.base import load_image
import torch
import torch.nn as nn
from skimage.transform import resize
from skimage.filters import threshold_otsu
from skimage.io import imsave

from albumentations import Compose
from albumentations import Resize
from albumentations import Normalize
from albumentations import HorizontalFlip
from albumentations import RandomCrop

from utils import blend, colored

def main(
    *, 
    model_filename='model.th',
    images_folder='coco/val2017', 
    out_folder='test',
    cuda=False
):
    model = torch.load(
        model_filename, 
        map_location=lambda storage, location:storage
    )
    if cuda:
        model.cuda()
    model.eval()
    image_size = 224
    transform = Compose([
        HorizontalFlip(p=0.0),
        Resize(height=image_size, width=image_size, p=1.0),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.),
    ])
    filenames = glob(os.path.join(images_folder, '*.*'))
    for filename in filenames:
        try:
            orig_im =  load_image(filename)
        except Exception:
            continue
        anns = transform(image=orig_im)
        im = anns['image']
        im = im.transpose((2, 0, 1))
        im = torch.from_numpy(im).float()
        im = im.view(1, im.size(0), im.size(1), im.size(2))
        if cuda:
            im = im.cuda()
        with torch.no_grad():
            pred_masks = model(im)

        orig_im = orig_im / 255.
        _, pred = pred_masks.max(dim=1)
        pred_score = nn.Softmax(dim=1)(pred_masks)[:, 1:].sum(dim=1)
        pred_gray = (pred_score[0]).cpu().numpy()
        pred_gray = cv2.medianBlur(pred_gray, 5)
        pred_gray = resize(pred_gray, orig_im.shape[0:2], preserve_range=True)
        pred_mask = pred_gray > threshold_otsu(pred_gray)
        pred_mask = pred_mask.astype('uint8')
        # fill holes in predicted mask
        _, contours, _ = cv2.findContours(
            (pred_mask*255).astype('uint8'),
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE
        )
        # draw bounding boxes from predicted mask
        pred_mask_no_holes = np.zeros_like(pred_mask)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(pred_mask_no_holes, (x, y), (x + w, y + h), (1, 1, 1), cv2.FILLED)
        _, contours, _ = cv2.findContours(
            (pred_mask_no_holes*255).astype('uint8'),
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE
        )
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(orig_im, (x, y), (x + w, y + h), (0, 1, 0), 2)
        #orig_im = blend(orig_im, colored(pred_mask, (0, 1, 0)))
        out_filename = os.path.join(
            out_folder,
            os.path.basename(filename)
        )
        imsave(out_filename, orig_im)
        print(filename)
if __name__ == '__main__':
    run(main)
