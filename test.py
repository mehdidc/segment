import numpy as np
import os
import cv2
from collections import Counter
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

from bounding_boxes import get_bounding_boxes, nms, match_ordered_boxes
from viz import draw_boxes


def main(
    *, 
    model_filename='model.th',
    images_folder='coco/val2017', 
    out_folder='test',
    cuda=False,
    score_threshold=0.5,
    image_size=224):

    model_dict = torch.load(
        model_filename, 
        map_location=lambda storage, location:storage
    )
    model = model_dict['model']
    transform = model_dict['transform']
    decode_class = model_dict['decode_class']
    rng = np.random.RandomState(42)
    colors = rng.randint(0, 255, size=(model.nb_classes, 3))
    if cuda:
        model.cuda()
    model.eval()
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
        pad = 100
        orig_im = np.pad(orig_im, ((pad, pad), (pad, pad), (0, 0)), mode='constant', constant_values=0)
        pred_gray = (pred_masks[0]).cpu().numpy()
        pred_gray = pred_gray.transpose((1, 2, 0))
        pred_gray = resize(pred_gray, orig_im.shape[0:2], preserve_range=True)
        pred_mask = pred_gray > score_threshold
        pred_mask = pred_mask.astype('uint8')
        
        pred_boxes, pred_class_ids, pred_scores = get_bounding_boxes(pred_mask)
        inds = nms(pred_boxes, pred_scores, thres=0.3)
        pred_boxes = pred_boxes[inds]
        pred_scores = pred_scores[inds]
        pred_class_ids = pred_class_ids[inds]
        pred_classes = [decode_class[class_id] for class_id in pred_class_ids]
        pred_class_ids = pred_class_ids.astype(int)
        draw_boxes(orig_im, pred_boxes, pred_classes, pred_scores, colors[pred_class_ids])
            
        out_filename = os.path.join(
            out_folder,
            os.path.basename(filename)
        )
        imsave(out_filename, orig_im)
        print(filename)


if __name__ == '__main__':
    run(main)
