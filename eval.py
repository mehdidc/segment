from clize import run
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from albumentations import Compose
from albumentations import Resize
from albumentations import Normalize
from albumentations import HorizontalFlip
from albumentations import RandomCrop

from data.coco import get_annotations
from data.base import load_image
from data.base import DetectionDataset
from data.base import load_image

from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.draw import circle_perimeter
from skimage.transform import resize
from skimage.io import imsave
from skimage.filters import threshold_otsu

import cv2

from model import UNet
from bounding_boxes import get_bounding_boxes, nms, match_ordered_boxes
from viz import draw_boxes


def main(
    *, 
    model_filename='model.th',
    images_folder='coco/val2017', 
    annotations_file='coco/annotations/instances_val2017', 
    cuda=False
):
    model_dict = torch.load(
        model_filename, 
        map_location=lambda storage, location:storage)
    model = model_dict['model']
    transform = model_dict['transform']
    if cuda:
        model.cuda()
    model.eval()
    filenames, annotations = get_annotations(
        images_folder=images_folder,
        annotations_file=annotations_file,
    )
    image_size = 224
    dataset = DetectionDataset(
        filenames, 
        annotations, 
        transform=transform,
    )
    dataloader = DataLoader(
        dataset, 
        batch_size=16, 
        shuffle=False,
        num_workers=1,
    )
    rng = np.random.RandomState(42)
    colors = rng.randint(0, 255, size=(model.nb_classes, 3))
    true_positives = 0
    nb_predicted_positives = 0
    nb_positives = 0

    for images, masks, filenames in dataloader:
        if cuda:
            images = images.cuda()
            masks = masks.cuda()
        with torch.no_grad():
            pred_masks = nn.Sigmoid()(model(images))
        # eval
        for idx in range(len(filenames)):
            im = load_image(filenames[idx]) 
            im = im.astype('uint8')
            true_mask = masks[idx].cpu().float().numpy()
            true_mask = true_mask.transpose((1, 2, 0))
            true_mask = resize(true_mask, im.shape[0:2], preserve_range=True)
            true_mask = true_mask > 0.5
            true_mask = true_mask.astype('uint8')
            
            pred_gray = (pred_masks[idx]).cpu().numpy()
            pred_gray = pred_gray.transpose((1, 2, 0))
            pred_gray = resize(pred_gray, im.shape[0:2], preserve_range=True)
            pred_mask = pred_gray > 0.5
            pred_mask = pred_mask.astype('uint8')
            
            true_boxes, true_class_ids, true_scores = get_bounding_boxes(true_mask)
            true_classes = [dataset.decode_class[class_id] for class_id in true_class_ids]

            pred_boxes, pred_class_ids, pred_scores = get_bounding_boxes(pred_mask)
            inds = nms(pred_boxes, pred_scores, thres=0.3)
            pred_boxes = pred_boxes[inds]
            pred_scores = pred_scores[inds]
            pred_class_ids = pred_class_ids[inds]
            pred_boxes = pred_boxes[pred_scores>0.5]
            pred_class_ids = pred_class_ids[pred_scores>0.5]
            pred_scores = pred_scores[pred_scores>0.5]
            pred_classes = [dataset.decode_class[class_id] for class_id in pred_class_ids]
            pred_class_ids = pred_class_ids.astype(int)
            

            for class_id in range(1, model.nb_classes):
                pred = pred_boxes[pred_class_ids==class_id]
                true = true_boxes[true_class_ids==class_id]
                if len(pred) == 0 or len(true) == 0:
                    continue
                match = match_ordered_boxes(pred, true)
                true_positives += (match.max(axis=1)==1).sum()
            nb_predicted_positives += len(pred_boxes)
            nb_positives += len(true_boxes)
            
            precision = true_positives / nb_predicted_positives
            recall = true_positives / nb_positives
            print('Precision : ', precision)
            print('Recall : ', recall)
            
            color = np.array([255, 0, 0]).reshape((1, 3)) * np.ones((len(true_boxes), 1))
            draw_boxes(im, true_boxes, true_classes, true_scores, color)
            draw_boxes(im, pred_boxes, pred_classes, pred_scores, colors[pred_class_ids])
            imsave(os.path.join('eval', os.path.basename(filenames[idx])), im)


            print(filenames[idx])

if __name__ == '__main__':
    run(main)
