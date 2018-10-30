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

def blend(a, b):
    return (a * 0.5 + 0.5 * b * (1 - 0.5)) / (0.5 + 0.5 * (1 - 0.5))

def colored(im, color):
    im = im[:, :, None] * np.ones((1, 1, 3)) 
    im[:, :, 0] *= color[0]
    im[:, :, 1] *= color[1]
    im[:, :, 2] *= color[2]
    return im.astype('uint8')


cuda = False

model = torch.load(
    'model.th', map_location=lambda storage, location:storage)
if cuda:
    model.cuda()
model.eval()

filenames, annotations = get_annotations(
    images_folder='alcohol/images/valid',
    annotations_file='alcohol/annotations/valid.json'
)
annotations = [[(box, 'alcohol') for box, class_id in anns] for anns in annotations]
image_size = 224
transform = Compose([
    HorizontalFlip(p=0.0),
    Resize(height=image_size, width=image_size, p=1.0),
    Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.),
])
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

for images, masks, filenames in dataloader:
    if cuda:
        images = images.cuda()
        masks = masks.cuda()
    with torch.no_grad():
        pred_masks = model(images)
    # eval
    _, pred = pred_masks.max(dim=1)
    pred_score = nn.Softmax(dim=1)(pred_masks)[:, 1:].sum(dim=1)
    _, true = masks.max(dim=1)
    pixel_acc = (pred == true).float().mean()
    for idx in range(len(filenames)):
        im = load_image(filenames[idx])
        im = im.astype(float)
        im /= 255.0
        true_mask = (true[idx]>0).cpu().float().numpy()
        true_mask = resize(true_mask, im.shape[0:2], preserve_range=True)
        true_mask = true_mask.astype('uint8')
        
        #pred_mask =(pred[idx]>0).cpu().float().numpy()
        #pred_mask = resize(pred_mask, im.shape[0:2], preserve_range=True)
    
        pred_gray = (pred_score[idx]).cpu().numpy()
        pred_gray = cv2.medianBlur(pred_gray, 5)
        pred_gray = resize(pred_gray, im.shape[0:2], preserve_range=True)
        pred_mask = pred_gray > threshold_otsu(pred_gray)
        pred_mask = pred_mask.astype('uint8')

    	# fill holes
        _, contours, _ = cv2.findContours(
            (pred_mask*255).astype('uint8'),
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE
        )
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(pred_mask, (x, y), (x + w, y + h), (1, 1, 1), cv2.FILLED)
        # draw bounding boxes
        _, contours, _ = cv2.findContours(
            (pred_mask*255).astype('uint8'),
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE
        )
        pred_mask_no_holes = np.zeros_like(pred_mask)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(pred_mask_no_holes, (x, y), (x + w, y + h), (1, 1, 1), 2)
        # blend tru and pred_mask with transparency
        im = blend(im, colored(true_mask, (1, 0, 0)))
        im = blend(im, colored(pred_mask, (0, 1, 0)))
        imsave(os.path.join('test', os.path.basename(filenames[idx])), im)
        print(filenames[idx])

