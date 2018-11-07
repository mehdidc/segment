import numpy as np
import time
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
from threading import Thread


class Worker:

    def __init__(self, capture):
        self.stop = False
        self.frame = None
        self.capture = capture

    def update_frame(self):
        while True:
            if self.stop:
                break
            _, self.frame = self.capture.read()


def webcam(
    *,
    capture_device='',
    model_filename='model.th',
    images_folder='coco/val2017', 
    score_threshold=0.5,
    image_size=224,
    cuda=False
    ):

    model_dict = torch.load(
        model_filename, 
        map_location=lambda storage, location:storage
    )
    model = model_dict['model']
    transform = model_dict['valid_transform']
    decode_class = model_dict['decode_calss']
    rng = np.random.RandomState(42)
    colors = rng.randint(0, 255, size=(model.nb_classes, 3))
    if cuda:
        model.cuda()
    cap = cv2.VideoCapture(capture_device)
    worker = Worker(cap)
    thread = Thread(target=worker.update_frame)
    thread.daemon = True
    thread.start()
    nb_frames_processed = 0
    t0 = time.time()
    while True:
        frame = worker.frame
        if frame is None:
            continue
        frame = frame[:, :, ::-1]
        frame = predict(
            frame, model, transform, colors, decode_class,
            cuda=cuda, 
            score_threshold=score_threshold
        )
        frame = frame[:, :, ::-1]
        nb_frames_processed += 1
        delta = time.time() - t0
        fps = nb_frames_processed / delta
        print('FPS ', fps)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    worker.stop = True
    thread.join()
    cap.release()
    cv2.destroyAllWindows()


def folder(
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
    transform = model_dict['valid_transform']
    decode_class = model_dict['decode_calss']
    rng = np.random.RandomState(42)
    colors = rng.randint(0, 255, size=(model.nb_classes, 3))
    if cuda:
        model.cuda()
    model.eval()
    filenames = glob(os.path.join(images_folder, '*.*'))
    for filename in filenames:
        try:
            im =  load_image(filename)
        except Exception:
            continue
        im = predict(
            im, model, transform, colors, decode_class,
            cuda=cuda, 
            score_threshold=score_threshold
        )
        out_filename = os.path.join(
            out_folder,
            os.path.basename(filename)
        )
        imsave(out_filename, im)
        print(filename)

def predict(orig_im, model, transform, colors, decode_class, cuda=False, score_threshold=0.5):
    anns = transform(image=orig_im)
    im = anns['image']
    im = im.transpose((2, 0, 1))
    im = torch.from_numpy(im).float()
    im = im.view(1, im.size(0), im.size(1), im.size(2))
    if cuda:
        im = im.cuda()
    with torch.no_grad():
        pred_masks = nn.Sigmoid()(model(im))
    pred_gray = (pred_masks[0]).cpu().numpy()
    pred_gray = pred_gray.transpose((1, 2, 0))
    pred_gray = resize(pred_gray, orig_im.shape[0:2], preserve_range=True)
    pred_mask = pred_gray > score_threshold
    pred_mask = pred_mask.astype('uint8')
    
    pred_boxes, pred_class_ids, pred_scores = get_bounding_boxes(
            mask=pred_mask, heat_map=pred_gray)
    inds = nms(pred_boxes, pred_scores, thres=0.3)
    pred_boxes = pred_boxes[inds]
    pred_scores = pred_scores[inds]
    pred_class_ids = pred_class_ids[inds]
    pred_classes = [decode_class[class_id] for class_id in pred_class_ids]
    pred_class_ids = pred_class_ids.astype(int)

    obj = pred_mask[:, :, 0]
    col = np.array([255, 0, 0]).reshape((1, 1, 3))
    obj = obj[:, :, np.newaxis] * col
    orig_im = (orig_im + obj) / (1 + pred_mask[:, :, 0])[:, :, np.newaxis] 
    orig_im = orig_im.astype('uint8')
    draw_boxes(orig_im, pred_boxes, pred_classes, pred_scores, colors[pred_class_ids])
    return orig_im


if __name__ == '__main__':
    run([folder, webcam])
