import os

import numpy as np

from torch.utils.data import DataLoader
from torch.nn.functional import smooth_l1_loss, cross_entropy, binary_cross_entropy_with_logits
from torch.optim import Adam, SGD, Adagrad
from skimage.transform import resize
import torch
import torch.nn as nn

from albumentations import Compose
from albumentations import Resize
from albumentations import Normalize
from albumentations import HorizontalFlip
from albumentations import RandomCrop
from data.coco import get_annotations
from data.base import load_image
from data.base import DetectionDataset

from model import UNet

from tensorboardX import SummaryWriter

filenames, annotations = get_annotations(
    images_folder='alcohol/images/train',
    annotations_file='alcohol/annotations/train.json'
)
annotations = [[(box, 'alcohol') for box, class_id in anns] for anns in annotations]
dataset = DetectionDataset(filenames, annotations)
image_size = 224
batch_size = 4
nb_epochs = 10000

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
    batch_size=batch_size, 
    shuffle=False,
    num_workers=1,
)
model = UNet(3, dataset.nb_classes)
model.cuda()
optimizer = Adam(model.parameters(), lr=1e-3)
nb_iter = 0

if os.path.exists('model.th'):
    model = torch.load('model.th')
    model.cuda()


writer = SummaryWriter(log_dir='log')
for epoch in range(nb_epochs):
    for batch_index, (images, masks, filenames) in enumerate(dataloader):
        print('Iter {}'.format(nb_iter))
        images = images.cuda()
        masks = masks.cuda()
        pred_masks = model(images)
        
        train = (nb_iter % 10) > 0
        
        model.zero_grad()
        predv = pred_masks.transpose(1, 3).contiguous().view(-1, dataset.nb_classes)
        _, truev = masks.transpose(1, 3).contiguous().view(-1, dataset.nb_classes).max(dim=1)
        loss = cross_entropy(predv, truev)
        if train:
            loss.backward()
            optimizer.step()

        # eval
        _, pred = pred_masks.max(dim=1)
        pred_score = nn.Softmax(dim=1)(pred_masks)[:, 1:].sum(dim=1)
        _, true = masks.max(dim=1)
        pixel_acc = (pred == true).float().mean()

        if train:
            writer.add_scalar('data/loss', loss.item(), nb_iter)
            writer.add_scalar('data/pixel_acc', pixel_acc.item(), nb_iter)
        else:
            writer.add_scalar('data/val_loss', loss.item(), nb_iter)
            writer.add_scalar('data/val_pixel_acc', pixel_acc.item(), nb_iter)

        if nb_iter % 10 == 0:
            idx = np.random.randint(0, len(filenames))
            im = load_image(filenames[idx])
            im = im.transpose((2, 0, 1))

            im_mask = (true[idx]>0).cpu().long().numpy()
            pred_mask =(pred[idx]>0).cpu().long().numpy()
            pred_mask_score = ((1-pred_score[idx])*255.0).cpu().long().numpy()

            im_mask = resize(im_mask, im.shape[1:], preserve_range=True)
            pred_mask = resize(pred_mask, im.shape[1:], preserve_range=True)
            #pred_mask_score = resize(pred_mask_score, im.shape[1:], preserve_range=True)
            
            writer.add_image('data/img', im, nb_iter)
            writer.add_image('data/true_mask', im_mask, nb_iter)
            writer.add_image('data/pred_mask', pred_mask, nb_iter)
            writer.add_image('data/pred_score', pred_mask_score, nb_iter)
            torch.save(model, 'model.th')
        nb_iter += 1
