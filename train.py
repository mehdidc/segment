import os

import numpy as np

from torch.utils.data import DataLoader
from torch.nn.functional import smooth_l1_loss, cross_entropy, binary_cross_entropy_with_logits
from torch.optim import Adam, SGD, Adagrad
from skimage.transform import resize
import torch
import torch.nn as nn

import albumentations as A

from tensorboardX import SummaryWriter

from data import coco, voc
from data.base import load_image
from data.base import DetectionDataset
from data.base import filter_classes

from bounding_boxes import iou_segmentation


from model import UNet
from viz import colored_mask

rng = np.random.RandomState(43)

"""
filenames, annotations = coco.get_annotations(
    images_folder='coco/train2017',
    annotations_file='coco/annotations/instances_train2017.json',
)
"""
filenames, annotations = voc.get_annotations(
    root_dir='voc/VOCdevkit/VOC2012',
    imageset_file='voc/VOCdevkit/VOC2012/ImageSets/Main/train.txt'
)
filenames, annotations = filter_classes(filenames, annotations, ['person'])
"""
filenames, annotations = coco.get_annotations(
    images_folder='alcohol/images/train',
    annotations_file='alcohol/annotations/train.json',
)
annotations = [[(box, 'alcohol') for box, class_id in anns] for anns in annotations]
"""
filenames = np.array(filenames)
annotations = np.array(annotations)
indices = np.arange(len(filenames))
rng.shuffle(indices)
filenames = filenames[indices]
annotations = annotations[indices]

dataset = DetectionDataset(filenames, annotations)
image_size = 224
batch_size = 2
nb_epochs = 10000
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
transform = A.Compose([
    A.SmallestMaxSize(301),
    A.RandomCrop(300, 300, p=0.5),
    A.HorizontalFlip(p=0.5),
    A.Blur(blur_limit=5, p=0.5),
    A.MotionBlur(blur_limit=5, p=0.5),
    A.RandomBrightness(p=0.5),
    A.RandomRotate90(p=0.5),
    A.RandomScale(scale_limit=0.3, p=0.5),
    A.Resize(height=image_size, width=image_size, p=1.0),
    A.Normalize(mean=mean, std=std, max_pixel_value=255.),
])
valid_transform = A.Compose([
    A.Resize(height=image_size, width=image_size, p=1.0),
    A.Normalize(mean=mean, std=std, max_pixel_value=255.),
])
def denormalize(im):
    mu = np.array(mean)[None, None, :]
    sigma = np.array(std)[None, None, :]
    return (im * sigma + mu)

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
colors = rng.uniform(0, 1, size=(dataset.nb_classes, 3))
model = UNet(3, dataset.nb_classes)
model.cuda()
optimizer = Adam(model.parameters(), lr=1e-3)
nb_iter = 0
first_epoch = 0
model_dict = {
    'model': model,
    'transform': transform,
    'valid_transform': valid_transform,
    'optimizer': optimizer,
    'encode_class': dataset.encode_class,
    'decode_class': dataset.decode_class,
}

if os.path.exists('model.th'):
    model_dict = torch.load('model.th')
    model = model_dict['model']
    model.cuda()
    optimizer = model_dict['optimizer']
    nb_iter = model_dict['nb_iter']
    first_epoch = model_dict['epoch']
writer = SummaryWriter(log_dir='log')
for epoch in range(first_epoch, nb_epochs):
    for batch_index, (images, masks, filenames) in enumerate(dataloader):
        print('Epoch {:04d} Iter {:04d} Batch {:04d}/{:04d}'.format(epoch, nb_iter, batch_index, len(dataloader)))
        images = images.cuda()
        true = masks.cuda()
        pred = model(images)
        #is_train = (batch_index % 10) > 0
        is_train = True
        model.zero_grad() 
        pred_vect = pred.transpose(1, 3).contiguous().view(-1, model.nb_classes)
        true_vect = true.transpose(1, 3).contiguous().view(-1, model.nb_classes)
        object_loss = binary_cross_entropy_with_logits(
            pred_vect[:, 0], 
            true_vect[:, 0]
        )
        is_positive = true_vect[:, 0]==1
        if is_positive.sum() == 0:
            class_loss = torch.Tensor([0.]).cuda()
        else:
            class_loss = binary_cross_entropy_with_logits(
                pred_vect[is_positive, 1:], 
                true_vect[is_positive, 1:]
            )
        loss = object_loss + class_loss
        if is_train:
            loss.backward()
            optimizer.step()
        # eval
        true_class = true.long().cpu().numpy()
        pred_proba = nn.Sigmoid()(pred).detach().cpu().numpy()

        pred_proba_vect = nn.Sigmoid()(pred_vect).detach().cpu().numpy()
        pred_class_vect = (pred_proba_vect > 0.5).astype(int)
        true_class_vect = true_vect.long().cpu().numpy()
        is_positive = (true_class_vect[:, 0]==1)
        ious = iou_segmentation(true_class_vect[is_positive], pred_class_vect[is_positive])
        iou_objectness = iou_segmentation(true_class_vect[:, 0:1], pred_class_vect[:, 0:1])[0]
        
        pixel_acc_per_class = (true_class_vect[is_positive] == pred_class_vect[is_positive]).mean(axis=0)
        pixel_acc = pixel_acc_per_class.mean()
        pixel_acc_objectness = (true_class_vect[:, 0] == pred_class_vect[:, 0]).mean()
        split = 'train' if is_train else 'val'
        writer.add_scalar('{}/loss'.format(split), loss.item(), nb_iter)
        writer.add_scalar('{}/object_loss'.format(split), object_loss.item(), nb_iter)
        writer.add_scalar('{}/class_loss'.format(split), class_loss.item(), nb_iter)
        writer.add_scalar('{}/pixel_acc'.format(split), pixel_acc, nb_iter)
        writer.add_scalar('{}/pixel_acc_objectness'.format(split), pixel_acc_objectness, nb_iter)
        writer.add_scalar('{}/iou_objectness'.format(split), iou_objectness, nb_iter)
        for class_id in range(1, model.nb_classes):
            acc = pixel_acc_per_class[class_id]
            iou = ious[class_id]
            writer.add_scalar('{}/pixel_acc_{}'.format(split, dataset.decode_class[class_id]), acc, nb_iter)
            writer.add_scalar('{}/iou_{}'.format(split, dataset.decode_class[class_id]), iou, nb_iter)
        writer.add_scalar('{}/iou'.format(split), ious[1:].mean(), nb_iter)

        if nb_iter % 10 == 0:
            idx = np.random.randint(0, len(filenames))
            im = load_image(filenames[idx])
            h, w = im.shape[0], im.shape[1]
            
            augmented_im = images[idx].cpu().numpy().transpose((1, 2, 0))
            augmented_im = resize(augmented_im, (h, w), preserve_range=True)
            augmented_im = denormalize(augmented_im)

            im_pred_proba = pred_proba[idx]
            im_pred_proba = im_pred_proba.transpose((1, 2, 0))

            im_true = true_class[idx].astype(float)
            im_true = im_true.transpose((1, 2, 0))

            im_true = resize(im_true, (h, w), preserve_range=True)
            im_pred_proba = resize(im_pred_proba, (h, w), preserve_range=True)
            im_pred = (im_pred_proba>0.5).astype(float)

            true_mask = colored_mask(im_true, colors)
            pred_mask = colored_mask(im_pred, colors)

            pred_obj = (im_pred[:, :, 0] == 1).astype(float)
            pred_obj = pred_obj.reshape((h, w, 1)) * np.ones((1, 1, 3))
            pred_mask[pred_obj == 0] = 0

            pred_obj_score = im_pred_proba[:, :, 0]

            im = im.transpose((2, 0, 1))
            true_mask = (255 * true_mask.transpose((2, 0, 1))).astype('uint8')
            pred_mask = (255 * pred_mask.transpose((2, 0, 1))).astype('uint8')
            pred_obj = (255  * pred_obj.transpose((2, 0, 1))).astype('uint8')
            pred_obj_score = (255 * pred_obj_score).astype('uint8')
            augmented_im = (255 * augmented_im.transpose((2, 0, 1))).astype('uint8')
            
            writer.add_image('data/img', im, nb_iter)
            writer.add_image('data/mask', true_mask, nb_iter)
            writer.add_image('data/pred_mask', pred_mask, nb_iter)
            writer.add_image('data/pred_object_mask', pred_obj, nb_iter)
            writer.add_image('data/pred_object_score', pred_obj_score, nb_iter)
            writer.add_image('data/augmented_im', augmented_im, nb_iter)

            model_dict['nb_iter'] = nb_iter
            model_dict['epoch'] = epoch
            torch.save(model_dict, 'model.th')
        nb_iter += 1
