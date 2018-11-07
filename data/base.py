import numpy as np
import torch
from torchvision.datasets.folder import default_loader
from imageio import imread

from bounding_boxes import boxes_min_max_to_width_height_format
from bounding_boxes import boxes_width_height_to_min_max_format
from bounding_boxes import scale_boxes

class DetectionDataset:

    def __init__(self, filenames, annotations, transform=None):
        self.filenames = filenames
        self.annotations = annotations
        self.transform = transform
        self._prepare()
    
    def _prepare(self):
        classes = [
            class_id for anns in self.annotations for box, class_id in anns]
        classes = np.unique(classes)
        classes = sorted(classes)
        self.encode_class = {
            class_id: (i + 1)
            for i, class_id in enumerate(classes)
        }
        self.decode_class = {
            (i + 1): class_id
            for i, class_id in enumerate(classes)
        }
        self.nb_classes = 1 + len(self.encode_class)

    def __getitem__(self, index):
        filename = self.filenames[index]
        annotations = self.annotations[index]
        boxes = [box for box, class_id in annotations]
        classes = [self.encode_class[class_id] for box, class_id in annotations]
        im = load_image(filename)
        image_height, image_width, _ = im.shape
        boxes = scale_boxes(boxes, 1 / image_width, 1 / image_height)
        if self.transform:
            boxes = boxes_width_height_to_min_max_format(boxes)
            annotations = {
                'image': im, 
                'bboxes': boxes, 
                'category_id': classes,
            }
            annotations = self.transform(**annotations)
            im = annotations['image']
            boxes = annotations['bboxes']
            classes = annotations['category_id']
            boxes = boxes_min_max_to_width_height_format(boxes)
        im_mask = np.zeros((self.nb_classes, im.shape[0], im.shape[1]))
        for box, class_id in zip(boxes, classes):
            x, y, w, h = box
            x *= im.shape[1]
            y *= im.shape[0]
            w *= im.shape[1]
            h *= im.shape[0]
            x = np.clip(x, 0, im.shape[1])
            y = np.clip(y, 0, im.shape[0])
            x = int(x)
            y = int(y)
            w = int(w)
            h = int(h)
            im_mask[class_id, y:y+h, x:x+w] = 1
            im_mask[0, y:y+h, x:x+w] = 1
        im = im.transpose((2, 0, 1))
        im = torch.from_numpy(im).float()
        im_mask = torch.from_numpy(im_mask).float()
        boxes = torch.from_numpy(np.array(boxes)).float()
        classes = torch.from_numpy(np.array(classes)).float()
        return im, im_mask, filename

    def __len__(self):
        return len(self.filenames)


def load_image(filename):
    im = imread(filename)
    if len(im.shape) == 2:
        im = im[:, :, np.newaxis]
        im = im * np.ones((1, 1, 3))
    elif len(im.shape) == 1:
        im = im[:, np.newaxis, np.newaxis]
        im = im * np.ones((1, im.shape[0], 3))
    im = im[:, :, 0:3]
    return im

def filter_classes(filenames, annotations, classes, remove_empty=True):
    annotations = [[(box, class_name) for box, class_name in anns if class_name in classes] for anns in annotations]
    if remove_empty:
        nbs = list(map(len, annotations))
        filenames = [f for f, nb in zip(filenames, nbs) if nb > 0]
        annotations = [a for a, nb in zip(annotations, nbs) if nb > 0]
    return filenames, annotations
