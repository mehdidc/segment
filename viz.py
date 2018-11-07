import numpy as np
import cv2

def colored_mask(mask, colors):
    h, w, nb_classes = mask.shape
    colors_ = colors.reshape((1, 1, nb_classes, 3))
    colors_ = colors_[:, :, 1:]
    mask = mask.reshape((h, w, nb_classes, 1)).copy()
    mask = mask[:, :, 1:]
    return blend(mask * colors_)


def blend(x, axis=2):
    result = x.sum(axis=axis) / (x>0).sum(axis=axis)
    result[result==np.nan] = 0
    return result

def draw_boxes(im, boxes, classes, scores, colors, font_scale=2):
    font = cv2.FONT_HERSHEY_PLAIN
    for (x, y, w, h), class_id, score, color in zip(boxes, classes, scores, colors):
        text = '{}({:.2f})'.format(class_id, score)
        color = color.tolist()
        cv2.rectangle(im, (x, y), (x + w, y + h), color, 2)
        cv2.putText(im, text, (x, y), font, font_scale, color, 2, cv2.LINE_AA)
    return im
