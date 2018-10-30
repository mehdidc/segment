import numpy as np

XMIN, YMIN, WIDTH, HEIGHT = 0, 1, 2, 3

def get_boxes_coords(boxes):
    x, y, w, h = (
        boxes[..., XMIN], 
        boxes[..., YMIN], 
        boxes[..., WIDTH], 
        boxes[..., HEIGHT]
    )
    return x, y, w, h


def scale_boxes(boxes, scale_w, scale_h):
    return [(x * scale_w, y * scale_h, w * scale_w, h * scale_h) for x, y, w, h in boxes]

def center_boxes(boxes):
    return [(x + w/2, y + h/2, w, h) for x, y, w, h in boxes]

def uncenter_boxes(boxes):
    return [(x - w/2, y - h/2, w, h) for x, y, w, h in boxes]

def boxes_width_height_to_min_max_format(boxes):
    return [(x, y, x + w, y + h) for x, y, w, h in boxes]

def boxes_min_max_to_width_height_format(boxes):
    return [(xmin, ymin, xmax - xmin, ymax - ymin) for xmin, ymin, xmax, ymax in boxes]
