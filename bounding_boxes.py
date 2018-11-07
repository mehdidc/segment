import cv2
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

def get_bounding_boxes(mask, heat_map):
    #mask : (h, w, nb_classes)
    h, w, nb_classes = mask.shape
    mask = mask * mask[:, :, 0:1]
    boxes = []
    scores = []
    classes = []
    for class_id in range(1, nb_classes):
        boxes_, scores_ = get_bounding_boxes_one_class(mask[:, :, class_id], heat_map[:, :, class_id])
        boxes.extend(boxes_)
        scores.extend(scores_)
        classes.extend([class_id] * len(boxes_))
    boxes = np.array(boxes)
    classes = np.array(classes)
    scores = np.array(scores)
    return boxes, classes, scores


def get_bounding_boxes_one_class(mask, heat_map):
    # mask : (h, w)
    _, contours, _ = cv2.findContours(
        (mask * 255).astype('uint8'),
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )
    # fill holes
    mask_no_holes = np.zeros_like(mask)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(mask_no_holes, (x, y), (x + w, y + h), (1, 1, 1), cv2.FILLED)
    # get final mask
    _, contours, _ = cv2.findContours(
        (mask_no_holes*255).astype('uint8'),
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )
    boxes = []
    scores = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        score = heat_map[y:y+h, x:x+w].mean()
        boxes.append((x, y, w, h))
        scores.append(score)
    return boxes, scores


def nms(boxes, confidences, thres=0.3):
    boxes = np.array(boxes)
    confidences = np.array(confidences)
    
    indices = np.argsort(confidences)[::-1]
    boxes = boxes[indices]
    confidences = confidences[indices]

    result_indices = []
    while len(boxes) > 0:
        cur_box = boxes[0]
        cur_confidence = confidences[0]
        cur_index = indices[0]

        next_boxes = boxes[1:]
        next_confidences = confidences[1:]
        next_indices = indices[1:]

        cur_box = cur_box.reshape((1, 4))

        ious = iou(cur_box, next_boxes)
        boxes = next_boxes[ious < thres]
        confidences = next_confidences[ious < thres]
        indices = next_indices[ious < thres]

        result_indices.append(cur_index)
    return np.array(result_indices).astype('int')

def iou_all_pairs(boxes, other_boxes):
    return iou(
        boxes.reshape((len(boxes), 1, 4)),
        other_boxes.reshape((1, len(other_boxes), 4)),
    )


def match_ordered_boxes(pred_boxes, true_boxes, iou_threshold=0.5):
    ious = iou_all_pairs(pred_boxes, true_boxes)
    matching = np.zeros_like(ious).astype('bool')
    true_already_matched = np.zeros(len(true_boxes)).astype('bool')
    for pred_ind in range(len(pred_boxes)):
        true_match_ind = -1
        true_best_iou = 0 
        for true_ind in range(len(true_boxes)):
            iou = ious[pred_ind, true_ind]
            if iou <= iou_threshold:
                continue
            if true_already_matched[true_ind]:
                continue
            if iou > true_best_iou:
                true_match_ind = true_ind
                true_best_iou = iou
        if true_match_ind < 0:
            continue
        true_already_matched[true_match_ind] = True
        matching[pred_ind, true_match_ind] = True
    return matching


def iou(boxes, other_boxes, eps=1e-10):
    ax, ay, aw, ah = get_boxes_coords(boxes)
    bx, by, bw, bh = get_boxes_coords(other_boxes)

    xmin = np.maximum(ax, bx)
    ymin = np.maximum(ay, by)
    xmax = np.minimum(ax + aw, bx + bw)
    ymax = np.minimum(ay + ah, by + bh)

    w_intersection = np.clip(xmax - xmin, a_min=0, a_max=None)
    h_intersection = np.clip(ymax - ymin, a_min=0, a_max=None)
    intersection = w_intersection * h_intersection
    union = aw * ah + bw * bh - intersection
    return intersection / (union + eps) 

def iou_segmentation(mask, other_mask):
    inter = ((mask == 1) & (other_mask == 1)).astype(int).sum(axis=0)
    union = mask.astype(int).sum(axis=0) + other_mask.astype(int).sum(axis=0) - inter
    return inter / (union + 1e-10)

