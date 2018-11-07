import pandas as pd
import os
from bs4 import BeautifulSoup

def get_annotations(root_dir, imageset_file):
    anns = get_all_obj_and_box(root_dir, imageset_file)
    anns = sorted(anns)
    anns = [(fname, bboxes) for fname, bboxes in anns if len(bboxes) > 0]
    filenames = [fname for fname, bboxes in anns]
    annotations = [bboxes for fname, bboxes in anns]
    return filenames, annotations

def imgs_from(dataset):
    return list(map(str.strip, open(dataset).readlines()))


def annotation_file_from_img(img_name, root_dir):
    """
    Given an image name, get the annotation file for that image
    Args:
        img_name (string): string of the image name, relative to
            the image directory.
    Returns:
        string: file path to the annotation file
    """
    ann_dir = os.path.join(root_dir, 'Annotations')
    return os.path.join(ann_dir, img_name) + '.xml'


def load_annotation(img_filename, root_dir):
    """
    Load annotation file for a given image.
    Args:
        img_name (string): string of the image name, relative to
            the image directory.
    Returns:
        BeautifulSoup structure: the annotation labels loaded as a
            BeautifulSoup data structure
    """
    xml = ""
    with open(annotation_file_from_img(img_filename, root_dir)) as f:
        xml = f.readlines()
    xml = ''.join([line.strip('\t') for line in xml])
    return BeautifulSoup(xml, 'lxml')


def get_all_obj_and_box(root_dir, img_set, classes=None):
    if classes:
        classes = set(classes)
    img_dir = os.path.join(root_dir, 'JPEGImages')
    obj_list = []
    img_list = imgs_from(img_set)
    for img in img_list:
        anno = load_annotation(img, root_dir)
        fname = anno.findChild('filename').contents[0]
        fname = os.path.join(img_dir, fname)
        objs = anno.findAll('object')
        data = []
        for obj in objs:
            obj_names = obj.findChildren('name')
            for name_tag in obj_names:
                if classes and name_tag.contents[0] not in classes: 
                    continue
                bbox = obj.findChildren('bndbox')[0]
                xmin = int(bbox.findChildren('xmin')[0].contents[0])
                ymin = int(bbox.findChildren('ymin')[0].contents[0])
                xmax = int(bbox.findChildren('xmax')[0].contents[0])
                ymax = int(bbox.findChildren('ymax')[0].contents[0])
                x, y = xmin, ymin
                w, h = xmax - xmin, ymax - ymin
                name = name_tag.contents[0].strip()
                data.append(((x, y, w, h), name))
        obj_list.append((fname, data))
    return obj_list

