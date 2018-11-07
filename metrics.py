import numpy as np
def global_pixel_acc(pred, true):
    return (pred==true).mean()

def pixel_acc_per_class(pred, true):
    return (pred==true).mean(axis=(0, 2, 3))
