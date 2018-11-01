import numpy as np

def blend(a, b):
    return (a * 0.5 + 0.5 * b * (1 - 0.5)) / (0.5 + 0.5 * (1 - 0.5))

def colored(im, color):
    im = im[:, :, None] * np.ones((1, 1, 3)) 
    im[:, :, 0] *= color[0]
    im[:, :, 1] *= color[1]
    im[:, :, 2] *= color[2]
    return im.astype('uint8')
