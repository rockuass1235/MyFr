import ssd
import matplotlib.pyplot as plt
import numpy as np



def mobile_ssd(n = 2, pretrained_file = ''):

    net = ssd.Mobile_SSD(n)
    if pretrained_file != '':
        net.load_parameters(pretrained_file)

    return net

def show_bbox(axes, img_shape, gts, color):
    h, w, _ = img_shape
    w -= 1
    h -= 1
    for gt in gts:

        if gt.shape[0] > 5:
            x1, y1, x2, y2 = (gt[2:] * np.array([w, h, w, h])).astype(int)
        else:
            x1, y1, x2, y2 = (gt[1:] * np.array([w, h, w, h])).astype(int)

        rect = plt.Rectangle(xy=(x1, y1), width=x2 - x1, height=y2 - y1, fill=False, edgecolor=color, linewidth=3)
        axes.add_patch(rect)

