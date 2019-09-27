import matplotlib.pyplot as plt
import numpy as np
import cv2




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

def draw_bbox(img, gts, color):

    h, w, _ = img.shape
    w -= 1
    h -= 1
    for gt in gts:

        if len(gt) > 5:
            x1, y1, x2, y2 = (gt[2:] * np.array([w, h, w, h])).astype(int)
        else:
            x1, y1, x2, y2 = (gt[1:] * np.array([w, h, w, h])).astype(int)


        cv2.rectangle(img,(x1,y1),(x2,y2),color,2)

def get_faces(img, gts, r = 0.1):

    h, w, _ = img.shape
    w -= 1
    h -= 1
    for gt in gts:

        if len(gt) > 5:
            x1, y1, x2, y2 = (gt[2:] * np.array([w, h, w, h])).astype(int)
        else:
            x1, y1, x2, y2 = (gt[1:] * np.array([w, h, w, h])).astype(int)


        x1 = max(int(x1 - r*w), 0)
        y1 = max(int(y1 - r*h), 0)
        x2 = min(int(x2 + r*w), w)
        y2 = min(int(y2 + r*h), h)


        yield img[y1:y2, x1:x2]




def zoom_in(img, r = 0.7):

    h, w, _ = img.shape
    x, y = w//2, h//2
    w, h = w*r, h*r
    x1, y1, x2, y2 = int(x - 0.5*w), int(y - 0.5*h), int(x + 0.5*w), int(y + 0.5*h)
    shape = (img.shape[1], img.shape[0])
    return cv2.resize(img[y1:y2, x1:x2], shape)
