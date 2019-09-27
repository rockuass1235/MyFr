
from mxnet.gluon import data as gdata
from mxnet import nd, contrib
import mxnet as mx


class Dectector:

    def __init__(self, net):
       
        self.net = net
        

    def _get_gts(self, img):
        
        imgs = img.expand_dims(axis = 0)
        
        anchors, cls_yhat, bbox_yhat = self.net(imgs)
        cls_probs = cls_yhat.softmax().transpose((0, 2, 1))
        output = contrib.nd.MultiBoxDetection(cls_probs, bbox_yhat.flatten(), anchors)
        idx = [i for i, row in enumerate(output[0]) if row[0].asscalar() != -1]
        if len(idx) <= 0:
            return None
        return output[0, idx]

    def __call__(self, img, threshold = 0.7):

        gts = self._get_gts(img)
        if gts is None:
            return []

        idx = [i for i, row in enumerate(gts) if row[1].asscalar() > threshold]

        if len(idx) <= 0:
            return []
        else:
            return gts[idx]








