import mxnet as mx
from mxnet import gluon, init, contrib, nd
from mxnet.gluon import nn, model_zoo
import numpy as np


def cls_blk(anchors_num, classes):
    out = anchors_num * (classes)
    net = nn.Conv2D(out, kernel_size=3, padding=1)
    net.initialize(init.Xavier())
    net.collect_params().setattr('lr_mult', 2)
    return net


def bbox_blk(anchors_num):
    out = anchors_num * 4
    net = nn.Conv2D(out, kernel_size=3, padding=1)
    net.initialize(init.Xavier())
    net.collect_params().setattr('lr_mult', 2)
    return net


def output():
    
    net = nn.GlobalMaxPool2D()
    net.collect_params().setattr('lr_mult', 3)
    return net

    
def concat_down(x):
    
    net = nn.Sequential()
    net.add(nn.MaxPool2D(2), nn.BatchNorm())
    net.initialize(ctx = x.context)
    
    return net(x)

    

    
    
class MyBlk(nn.Block):

    def __init__(self, blk, classes, size, ratio, **kwargs):
        super(MyBlk, self).__init__(**kwargs)

        self.size = size
        self.ratio = ratio
        N = len(size) + len(ratio) - 1

        self.blk = blk
        self.cls_blk = cls_blk(N, classes)
        self.bbox_blk = bbox_blk(N)

    def forward(self, x):
        yhat = self.blk(x)
        anchors = contrib.nd.MultiBoxPrior(yhat, sizes=self.size, ratios=self.ratio)
        cls_yhat = self.cls_blk(yhat)
        bbox_yhat = self.bbox_blk(yhat)

        return yhat, anchors, cls_yhat, bbox_yhat


class VGG16_SSD(nn.Block):

    def __init__(self, classes, ctx = mx.cpu(), **kwargs):
        super(VGG16_SSD, self).__init__(**kwargs)
        self.classes = classes
        self.ctx = ctx

        
        m = 1/16
        n = (1.05- m)/6
        r = [0.5, 1, 2]
        
        net = model_zoo.vision.vgg16_bn(pretrained = True).features
        print(net)
        
        self.net_0 = MyBlk(net[:7], classes, [m, ((m)*(m+n))**0.5], r)
        self.net_1 = MyBlk(net[7:14], classes, [m+n, ((m+n)*(m+2*n))**0.5], r)
        self.net_2 = MyBlk(net[14:24], classes, [m+2*n, ((m+2*n)*(m+3*n))**0.5], r)
        self.net_3 = MyBlk(net[24:34], classes, [m+3*n, ((m+3*n)*(m+4*n))**0.5], r)
        self.net_4 = MyBlk(net[34:44], classes, [m+4*n, ((m+4*n)*(m+5*n))**0.5], r)
        self.net_5 = MyBlk(output(), classes, [m+5*n, ((m+5*n)*(m+6*n))**0.5], r)
       
        

    def forward(self, x):
        anchors, cls_yhats, bbox_yhats = [], [], []

        for i in range(6):
            net = getattr(self, 'net_%d' % i)
            x, anch, cls_yhat, bbox_yhat = net(x)

            cls_yhat = cls_yhat.transpose((0, 2, 3, 1)).flatten()
            cls_yhat = cls_yhat.reshape((cls_yhat.shape[0], -1, self.classes))

            bbox_yhat = bbox_yhat.transpose((0, 2, 3, 1)).flatten()
            bbox_yhat = bbox_yhat.reshape((bbox_yhat.shape[0], -1, 4))

            anchors.append(anch)
            cls_yhats.append(cls_yhat)
            bbox_yhats.append(bbox_yhat)

        return nd.concat(*anchors, dim=1), nd.concat(*cls_yhats, dim=1), nd.concat(*bbox_yhats, dim=1)
    
class Resnet18_SSD(nn.Block):

    def __init__(self, classes, ctx = mx.cpu(), **kwargs):
        super(Resnet18_SSD, self).__init__(**kwargs)
        self.classes = classes
        self.ctx = ctx

        
        m = 0.1
        n = (1.05- m)/6
        r = [0.5, 1, 2]
        
        net = model_zoo.vision.resnet18_v2(pretrained = True).features
        
        self.net_0 = MyBlk(net[:5], classes, [m, ((m)*(m+n))**0.5, m+n], r)
        self.net_1 = MyBlk(net[5], classes, [m+n, ((m+n)*(m+2*n))**0.5, m+2*n], r)
        self.net_2 = MyBlk(net[6], classes, [m+2*n, ((m+2*n)*(m+3*n))**0.5, m+3*n], r)
        self.net_3 = MyBlk(net[7], classes, [m+3*n, ((m+3*n)*(m+4*n))**0.5, m+4*n], r)
        self.net_4 = MyBlk(net[8], classes, [m+4*n, ((m+4*n)*(m+5*n))**0.5, m+5*n], r)
        self.net_5 = MyBlk(net[9:12], classes, [m+5*n, ((m+5*n)*(m+6*n))**0.5, m+6*n], r)
       
        

    def forward(self, x):
        anchors, cls_yhats, bbox_yhats = [], [], []

        for i in range(6):
            net = getattr(self, 'net_%d' % i)
            x, anch, cls_yhat, bbox_yhat = net(x)

            cls_yhat = cls_yhat.transpose((0, 2, 3, 1)).flatten()
            cls_yhat = cls_yhat.reshape((cls_yhat.shape[0], -1, self.classes))

            bbox_yhat = bbox_yhat.transpose((0, 2, 3, 1)).flatten()
            bbox_yhat = bbox_yhat.reshape((bbox_yhat.shape[0], -1, 4))

            anchors.append(anch)
            cls_yhats.append(cls_yhat)
            bbox_yhats.append(bbox_yhat)

        return nd.concat(*anchors, dim=1), nd.concat(*cls_yhats, dim=1), nd.concat(*bbox_yhats, dim=1)
    
    
    
class Mobile_SSD(nn.Block):

    def __init__(self, classes, ctx = mx.cpu(), **kwargs):
        super(Mobile_SSD, self).__init__(**kwargs)
        self.classes = classes
        self.ctx = ctx

        
        m = 0.1
        n = (1.05- m)/4
        r = [1]
        
        net = model_zoo.vision.mobilenet_v2_0_5(pretrained = True).features
        
        self.net_0 = MyBlk(net[:9], classes, [m, ((m)*(m+n))**0.5], r)
        self.net_1 = MyBlk(net[9:16], classes, [m+n, ((m+n)*(m+2*n))**0.5], r)
        self.net_2 = MyBlk(net[16:23], classes, [m+2*n, ((m+2*n)*(m+3*n))**0.5], r)
        self.net_3 = MyBlk(net[23], classes, [m+3*n, ((m+3*n)*(m+4*n))**0.5], r)
       
       
        

    def forward(self, x):
        anchors, cls_yhats, bbox_yhats = [], [], []

        for i in range(4):
            net = getattr(self, 'net_%d' % i)
            x, anch, cls_yhat, bbox_yhat = net(x)

            cls_yhat = cls_yhat.transpose((0, 2, 3, 1)).flatten()
            cls_yhat = cls_yhat.reshape((cls_yhat.shape[0], -1, self.classes))

            bbox_yhat = bbox_yhat.transpose((0, 2, 3, 1)).flatten()
            bbox_yhat = bbox_yhat.reshape((bbox_yhat.shape[0], -1, 4))

            anchors.append(anch)
            cls_yhats.append(cls_yhat)
            bbox_yhats.append(bbox_yhat)

        return nd.concat(*anchors, dim=1), nd.concat(*cls_yhats, dim=1), nd.concat(*bbox_yhats, dim=1)