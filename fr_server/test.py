import matplotlib.pyplot as plt
from mxnet.gluon import data as gdata
from detector import Dectector
import dataset as mydata
import utils


if __name__ == '__main__':

    net = utils.mobile_ssd(2, pretrained_file='mod/face_detectv4.params')
    detector = Dectector(net)
    file_name = 'data/lfw_5590_test'
    data_path = 'D:/data/lfw_5590/'
    resize_aug = gdata.vision.transforms.Resize(size=(400, 400))
    dataset = mydata.fr_dataset(file_name, data_path).transform_first(resize_aug)


    for i in range(len(dataset)):
        img, rec = dataset[i]
        fig = plt.imshow(img.asnumpy())
        pred_boxes = detector(img, 0.5)
        utils.show_bbox(fig.axes, img.shape, rec.asnumpy(), 'blue')
        if len(pred_boxes) > 0:
            utils.show_bbox(fig.axes, img.shape, pred_boxes.asnumpy(), 'red')
        plt.show()

