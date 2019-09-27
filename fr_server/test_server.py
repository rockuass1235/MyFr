import socket
import threading
import numpy as np
import struct
import cv2
from mxnet import nd
from mxnet.gluon import data as gdata
import utils
from detector import Dectector
import mxnet as mx
import _pickle as pk
import dlib


ctx = mx.gpu()


def recv_all(sock, n):

    buf = b''
    while n > 0:

        t, _ = sock.recvfrom(n)
        buf += t
        n -= len(t)
    return buf

def send(conn, addr, data):
   
    data = pk.dumps(data)  # 轉binary
    n = len(data)
    n = struct.pack('i', n)

    conn.sendto(n, addr)
    conn.sendto(data, addr)
        

def recv(conn):
        
   
    n, addr = conn.recvfrom(4)
    if not n:
        return None
    n = struct.unpack('i', n)[0]
    data = recv_all(conn, n)
    
    data = pk.loads(data)
    return data, addr



class Server():

    def __init__(self, host='', port=1111):

        self.host = host
        self.port = port
        self.sock = None
        
        net = utils.mobile_ssd(2, pretrained_file='mod/face_detectv4.params')
        net.collect_params().reset_ctx(ctx)
        self.detector = Dectector(net)



    def start(self):

        print('sock initialize')
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # 避免port無法釋放
        
        addr = (self.host, self.port)
        sock.bind(addr)
        self.sock = sock
        
    

    def sub_service(self):
        augs = gdata.vision.transforms.Compose([gdata.vision.transforms.Resize(size=(500, 500)),
                                                gdata.vision.transforms.ToTensor()])
        
        
        conn = self.sock

        while True:


            img, addr = recv(conn)  # 收到的影像
            img = cv2.imdecode(img, 1)
            h, w, _ = img.shape
            '''
            im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            im = nd.array(im)
            im = augs(im)
            im = im.as_in_context(ctx)
            pred_boxes = self.detector(im, 0.5)
            print(pred_boxes)
           
           
            if len(pred_boxes) <= 0:
                send(conn, np.array([]))
            else:
                pred_boxes = pred_boxes.asnumpy()
                print(pred_boxes)
                send(conn, pred_boxes)
            '''    
            
            detector = dlib.cnn_face_detection_model_v1('mod/mmod_human_face_detector.dat')
            dets = detector(img, 0)
            pred_boxes = []
            for det in dets:
                det = det.rect
                pred_boxes.append([0, det.left()/w, det.top()/h, det.right()/w, det.bottom()/h])
            send(conn, addr, np.array(pred_boxes))

            

        print('terminated')
        return




if __name__ == '__main__':
    
    host = '120.110.114.14'
    server = Server(host = host)
    server.start()
    server.sub_service()
