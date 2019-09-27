import socket
import cv2
import struct
import _pickle as pk

def recv_all(sock, n):

    buf = b''
    while n > 0:
        t = sock.recv(n)
        buf += t
        n -= len(t)
    return buf

class Client:

    def __init__(self, host, port = 1111):
        self.host = host
        self.port = port
        self.sock = self._connect()

    def _connect(self):
        print('connecting to server [%s:%d]' % (self.host, self.port))
        sock = socket.socket(socket.AF_INET,  socket.SOCK_DGRAM)
        sock.connect((self.host, self.port))
        print('connecting success')
        return sock

    def send_img(self, img):


        _, data = cv2.imencode('.jpg', img)  #壓縮降低資料大小
        self.send(data)


    def send(self, data):

        #print('sending........')
        data = pk.dumps(data)  # 轉 binary
        n = len(data)
        n = struct.pack('i', n)

        self.sock.send(n)
        self.sock.send(data)

    def recv(self):
        #print('receiving........')
        n = self.sock.recv(4)
        if not n:
            return None
        n = struct.unpack('i', n)[0]
        data = recv_all(self.sock, n)
        data = pk.loads(data)
        return data


    def disconnect(self):
        self.sock.close()
        print('transport terminated')










