import cv2
import utils
from client import Client
import time


host = '120.110.114.14'
client = Client(host = host)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 700)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 700)


while True:
    ret, img = cap.read()
    h, w, _ = img.shape
    img = utils.zoom_in(img)

    if not ret:
        break
    client.send_img(cv2.resize(img, (250, 250)))

    pred_boxes = client.recv()

    if len(pred_boxes) > 0:
        print(pred_boxes)
        s = time.strftime("%Y%m%d%H%M%S", time.localtime())
        cv2.imwrite('images/' + s + '.jpg', img)

        count = 0
        for face in utils.get_faces(img, pred_boxes, r = 0.05):
            cv2.imwrite('faces/' + str(count) + s + '.jpg', face)
            count += 1

        utils.draw_bbox(img, pred_boxes, (0,0,255))


    cv2.imshow('', img)

    if cv2.waitKey(1000//80) & 0xfff == ord('q'):
        break


client.disconnect()
cap.release()
cv2.destroyAllWindows()