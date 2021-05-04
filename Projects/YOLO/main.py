import librosa

import darknet.darknet as dn
import cv2

labels = b"darknet/data/coco.names"
model = b"darknet/cfg/yolov3-tiny.cfg"
weights = b"darknet/yolov3-tiny.weights"

net = dn.load_net(model, weights, 0)
meta = dn.METADATA(labels)


class Label:
    def __init__(self, data):
        self.name, self.weight, self.rect = data
        self.x, self.y, self.width, self.height = self.rect
        self.p1, self.p2 = (int(self.x-self.width/2), int(self.y-self.height/2)), \
                           (int(self.x+self.width/2), int(self.y+self.height/2))

    def draw(self, img):
        return cv2.rectangle(img=img, pt1=self.p1, pt2=self.p2, color=(0, 255, 0), lineType=1)


def scan(file):
    results = dn.detect(net, meta, bytes(file, "utf-8"))
    labels = [Label(item) for item in results]
    return labels

def annotate_frame(img, labels):
    for label in labels:
        img = label.draw(img)
    return img
