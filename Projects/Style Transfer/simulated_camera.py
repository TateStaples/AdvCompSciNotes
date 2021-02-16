import cv2
import torch
from transformer import TransformerNet
import utils
import numpy as np
from real_time import style_img

style = "udnie"
style_model = TransformerNet()
style_model.load_state_dict(torch.load(f"saved-models/{style}.pth"))

video = cv2.VideoCapture(1)

while True:
    ret, frame = video.read()
    # cv2.imwrite("source.png", frame)
    print(frame)
    # print(img.shape)
    img = style_img(img, style_model)

    # cv2.imwrite("test.png", img)
    # break


    # Display frame and break if user hits q
    cv2.imshow('frame', img)
    cv2.imshow("source", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


if __name__ == '__main__':
    pass