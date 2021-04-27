import cv2
import torch
from transformer import TransformerNet
import real_time
import threading
import time

style_model = TransformerNet()
style_model.load_state_dict(torch.load("saved-models/udnie.pth"))

video = cv2.VideoCapture(0)

while True:
    ret, frame = video.read()
    if not ret: continue
    img = real_time.style_img(frame, style_model)

    # cv2.imwrite("source.png")
    cv2.imwrite("test.png", img)
    # break


    # Display frame and break if user hits q
    cv2.imshow('frame', cv2.imread("test.png"))
    # cv2.imshow("source", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


if __name__ == '__main__':
    pass