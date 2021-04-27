from playsound import playsound
import cv2
import main

video = cv2.VideoCapture(0)

while True:
    ret, frame = video.read()
    if cv2.waitKey(1): pass

    labels = main.scan()
    cv2.imshow("camera", frame)


if __name__ == '__main__':
    pass
