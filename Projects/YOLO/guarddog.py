import cv2
from YOLO_wrapper import *
from SoundManager import SoundManager

video = cv2.VideoCapture(0)

if __name__ == '__main__':
    while True:
        audio = SoundManager("resources/Dog-barking-sound.mp3")
        ret, frame = video.read()
        if cv2.waitKey(1):
            pass

        cv2.imwrite("resources/placeholder.png", frame)
        labels = scan("resources/placeholder.png")
        draw_labels(frame, labels)
        audio.playing = "person" in labels or "dog" in labels
        cv2.imshow("camera", frame)
