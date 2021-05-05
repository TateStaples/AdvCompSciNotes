import os
import time

from SoundManager import SoundManager
from YOLO_wrapper import *


base_path = "resources/"
quip_type = "Prequels"
files = os.listdir(base_path+quip_type)


video = cv2.VideoCapture(0)
audio = SoundManager(labeled_sounds={file[:file.index(".")]: f"{base_path}{quip_type}/{file}" for file in files})
audio.loop = False


if __name__ == '__main__':
    while True:
        ret, frame = video.read()
        cv2.imwrite("resources/placeholder.png", frame)
        labels = scan("resources/placeholder.png")
        frame = draw_labels(frame, labels)
        print(labels)
        if "person" in labels:
            audio.randomize_audio()
            time.sleep(10)
        else:
            audio.playing = False
        # if cv2.waitKey(1):
        #     pass
        # cv2.imshow("view", frame)
