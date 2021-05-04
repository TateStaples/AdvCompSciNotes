import sounddevice
import librosa
import cv2
import main
import time
import threading

video = cv2.VideoCapture(0)


class SoundManager:
    def __init__(self, sound_file=None, **kwargs):

        if len(kwargs):
            self.audios = {key: librosa.load(kwargs[key]) for key in kwargs}
        else:
            self.audios = dict()
        self.full_audio, self.sample_rate = librosa.load(sound_file) if sound_file is not None else kwargs[kwargs.keys()[0]]
        self._playing = False
        self.audio = self.full_audio.copy()
        self.prev_time = None

    @property
    def playing(self):
        return self._playing

    @playing.setter
    def playing(self, value):
        if value:
            threading.Thread(target=self._play).start()
        else:
            self._stop()

    def _play(self):
        self._playing = True
        while self._playing:  # loop
            self.prev_time = time.time()
            sounddevice.play(self.audio, self.sample_rate)
            time.sleep(self.audio.shape[0] / self.sample_rate + 1)
            sounddevice.stop()

    def _stop(self):
        sounddevice.stop()
        self._playing = False
        if self.prev_time is not None:
            sample = (time.time() - self.prev_time) * self.sample_rate
            self.audio = self.audio[sample:]

    def set_audio(self, name):
        if len(self.audios) and name in self.audios:
            self.full_audio = self.audios[name]
            self.audio = self.full_audio.copy()
            self.prev_time = False
            self._stop()
            self.playing = True



if __name__ == '__main__':
    while True:
        audio = SoundManager("Dog-barking-sound.mp3")
        ret, frame = video.read()
        if cv2.waitKey(1): pass

        cv2.imwrite("placeholder.png", frame)

        labels = main.scan("placeholder.png")
        found = False
        for label in labels:
            print(label.name)
            if label.name == "person" and not audio.playing:
                audio.playing = True
                found = True
            frame = label.draw(frame)
        if not found:
            audio.playing = False
        cv2.imshow("camera", frame)
