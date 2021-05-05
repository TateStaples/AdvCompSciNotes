import librosa
import threading
import sounddevice
import time
import random


class SoundManager:
    loop = True

    def __init__(self, sound_file=None, labeled_sounds=None):
        if labeled_sounds is not None:
            self.audios = {key: librosa.load(labeled_sounds[key]) for key in labeled_sounds}
        else:
            self.audios = dict()
        self.full_audio, self.sample_rate = \
            librosa.load(sound_file if sound_file is not None else tuple(labeled_sounds.values())[0])
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
            sounddevice.play(self.audio.copy(), self.sample_rate)
            time.sleep(self.audio.shape[0] / self.sample_rate + 1)
            # if self._playing:
            #     sounddevice.stop()
            if not self.loop:
                break

    def _stop(self):
        sounddevice.stop()
        self._playing = False
        if self.prev_time is not None:
            sample = (time.time() - self.prev_time) * self.sample_rate
            self.audio = self.audio[int(sample):]

    def set_audio(self, name):
        if len(self.audios) and name in self.audios:
            self.full_audio, self.sample_rate = self.audios[name]
            self.audio = self.full_audio.copy()
            self.prev_time = None
            self._stop()
            self.playing = True

    def randomize_audio(self):
        new_audio = random.choice(tuple(self.audios.keys()))
        self.set_audio(new_audio)
