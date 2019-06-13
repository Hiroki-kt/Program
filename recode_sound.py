#coding:utf-8
#!/usr/bin/env python
'''
ある一定値以上の音がなると規定された時間だけ音を録音するプログラム
'''

import pyaudio
import wave
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime

class RecodeSound():
    def __init__(self):
        pass

    def recodeSound(self):
        chunk = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 6
        RATE = 44100
        RECODE_SECONDS = 5
        index = 3
        file_path = 'output.wav'

        threshold = 10  # しきい値

        p = pyaudio.PyAudio()

        stream = p.open(format = FORMAT,
                        channels = CHANNELS,
                        rate = RATE,
                        input = True,
                        input_device_index = index,
                        frames_per_buffer = chunk
                        )

        print("recording start...")

        # 録音処理
        frames = []
        for i in range(0, int(RATE/chunk * RECODE_SECONDS)):
            data = stream.read(chunk)
            frames.append(data)

        print("recording end...")

        # 録音修了処理
        stream.stop_stream()
        stream.close()
        p.terminate()

        # 録音データをファイルに保存
        wav = wave.open(file_path, 'wb')
        wav.setnchannels(CHANNELS)
        wav.setsampwidth(p.get_sample_size(FORMAT))
        wav.setframerate(RATE)
        wav.writeframes(b''.join(frames))
        wav.close()

if __name__ == '__main__':
    recode = RecodeSound()
    recode.recodeSound()