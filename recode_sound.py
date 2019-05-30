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
        CHANNELS = 1
        RATE = 44100
        RECODE_SECONDS = 2

        threshold = 3000  # しきい値

        p = pyaudio.PyAudio()

        stream = p.open(format = FORMAT,
                        channels = CHANNELS,
                        rate = RATE,
                        input = True,
                        frames_per_buffer = chunk
                        )
        cnt = 0

        while True:
            # 音を読む感じであれば、おまじないてきだね
            data = stream.read(chunk)
            # bufferからarrayに変換
            x = np.frombuffer(data, dtype="int16")/32768.0

            '''フーリエ変換して周波数領域になおしてみる'''
            f = np.fft.fft(x)

            a = [np.asscalar(i) for i in x]

            print(a[0])

            if x.max() > threshold:
                filename = datetime.today().strftime("%Y%m%d%H%M%S") + ".wav"
                print(cnt, filename)

                all = []
                all.append(data)
                for i in range(0, int(RATE / chunk * int(RECODE_SECONDS))):
                    data = stream.read(chunk)
                    all.append(data)
                data = b''.join(all)

                out = wave .open(filename, 'w')
                out.setnchannels(CHANNELS)
                out.setsampwidth(2)
                out.setframerate(RATE)
                out.writeframes(data)
                out.close()

                print("saved.")

                cnt += 1
            if cnt > 5:
                break

        # stop stream
        stream.close()
        # close pyaudio
        p.terminate()

if __name__ == '__main__':
    recode = RecodeSound()
    recode.recodeSound()