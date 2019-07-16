import numpy as np
import wave
import struct
# from matplotlib import pyplot as plt

FRAME = 'sinwave_440hz.wav'
CHANNELS = 1  # チャンネル数：モノラル
WIDTH = 2  # 量子化精度：2byte=16bit=65,536段階=±32,767
SAMPLING_RATE = 44100  # サンプリング周波数：44.1kHz

freq = 440  # 基本周波数：440Hz(A4)
time = 0.2  # 録音時間：3秒間
time_rest = 3
samples = int(time * SAMPLING_RATE)  # サンプル数
samples_rest = int(time_rest * SAMPLING_RATE)
# f_list = np.linspace(100, 5000, int((5000 - 100) / 20))
f_list = (300, 500, 1000, 2000, 3000)
print(len(f_list))
# sound_samples = int((1 + 2 * len(f_list)) * samples)
sound_samples = int((1 + len(f_list)) * samples + (len(f_list)) * samples_rest)
print('sound sample', sound_samples)
print('sound second', sound_samples/SAMPLING_RATE)

t = np.linspace(0, time, samples + 1)
t_rest = np.linspace(0, time_rest, samples_rest + 1)
s = 32767 * np.sin(2 * np.pi * 800 * t)
for i, name in enumerate(f_list):
    s_1 = 32767 * np.sin(2 * np.pi * name * t)
    rest = np.zeros((len(t_rest),))
    s = np.append(s, rest)
    s = np.append(s, s_1)
s = np.rint(s)
s = s.astype(np.int16)
print(s.shape)

s = s[0:sound_samples]
data = struct.pack("h" * sound_samples, *s)  # ndarrayからbytesオブジェクトに変換

wf = wave.open(FRAME, 'w')
wf.setnchannels(CHANNELS)
wf.setsampwidth(WIDTH)
wf.setframerate(SAMPLING_RATE)
wf.writeframes(data)
wf.close()
