# coding:utf-8
import pyaudio
import wave
import numpy as np
import os
from datetime import datetime


def get_index(device_name):
    audio = pyaudio.PyAudio()
    audio_num = audio.get_device_count()
    print('searching:' + device_name + '......')
    for x in range(0, audio_num):
        device_info = audio.get_device_info_by_index(x)
        if device_name in device_info.values():
            print('find mic')
            # print(device_info)
            channels = device_info['maxInputChannels']
            return x, channels
    print('can not find:' + device_name)


class RecodeSound:
    def __init__(self):
        device_name = 'ReSpeaker 4 Mic Array (UAC1.0)'
        self.index, self.channels = get_index(device_name)

    def recode_sound(self):
        chunk = 1024
        sound_format = pyaudio.paInt16
        channels = self.channels
        sampling_rate = 44100
        recode_seconds = 16.2
        index = self.index

        threshold = 0.01  # しきい値

        p = pyaudio.PyAudio()

        stream = p.open(format=sound_format,
                        channels=channels,
                        rate=sampling_rate,
                        input=True,
                        input_device_index=index,
                        frames_per_buffer=chunk
                        )

        print("Complete setting of recode!")

        # 録音処理
        file_path = '../_exp/19' + datetime.today().strftime("%m%d") + '/recode_data/'
        my_makedirs(file_path)

        while True:
            data = stream.read(chunk)
            x = np.frombuffer(data, dtype="int16") / 32768.0
            # print(x.max())
            if x.max() > threshold:
                print('Recode start!')
                print('Now, recording....')
                # file_path = '../_exp/19' + datetime.today().strftime("%m%d") + '/recode_data/'
                filename = file_path + datetime.today().strftime("%H%M%S") + ".wav"
                recording_data = [data]
                for i in range(0, int(sampling_rate / chunk * recode_seconds)):
                    data = stream.read(chunk)
                    recording_data.append(data)
                data = b''.join(recording_data)

                out = wave.open(filename, 'w')
                out.setnchannels(channels)
                out.setsampwidth(2)
                out.setframerate(sampling_rate)
                out.writeframes(data)
                out.close()
                print("Saved.")
                break

        # 録音修了処理
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        
def my_makedirs(path):
    if not os.path.isdir(path):
        os.makedirs(path)


if __name__ == '__main__':
    recode = RecodeSound()
    recode.recode_sound()
