# coding:utf-8
import pyaudio
import wave
import numpy as np
import os
from datetime import datetime
try:
    from msvcrt import getch  # Windowsでは使えるらしい
except ImportError:
    def getch():
        import sys
        import tty
        import termios
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            return sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)


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

    def recode_sound(self, key_input=False):
        chunk = 1024
        sound_format = pyaudio.paInt16
        channels = self.channels
        sampling_rate = 44100
        recode_seconds = 17
        index = self.index
        NEXT = 110
        CTRL_C = 3

        threshold = 0.2  # しきい値

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
        file_path = '../_exp/19' + datetime.today().strftime("%m%d") + '/recode_data/' + \
                    datetime.today().strftime("%H%M%S") + "/"
        my_makedirs(file_path)

        deg_list = [-50, -40, -30, -20, -10, 0, 10, 20, 30, 40]
        # deg_list = [k + 8 for k in deg_list]
        deg_list = [22, 32, 42, 24]
        count = 0
        while True:
            data = stream.read(chunk)
            x = np.frombuffer(data, dtype="int16") / 32768.0
            # print(x.max())
            if x.max() > threshold:
                print('Recode start!')
                print('Now, recording....')
                # file_path = '../_exp/19' + datetime.today().strftime("%m%d") + '/recode_data/'
                filename = file_path + str(deg_list[count]) + ".wav"
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
                print("Saved." + str(deg_list[count]) + ".wav")
                if key_input:
                    print("Press key crtl+c(finish) or other(continue)")
                    key = ord(getch())
                    if key == CTRL_C:
                        break
                    else:
                        while True:
                            if key == NEXT:
                                print("NEXT")
                                stream.stop_stream()
                                break
                            else:
                                message = 'input, {0}'.format(chr(key))
                                print(message)
                                key = ord(getch())
                if deg_list[count] == 49:
                    break
                count += 1
                # if count == 10:
                #     deg_list = [k + 1 for k in deg_list]
                #     count = 0
                
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
