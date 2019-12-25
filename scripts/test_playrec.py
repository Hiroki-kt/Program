import sounddevice as sd
from _function import MyFunc as mf
import numpy as np
import pyaudio
import wave
import sys
from _sound_recoder import RecodeSound

data, channels, sampling, frames = mf.wave_read_func('../_exp/Speaker_Sound/up_tsp_1num.wav')


def sounddevice_module(data):
    # 諦めた...けど一応残してるよ
    data = np.append(data, data, axis=0).T
    print("start")
    sd.default.device = [3, 1]
    sd.rec(int(1.5 * sampling), samplerate=sampling, channels=6)
    sd.play(data, sampling, blocking=True)
    while True:
        if sd.wait() is None:
            break
    test = sd.get_stream()
    print(test.shape)
    # mf.wave_save(myrecoding, channels=6)


def pyaudio_module():
    file_name = '../_exp/Speaker_Sound/up_tsp_1num.wav'
    device_name = 'ReSpeaker 4 Mic Array (UAC1.0)'
    recode_second = 1.2
    wf = wave.open(file_name, 'rb')
    index, channels = mf.get_mic_index(device_name)
    CHUNK = 1024
    RATE = 44100
    p = pyaudio.PyAudio()
    
    stream1 = p.open(format=pyaudio.paInt16,
                     channels=channels,
                     rate=RATE,
                     frames_per_buffer=CHUNK,
                     input=True,
                     input_device_index=index,
                     )
    
    stream2 = p.open(format=pyaudio.paInt16,
                     channels=1,
                     rate=RATE,
                     frames_per_buffer=CHUNK,
                     output=True
                     )

    out_data = wf.readframes(CHUNK)
    in_data = stream1.read(CHUNK)
    recoding_data = [in_data]
    if sampling * recode_second < wf.getnframes():
        print('Error recode time is not enough')
        sys.exit()
    
    elif sampling * recode_second > wf.getnframes() * 2:
        print('Error recode time is too long')
        sys.exit()
    
    else:
        for i in range(0, int(sampling / CHUNK * recode_second)):
            input_data = stream1.read(CHUNK)
            recoding_data.append(input_data)
            if out_data != b'':
                stream2.write(out_data)
                out_data = wf.readframes(CHUNK)
        recoded_data = b''.join(recoding_data)
        print(type(recoded_data))
        mf.wave_save(recoded_data, channels=channels, sampling=sampling)
        
        stream1.stop_stream()
        stream2.stop_stream()
        stream1.close()
        stream2.close()
        p.terminate()


if __name__ == '__main__':
    pyaudio_module()
