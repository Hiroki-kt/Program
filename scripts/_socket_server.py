# coding:utf-8
import pyaudio
import wave
import sys
# import os
# import numpy as np
import socket
import threading
from datetime import datetime
# from _recode_func import RecodeFunc

HOST_IP = "163.221.139.121"  # サーバーのIPアドレス
PORT = 12345  # 使用するポート
CLIENTNUM = 3  # クライアントの接続上限数
DATESIZE = 1024  # 受信データバイト数


class SocketServer:
    def __init__(self, host, port):
        self.host = host
        self.port = port
    
    def run_server(self):
        # サーバー起動
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind((self.host, self.port))
            server_socket.listen(CLIENTNUM)
            print('[{}] run server'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            
            while True:
                client_socket, address = server_socket.accept()  # クライアントからの接続要求受け入れ
                print('[{0}] connect client -> address : {1}'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                                                     address))
                client_socket.settimeout(60)
                t = threading.Thread(target=self.conn_client, args=(client_socket, address))
                t.setDaemon(True)
                t.start()  # クライアントごとにThread起動 send/recvのやり取りをする

    def conn_client(self, client_socket, address):
        with client_socket:
            while True:
                rcv_data = client_socket.recv(DATESIZE)  # クライアントからデータ受信
                if rcv_data:
                    print('[{0}] recv date : {1}'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                                         rcv_data.decode('utf-8')))
                    if format(rcv_data.decode('utf-8')) == 'go':
                        play_rec('../2_up_tsp_1num_stereo.wav', 1.2, client_socket)
                    client_socket.send(rcv_data)  # データ受信したデータをそのままクライアントへ送信
                else:
                    break
    
        print('[{0}] disconnect client -> address : {1}'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                                                address))
        
    
# def wave_save(data, channels=1, width=2, sampling=44100, wave_file='./out_put.wav'):
#     wf = wave.Wave_write(wave_file)
#     wf.setnchannels(channels)
#     wf.setsampwidth(width)
#     wf.setframerate(sampling)
#     wf.writeframes(data)
#     wf.close()
#     # print('saved', wave_file)


def play_rec(out_file_name, recode_second, client, CHUNK=1024):
    wf = wave.open(out_file_name, 'rb')
    sampling = wf.getframerate()
    channels = 8
    p = pyaudio.PyAudio()

    stream1 = p.open(format=pyaudio.paInt16,
                     channels=channels,
                     rate=sampling,
                     frames_per_buffer=CHUNK,
                     input=True,
                     # input_device_index=index,
                     )

    stream2 = p.open(format=pyaudio.paInt16,
                     channels=2,
                     rate=sampling,
                     frames_per_buffer=CHUNK,
                     output=True,
                     output_device_index=5
                     )

    if sampling * recode_second < wf.getnframes():
        print('Error recode time is not enough')
        sys.exit()

    elif sampling * recode_second > wf.getnframes() * 2:
        print('Error recode time is too long')
        sys.exit()

    else:
        out_data = wf.readframes(CHUNK)
        in_data = stream1.read(CHUNK)
        client.send(in_data.encode('utf-8'))
        # recoding_data = [in_data]
        for i in range(0, int(sampling / CHUNK * recode_second)):
            input_data = stream1.read(CHUNK)
            client.send(input_data.encode('utf-8'))
            # recoding_data.append(input_data)
            if out_data != b'':
                stream2.write(out_data)
                out_data = wf.readframes(CHUNK)
        # recoded_data = b''.join(recoding_data)
        # print(type(recoded_data))
        # wave_save(recoded_data, channels=channels, sampling=sampling, wave_file='../test_output.wav')

        stream1.stop_stream()
        stream2.stop_stream()
        stream1.close()
        stream2.close()
        p.terminate()


if __name__ == "__main__":
    SocketServer(HOST_IP, PORT).run_server()