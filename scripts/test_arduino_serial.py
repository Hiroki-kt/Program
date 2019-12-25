# coding:utf-8

import serial
import time


def main():
    # COMポートを開く
    print("OPEN PORT")
    ser = serial.Serial('/dev/cu.usbmodem14101', 9600)
    while True:
        ser.write(b"1")
        time.sleep(1)
        
        ser.write(b"0")
        time.sleep(1)


if __name__ == '__main__':
    main()