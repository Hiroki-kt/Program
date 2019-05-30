#coding:utf-8
import numpy as np
import matplotlib as plt

from beamforming import beamforming

def pubSystem():
    bm = beamforming('./config.ini')
    bm.steering_vector()

if __name__ == '__main__':
    pubSystem()