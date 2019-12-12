#!/bin/bash

PYTHON=/Users/hiroki-kt/.pyenv/versions/anaconda3-5.1.0/envs/Reseach_envs/bin/python

SCRIPT_DIR=$(cd $(dirname $0); pwd)
echo $SCRIPT_DIR

for num in {0..1}; do
    PYTHON main_nonpara.py
    echo $num"回目のループです"
done
