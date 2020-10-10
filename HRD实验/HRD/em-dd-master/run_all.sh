#!/bin/bash

nohup python3 emdd.py musk1 > musk1_avg.out &
nohup python3 emdd.py musk2 > musk2_avg.out &
nohup python3 emdd.py synth1 > synth1_avg.out &
nohup python3 emdd.py synth4 > synth4_avg.out &
nohup python3 emdd.py dr > dr_avg.out &
nohup python3 emdd.py elephant > elephant_avg.out &
nohup python3 emdd.py fox > fox_avg.out &
nohup python3 emdd.py tiger > tiger_avg.out &
