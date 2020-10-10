# -*- coding:utf-8 -*-
from __future__ import division
import logging
import numpy as np
from multiprocessing import *
# from multiprocessing import Pool
import pysam
import math
import argparse
import logging
import vcf
import os
import pandas as pd
import shutil
import pandas as pd
import copy
import csv, os, argparse, string, sys, multiprocessing, logging, functools, time, signal, re

sys.dont_write_bytecode = True
csv.register_dialect("line_terminator", lineterminator="\n")
VERSION = '1.0.1'



def get_indel_work_sh(root_dir,work_dir):
    f = open(work_dir+'/getIndelWork00000.sh', 'w')
    
    list = os.listdir(root_dir)  # 列出文件夹下所有的目录与文件
    j = 0
    
    for i in range(0, len(list)):
        if j < 40:
            
            path = os.path.join(root_dir, list[i])
            fileName = path.split('/')[-1]
            #print path
            if path.endswith("vcf"):
                j +=1
                f.write("vcftools --vcf " + path + " --keep-only-indels --out " + work_dir+'/'+fileName + ".indel"+" --recode --recode-INFO-all" + "\n")
    f.close()



if __name__ == "__main__":
    #parser = argparse.ArgumentParser()
    #parser.add_argument('-i', '--input', help='input root dir', required=True)
    #parser.add_argument('-o', '--output', help='output work dir', required=True)
    #args = parser.parse_args()
    #root_dir = os.path.abspath(args.input)
    #work_dir = os.path.abspath(args.output)
    #get_indel_work_sh(root_dir,work_dir)
    
    f = open('emddn.sh', 'w')
    for i in range(0, 50):
        f.write('/usr/local/bin/python3 emddNew3.py -n 4 -s '+str(i)+"\n")
    f.close()


