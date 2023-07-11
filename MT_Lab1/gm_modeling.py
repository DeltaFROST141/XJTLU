#!/usr/bin/env python
# -*- coding: utf-8 -*-
##
# @file     gm_modeling.py
# @author   Kyeong Soo (Joseph) Kim <kyeongsoo.kim@gmail.com>
# @date     2020-04-18
#
# @brief    Estimate the parameters of the Guilbert model (GM) using
#           the method proposed by Gilbert [1] and do a comparative
#           analysis between the real and the synthetic loss traces.
#
# @remarks  References
#           [1] E. N. Gilbert, "Capacity of a burst-noise channel,"
#               Bell System Technical Journal, vol. 39, no. 5,
#               pp. 1253–1265, Sep. 1960.
#

import numpy as np
import os
import re
import sys
# import private modules, which are assumed to be in the same directory
sys.path.insert(0, os.getcwd())
from binary_runlengths import binary_runlengths
from model_analysis import rl_histograms, rl_psds
from gm_generate import gm_generate


def gm_parameters(x):
    """
    Estimate GM parameters 'p', 'q', and 'h' for the loss trace.

    Args:
        x (numpy array): loss trace (1 -> loss)
    """
    y = ''.join(map(str, x))
    
    # find overlapping matching by lookaread assertion:
    # https://stackoverflow.com/questions/5616822/how-to-use-regex-to-find-all-overlapping-matches
    n1 = len(re.findall('(?=1)', y)) # or "n1 = sum(x1)"
    n0 = len(re.findall('(?=0)', y)) # or "n0 = len(x1) - sum(x1)"
    n10 = len(re.findall('(?=10)', y))
    n11 = len(re.findall('(?=11)', y))
    n111 = len(re.findall('(?=111)', y))
    n101 = len(re.findall('(?=101)', y))

    a = n1/(n0 + n1) # P(1)
    b = n11/(n10 + n11) # P(1|1)
    try:
        c = n111/(n101 + n111)
    except ZeroDivisionError as e:
        print("Error: " + str(e))
        print("Now we use the alternative estimation based on h=0.5.\n")
        h = 0.5 # see page 1261 of [1]
        q = 1 - 2*b
        p = a*q/(1 - h - a)
        return p, q, h

    q = 1 - (a*c - b**2)/(2*a*c - b*(a + c))
    h = 1 - b/(1 - q)
    p = a*q/(1 - h - a)

    return p, q, h


if __name__ == "__main__":
    fnames = [
        "dataset-A-adsl5-cbr1.0-20091011-035000",
        "dataset-A-adsl1-cbr6.0-20090628-223500", 
    ]
    for fname in fnames:
        x1 = np.fromfile(fname+".bitmap", dtype=int, sep=" ")       
        
        # GM parameters
        p, q, h = gm_parameters(x1)
        print(f'p={p:.4E}, q={q:.4E}, h={h:.4E}')

        # histograms and PSDs
        x2 = gm_generate(len(x1), p, q, h)
        rl_histograms(fname, x1, "GM", x2)
        rl_psds(fname, x1, "GM", x2)
