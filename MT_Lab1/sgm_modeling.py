#!/usr/bin/env python
# -*- coding: utf-8 -*-
##
# @file     sgm_modeling.py
# @author   Kyeong Soo (Joseph) Kim <kyeongsoo.kim@gmail.com>
# @date     2020-04-18
#
# @brief    Estimate the parameters of the simple Guilbert model (SGM)
#           using the method proposed by Yajnik et al. [1] and do a
#           comparative analysis between the real and the synthetic
#           loss traces.
#
# @remarks  References
#           [1] M. Yajnik, S. Moon, J. Kurose, and D. Towsley,
#               "Measurement and modelling of the temporal dependence in packet loss,"
#               in Proc. 1999 IEEE INFOCOM, vol. 1, Mar. 1999, pp. 345–352.
#

import numpy as np
import os
import re
import sys
# import private modules, which are assumed to be in the same directory
sys.path.insert(0, os.getcwd())
from binary_runlengths import binary_runlengths
from model_analysis import rl_histograms, rl_psds
from sgm_generate import sgm_generate


def sgm_parameters(x):
    """
    Estimate SGM parameters 'p' and 'q' for the loss trace.

    Args:
        x (numpy array): loss trace (1 -> loss)
    """
    y = ''.join(map(str, x))
    
    # find overlapping matching by lookaread assertion:
    # https://stackoverflow.com/questions/5616822/how-to-use-regex-to-find-all-overlapping-matches
    n1 = len(re.findall('(?=1)', y)) # or "n1 = sum(x)"
    n0 = len(re.findall('(?=0)', y)) # or "n0 = len(x) - sum(x)"
    n10 = len(re.findall('(?=10)', y))
    n01 = len(re.findall('(?=01)', y))

    p = n01 / n0
    q = n10 / n1

    return p, q


if __name__ == "__main__":
    fnames = [
        "dataset-A-adsl5-cbr1.0-20091011-035000",
        "dataset-A-adsl1-cbr6.0-20090628-223500", 
    ]
    for fname in fnames:
        x1 = np.fromfile(f"{fname}.bitmap", dtype=int, sep=" ")     

        # SGM parameters
        p, q = sgm_parameters(x1)
        print(f'{fname}: p={p:.4E}, q={q:.4E}')

        # histograms and PSDs
        x2 = sgm_generate(len(x1), p, q)
        rl_histograms(fname, x1, "SGM", x2)
        rl_psds(fname, x1, "SGM", x2)
