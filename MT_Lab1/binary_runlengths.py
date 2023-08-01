#!/usr/bin/env python
# -*- coding: utf-8 -*-
##
# @file     binary_runlengths.py
# @author   Kyeong Soo (Joseph) Kim <kyeongsoo.kim@gmail.com>
# @date     2020-03-25
#
# @brief    A function for obtaining run lengths from a binary sequence.
#

import numpy as np


def binary_runlengths(seq):
    """
    Generates a sequence of run lengths for zero (zerorl) and one
    (onerl), respectively, for a given binary sequence seq.
    """

    seq = (np.asarray(seq)).flatten()  # make sure seq is numpy 1D array

    w = np.concatenate(([1], seq, [1]))  # add 1 before and after seq
    zerorl = np.nonzero(np.diff(w) == 1)[0] - np.nonzero(np.diff(w) == -1)[0]

    w = np.concatenate(([0], seq, [0]))  # auxiliary vector
    onerl = np.nonzero(np.diff(w) == -1)[0] - np.nonzero(np.diff(w) == 1)[0]

    return (zerorl, onerl)
