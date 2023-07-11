#!/usr/bin/env python
# -*- coding: utf-8 -*-
##
# @file     model_analysis.py
# @author   Kyeong Soo (Joseph) Kim <kyeongsoo.kim@gmail.com>
# @date     2020-04-20
#
# @brief    Compare the real and the synthetic loss traces using
#           histograms and PSDs.
#

import matplotlib.pyplot as plt
import numpy as np
import os
import sys
# import private modules, which are assumed to be in the same directory
sys.path.insert(0, os.getcwd())
from binary_runlengths import binary_runlengths


# global settings
plt.rcParams["font.family"] = "Times New Roman"


def rl_histograms(tname, x1, mname, x2):
    """
    Generate runlength historgrams for real and synthetic loss traces.

    Args:
        tname (str): loss trace name
        x1 (numpy array): real loss trace (1 -> loss)
        mname (str): model name
        x2 (numpy array): synthetic loss trace (1 -> loss)
    """
    rls = [None]*4
    rls[0], rls[1] = binary_runlengths(x1)
    rls[2], rls[3] = binary_runlengths(x2)
    max0 = max(max(rls[0]), max(rls[2]))
    max1 = max(max(rls[1]), max(rls[3]))
    bins = [None]*2
    bins[0] = np.arange(1, max0+2) - 0.5 # zl's
    bins[1] = np.arange(1, max1+2) - 0.5 # ol's
    lbls = ["Real", mname]
    titles = ["Zero", "One"]

    fig = plt.figure()
    axs = []
    axs.append(fig.add_subplot(2, 2, 1)) # zl1
    axs.append(fig.add_subplot(2, 2, 2)) # ol1
    axs.append(fig.add_subplot(2, 2, 3, sharex=axs[0], sharey=axs[0])) # zl2
    axs.append(fig.add_subplot(2, 2, 4, sharex=axs[1], sharey=axs[1])) # ol2
    for i in range(len(axs)):
        axs[i].hist(rls[i], bins=bins[i%2], histtype='bar', rwidth=0.8, label=lbls[i//2])
        axs[i].legend(loc='upper right')
        if i < 2:
            axs[i].set_title(titles[i%2])
    fig.supxlabel('Run lengths')
    fig.supylabel('Frequency')
    plt.tight_layout()
    plt.savefig(tname+"_"+mname.lower()+"_hist.pdf", format="pdf", bbox_inches="tight")
    plt.close('all')


def rl_psds(tname, x1, mname, x2):
    """
    Generate PSDs for real and synthetic loss traces.

    Args:
        tname (str): loss trace name
        x1 (numpy array): real loss trace (1 -> loss)
        mname (str): model name
        x2 (numpy array): synthetic loss trace (1 -> loss)
    """
    xs = [x1, x2]
    lbls = ["Real", mname]
    
    fig = plt.figure()
    axs = []
    axs.append(fig.add_subplot(2, 1, 1))
    axs.append(fig.add_subplot(2, 1, 2, sharex=axs[0], sharey=axs[0]))
    for i in range(len(axs)):
        axs[i].psd(xs[i], NFFT=1024, label=lbls[i])
        axs[i].legend(loc='upper right')
        # axs[i].set_ylim(-40, 5)
        axs[i].set_xlabel('')
        axs[i].set_ylabel('')
    fig.supxlabel('Normalized Frequency')
    fig.supylabel('Power Spectral Density [dB/Hz]')
    plt.tight_layout()
    plt.savefig(tname+"_"+mname.lower()+"_psd.pdf", format="pdf", bbox_inches="tight")
    plt.close('all')