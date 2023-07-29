#!/usr/bin/env python
# -*- coding: utf-8 -*-
##
# @file     gm_generate.py
# @author   Kyeong Soo (Joseph) Kim <kyeongsoo.kim@gmail.com>
# @date     2023-04-20
#
# @brief    A function for generating loss pattern based on the
#           Guilbert model (GM).
#

import numpy as np
import random
import sys


def gm_generate(len, p, q, h):
    """
    Generate a binary sequence of 0 (GOOD) and 1 (BAD) of length len
    from the GM specified by transition probabilites 'p' (GOOD->BAD)
    and 'q' (BAD->GOOD) with no loss probability at BAD state 'h'.

    This function assumes that the GM starts in GOOD (0) state.

    Examples:

    seq = gm_generate(100, 0.95, 0.9, 0.5)
    """

    seq = np.zeros(len)

    # check transition probabilites
    if p < 0 or p > 1:
        sys.exit("The value of the transition probability 'p' is not valid.")
    elif q < 0 or q > 1:
        sys.exit("The value of the transition probability 'q' is not valid.")
    elif h < 0 or h > 1:
        sys.exit("The value of the no loss probability at BAD state 'h' is not valid.")
    else:
        tr = [p, q]

    # create a random sequence for state changes
    statechange = np.random.rand(len)

    # Assume that we start in GOOD state (0).
    state = 0

    # main loop
    for i in range(len):
        if statechange[i] <= tr[state]:
            # transition into the other state
            state ^= 1
        # add a binary value to output
        seq[i] = 1 if state == 1 and random.uniform(0, 1) > h else 0
    return seq


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-l",
        "--length",
        help="the length of the loss pattern to be generated; default is 10",
        default=10,
        type=int)
    parser.add_argument(
        "-p",
        help="GOOD to BAD transition probability; default is 0.95",
        default=0.95,
        type=float)
    parser.add_argument(
        "-q",
        help="BAD to GOOD transition probability; default is 0.9",
        default=0.9,
        type=float)
    parser.add_argument(
        "--no_loss",
        help="no loss probability at BAD state; default is 0.5",
        default=0.5,
        type=float)
    args = parser.parse_args()
    print(gm_generate(args.length, args.p, args.q, args.no_loss))
