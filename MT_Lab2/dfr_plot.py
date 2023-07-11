#!/usr/bin/env python
# -*- coding: utf-8 -*-
##
# @file     dfr_plot.py
# @author   Kyeong Soo (Joseph) Kim <kyeongsoo.kim@gmail.com>
# @date     2023-05-14
#
# @brief Plot the results of decodable frame rate (DFR) simulations.
#


import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys
sys.path.insert(0, os.getcwd()) # the modules below need to be in the current directory.
from dfr_simulation import dfr_simulation


# global settings
plt.rcParams["font.family"] = "Times New Roman"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-R",
        "--random_seed",
        help="seed for numpy random number generation; default is 406",
        default=777,
        type=int)
    
    # debugging
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--no-debug', dest='debug', action='store_false')
    parser.set_defaults(debug=False)

    args = parser.parse_args()

    # check if there is a CSV file 'dfr_plot.csv'.
    if os.path.isfile('dfr_plot.csv') and (os.path.getmtime('dfr_plot.csv') > os.path.getmtime(__file__)):
        # read data into a Pandas DataFrame.
        df = pd.read_csv('dfr_plot.csv')
    else:
        # create a new Pandas DataFrame.
        df = pd.DataFrame({'ec':pd.Series(dtype='int'), # error control
                           'pl': pd.Series(dtype='float'), # loss probability
                           'dfr': pd.Series(dtype='float')}) # decodable frame rate
        # run new simulations.
        for i in range(3):
            for j in range(10):
                match i:
                    case 0:
                        fec = False
                        ci = False
                    case 1:
                        fec = True
                        ci = False
                    case 2:
                        fec = True
                        ci = True
                pl = 1e-4*(j+1)

                # N.B.: reset random seed each time for consistent random
                # number generation among different error control options.
                np.random.seed(args.random_seed)
                
                dfr = dfr_simulation(
                    10000,
                    pl,
                    "starWars4_verbose",
                    fec,
                    ci)
                df.loc[len(df)] = [i, pl, dfr]
                if args.debug:
                    print(f"{i=:d}\t{pl=:.1E}\t{dfr=:.4E}")

        # save the DataFrame to a CSV file.
        df.to_csv('dfr_plot.csv', index=False)
            
    # plot DFRs
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    x = df[df['ec']==0]['pl'].to_numpy()
    ax.set_xticks(x)
    ax.grid(which='major', alpha=0.5)
    colors_symbols = ['bo', 'g^', 'rv']
    labels = ['No Error Control', 'With FEC', 'With FEC & CI']
    for i in range(3):
        plt.plot(x, df[df['ec']==i]['dfr'], colors_symbols[i], label=labels[i])
    plt.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
    plt.xlabel(r'Packet Loss Rate ($p_{L}$)')
    plt.ylabel(r'Decodable Frame Rate ($Q$)')
    plt.legend(loc='lower left')
    plt.savefig('dfr_plot.pdf')
    plt.show()
    