import pickle, math
import numpy as np
import gurobipy as gp
from gurobipy import *
import time, sys, gc
from queue import PriorityQueue
from numpy import *
import sys
import logging
sys.path.insert(0,'..')
from kuiper import create_testing_selector

import os
import matplotlib.pyplot as plt
from matplotlib import rc

rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=True)

#logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


def plot_cdf(datas, linelabels = None, label = None, y_label = "CDF", name = "ss"):

    def cdf_transfer(X):
        X = sorted(X)
        Y=[]
        l=len(X)
        Y.append(float(1)/l)
        for i in range(2,l+1):
            Y.append(float(1)/l+Y[i-2])
        return X, Y

    _fontsize = 9
    fig = plt.figure(figsize=(2, 1.6))
    ax = fig.add_subplot(111)

    plt.ylabel(y_label, fontsize=_fontsize)
    plt.xlabel(label, fontsize=_fontsize)

    colors = [
    'blueviolet',
    'black',#(0.9333333333333333, 0.5215686274509804, 0.2901960784313726),
    (1.0, 0.589256862745098, 0.0130736618971914),
    'black',
    'red']

    linetype = ['-', '--', '--', ':', ':' ,':']
    markertype = ['o', '|', '+', 'x']

    X_max = -1
    index = -1
    for i, data in enumerate(datas):
        index += 1
        X, Y = cdf_transfer(data)
        plt.plot(X,Y, linetype[i%len(linetype)], color=colors[i%len(colors)], label=linelabels[i], linewidth=1.)#, marker = markertype[i%len(markertype)])
        X_max = max(X_max, max(X))

    plt.ylim(0, 1)
    plt.xlim(0, 1000)

    plt.yticks([0.00, 0.25, 0.50, 0.75, 1.00], fontsize=_fontsize)
    plt.xticks([250 * i for i in range(5)], fontsize=_fontsize)
    plt.xticks(fontsize=_fontsize)

    legend_properties = {'size':8}

    plt.legend(
        handletextpad=0.4,
        prop = legend_properties,
        handlelength=1.5,
        frameon = False)

    ax.tick_params(axis="both", direction="in", length = 2, width = 1)

    # Remove frame top and right
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout(pad=0.2, w_pad=0.03, h_pad=0.01)
    plt.savefig(name)

def load_profiles(datafile, sysfile):
    # load user data information
    # num_of_clients x num_of_classes
    datas = pickle.load(open(datafile, 'rb'))

    # load user system information
    systems = pickle.load(open(sysfile, 'rb'))

    # Generate global data distribution
    distr = datas.sum(axis=0)
    distr /= float(distr.sum())

    return datas, systems, distr

def run_query():
    """
    Generate queries for fig 17 and plot results
    """
    data, systems, distr = load_profiles('openimg_distr.pkl', 'client_profile.pkl')

    budgets = [100, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
    query_samples = [1000 * i for i in range(1, 11)] + [10000 * i for i in range(2, 20)]

    selector = create_testing_selector(data_distribution=data, client_info=systems, model_size=65536)

    failed_queries = []

    #============ Run Kuiper  =============#
    kuiper_results = []
    for budget in budgets:
        for req in query_samples:
            #logging.info("Running budget " + str(budget) + " query_samples " + str(req))
            print("Running budget " + str(budget) + " query_samples " + str(req))
            req_list = req * distr
            client_sample_matrix, test_duration, lp_overhead = selector.select_by_category(
                                        req_list, max_num_clients=budget, greedy_heuristic=True)

            #  test_duration == -1 indicates failure                           
            if test_duration != -1:
                kuiper_results.append(test_duration+lp_overhead)
            else: 
                failed_queries.append((budget, req))


    #============ Run MILP =============#
    # E2E = test_durationn + lp_overhead
    lp_results = []
    for budget in budgets:
        for req in query_samples:
            
            # Skip failed queries
            if (budget, req) in failed_queries:
                continue

            #logging.info("Running budget " + str(budget) + " query_samples " + str(req))
            print("Running budget " + str(budget) + " query_samples " + str(req))
            req_list = req * distr
            client_sample_matrix, test_duration, lp_overhead = selector.select_by_category(
                                        req_list, max_num_clients=budget, greedy_heuristic=False)
            if test_duration != -1:
                lp_results.append(test_duration+lp_overhead)



    #============ Plot E2E time =============#
    plot_cdf([kuiper_results, lp_results], ['Kuiper', 'MILP'], "End-to-End Time (s)", "CDF across Queries", "testing_e2e.pdf")

run_query()
