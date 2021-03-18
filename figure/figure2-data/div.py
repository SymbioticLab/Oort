import matplotlib.pyplot as plt
import numpy as np
from numpy import *
import re, math
from matplotlib import rcParams
import pandas as pd
import matplotlib, csv
from matplotlib import rc
import dit
import pickle
from scipy.spatial import distance

rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=True)


def cdf_transfer(X):
    X = sorted(X)
    Y=[]
    l=len(X)
    Y.append(float(1)/l)
    for i in range(2,l+1):
        Y.append(float(1)/l+Y[i-2])

    return X, Y

def plot_cdf(datas, linelabels = None, label = None, y_label = "CDF", name = "ss"):
    _fontsize = 8
    fig = plt.figure(figsize=(2, 1.6)) # 2.5 inch for 1/3 double column width
    #fig = plt.figure(figsize=(3.7, 2.4))
    ax = fig.add_subplot(111)

    plt.ylabel(y_label, fontsize=9)
    plt.xlabel(label, fontsize=9)

    # Using seaborn color palatte "muted"
    colors = [
    (1.0, 0.589256862745098, 0.0130736618971914),#(0.9333333333333333, 0.5215686274509804, 0.2901960784313726),
    '#1874CD',
    'grey',
    'black']#(0.31,0.443,0.745),]
    # colors = [
    # '#1874CD',
    # 'black',
    # #'grey',#(0.9333333333333333, 0.5215686274509804, 0.2901960784313726),
    # (1.0, 0.589256862745098, 0.0130736618971914),
    # (0.31,0.443,0.745),]

    linetype = ['-.', '-', '--', ':', ':' ,':']
    markertype = ['o', '|', '+', 'x']

    X_max = -1
    index = -1
    for i, data in enumerate(datas):
        index += 1
        print(index)
        X, Y = cdf_transfer(data)
        plt.plot(X,Y, linetype[i%len(linetype)], color=colors[i%len(colors)], label=linelabels[i], linewidth=1.2)#, marker = markertype[i%len(markertype)])
        X_max = max(X_max, max(X))

    plt.ylim(0, 1)
    plt.xlim(0, 1)
    plt.yticks([0.00, 0.25, 0.50, 0.75, 1.00], fontsize=9)
    plt.xticks([0.25, 0.50, 0.75, 1.00],fontsize=9)

    #plt.gca().xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    # plt.grid(linestyle = ':' ,linewidth = 1)

    legend_properties = {'size':_fontsize}

    plt.legend(
        loc=(0.45, 0.02),#'lower right',
        # loc=(0.03, 0.55),
        # bbox_to_anchor=(1, 0.75),
        handletextpad=0.4,
        #bbox_to_anchor=(0.5, 0.52),
        prop = legend_properties,
        handlelength=1.5,
        frameon = False)

    ax.tick_params(axis="both", direction="in", length = 2, width = 1)

    # Remove frame top and right
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    #plt.tight_layout(pad=0.1, w_pad=0.01, h_pad=0.01)

    #ax.set_xscale('log')
    plt.tight_layout(pad=0.2, w_pad=0.01, h_pad=0.01)

    #plt.xlim(0, 1.21e6)#, 150)
    #plt.xlim(0, 1.0)
    #plt.xlim(0, 2000)
    #plt.show()

    plt.savefig(name)

def read_pkl(fname):
    with open(fname + '_div.pkl', 'rb') as f:
        div = np.array(pickle.load(f))
    
    res = sorted(div)
    upper = int(len(res) * 0.90)
    lower = int(len(res) * 0.10)
    res = res[lower:upper]
    nres = [float(x - res[0])/float(res[-1] - res[0]) for x in res]
    return np.array(nres)

print("Loading Speech")
speech_div = read_pkl('speech')
print(speech_div.shape)
print("Loading Stack Overflow")
so_div = read_pkl('stackoverflow')
print("Loading reddit")
reddit_div = read_pkl('reddit')
print("Loading openimg")
openimg_div = read_pkl('openimg')

# plot_cdf([openimg_size, so_size, reddit_size, speech_size], ['OpenImage', 'StackOverflow', 'Reddit', 'Speech'], "Normalized Data Size", "CDF across Clients", "cSample.pdf")
plot_cdf([openimg_div, so_div, reddit_div, speech_div], ['OpenImage', 'StackOverflow', 'Reddit', 'Speech'], "Normalized Data Size", "CDF across Clients", "cDiv.pdf")