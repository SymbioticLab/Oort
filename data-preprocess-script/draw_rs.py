#import matplotlib.pyplot as plt
import numpy as np
from numpy import *
import re, math
#import seaborn as sns
#from matplotlib import rcParams
#import pandas as pd
#import matplotlib, csv
#from matplotlib import rc
#from pyemd import emd
import dit
from dit.divergences import jensen_shannon_divergence
import pickle
import random
#from data_div import measureAllDistance

# rc('font',**{'family':'serif','serif':['Times']})
# rc('text', usetex=True)

random.seed(33334)

numOfTries = 20

def cdf_transfer(X):
    X = sorted(X)
    Y=[]
    l=len(X)
    Y.append(float(1)/l)
    for i in range(2,l+1):
        Y.append(float(1)/l+Y[i-2])

    return X, Y


def plot_line(datas, xs, linelabels = None, label = None, y_label = "CDF", name = "ss", _type=-1):
    _fontsize = 10
    fig = plt.figure(figsize=(2.5, 1.8)) # 2.5 inch for 1/3 double column width
    #fig = plt.figure(figsize=(3.7, 2.4))
    ax = fig.add_subplot(111)

    plt.ylabel(y_label, fontsize=_fontsize)
    plt.xlabel(label, fontsize=_fontsize)

    #'limegreen','mediumslateblue','chocolate'
    #colors = ['black', '#FF7F24', 'blue', '#EE1289', 'red', 'blue', 'red', 'red', 'grey', 'pink']
    colors = ['grey', '#1874CD', '#EE7621', 'slateblue', 'DeepPink', '#FF7F24', 'blue', 'blue', 'blue', 'red', 'blue', 'red', 'red', 'grey', 'pink']
    linetype = ['--', '-.', '-', '-', '-' ,':']
    markertype = ['o', '|', '+', 'x']

    X_max = 9999999999999
    Y_max = -1

    X = [i for i in range(len(datas[0]))]

    for i, data in enumerate(datas):
        _type = max(_type, i)
        plt.plot(xs[i], data, linetype[_type%len(linetype)], color=colors[i%len(colors)], label=linelabels[i], linewidth=1.3)
        #plt.fill_between(X, data, alpha=0.3, color=colors[_type%len(colors)])#, marker=markertype[i%len(markertype)], color=colors[i%len(colors)])
        # plt.scatter(X, data, marker=markertype[i%len(markertype)], color=colors[i%len(colors)])
        #plt.scatter(X, data, linetype[i%len(linetype)], color=colors[i%len(colors)], label=linelabels[i], linewidth=1.1)#, marker = markertype[i%len(markertype)])
        X_max = min(X_max, max(xs[i]))
        Y_max = max(Y_max, max(data))

    #plt.ylim(0, 1)
    plt.yticks(fontsize=_fontsize)
    plt.xticks([0.1 * i for i in range(1, 6)], fontsize=_fontsize)

    #plt.gca().xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    #plt.grid(linestyle = ':' ,linewidth = 1)

    legend_properties = {'size':_fontsize}

    plt.legend(
        loc = 'lower right',
        # bbox_to_anchor=(1, 0.75),
        prop = legend_properties,
        frameon = False)
    # plt.legend(bbox_to_anchor=(1, 0.75), prop = legend_properties)
    # plt.legend(bbox_to_anchor=(1.3, 0.7), loc=1, borderaxespad=0., prop = legend_properties, frameon = False)

    #plt.legend(bbox_to_anchor=(0.60, 1.03), loc=1, borderaxespad=0., prop = legend_properties, frameon = False)
    #plt.scatter(1.5, 0.645, marker = 'o', s=400, alpha=0.6, facecolors='none', edgecolors='r')
    #plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0), fontsize=_fontsize, fontweight = 'bold')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    #ax.tick_params(axis="y", direction="in", length = 5, width = 1.2)
    #ax.tick_params(axis="x", direction="in", length = 5, width = 1.2)
    #ax.tick_params(axis="both", direction="in", length = 2, width = 1)

    plt.tight_layout()

    #ax.set_xscale('log')

    plt.tight_layout(pad=0.2, w_pad=0.01, h_pad=0.01)

    plt.xlim(0)
    plt.ylim(0, Y_max)
    #plt.ylim(30, 70)
    #plt.grid()
    #plt.show()

    plt.savefig(name)

def plot_cdf(datas, linelabels = None, label = None, y_label = "CDF", name = "ss"):
    _fontsize = 9.5
    fig = plt.figure(figsize=(2.5, 1.8)) # 2.5 inch for 1/3 double column width
    #fig = plt.figure(figsize=(3.7, 2.4))
    ax = fig.add_subplot(111)

    plt.ylabel(y_label, fontsize=_fontsize)
    plt.xlabel(label, fontsize=_fontsize)

    # Using seaborn color palatte "muted"
    colors = [
    'grey',#(0.9333333333333333, 0.5215686274509804, 0.2901960784313726),
    (1.0, 0.589256862745098, 0.0130736618971914),
    'black',
    'red']#(0.31,0.443,0.745),]

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
    plt.yticks([0.00, 0.25, 0.50, 0.75, 1.00], fontsize=_fontsize)
    plt.xticks([0.25, 0.50, 0.75, 1.00],fontsize=_fontsize)

    #plt.gca().xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    # plt.grid(linestyle = ':' ,linewidth = 1)

    legend_properties = {'size':_fontsize}

    plt.legend(
        loc=(0.55, 0.02),#'lower right',
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

def read_div(name):
    div = []
    index = -1
    with open(name, "r") as f:
        lines = f.readlines()
        for l in lines:
            index += 1
            print(index)
            dig = l.replace(',', ' ').strip().split()
            for d in dig:
                div.append(float(d))
    # normalize all size
    div = sorted(div)

    pickL = int(len(div) * 0.95)
    ndiv = [float(x - div[0])/float(div[pickL]) for x in div]
    return ndiv

def read_from_sum(name, threshold=99999999):
    res = []
    with open(name, 'r') as fin:
        lines = fin.readlines()
        threshold = min(threshold, len(lines))
        for l in lines[:threshold]:
            clientSamples = [int(x) for x in l.strip().split()]
            res.append(sum(clientSamples))

    # normalize all size
    res = sorted(res)
    pickL = int(len(res) * 0.95)
    nres = [float(x - res[0])/float(res[pickL]) for x in res]

    return nres

def read_client_samples(name, threshold=99999999):
    res = []
    totalClients = 0

    with open(name, 'r') as fin:
        lines = fin.readlines()
        threshold = min(threshold, len(lines))
        for l in lines[:threshold]:
            clientSamples = [int(x) for x in l.strip().split()]

            if sum(clientSamples) > 0:
                res.append(clientSamples)
                totalClients += 1

    overall = [0 for x in range(len(res[0]))]

    for client in res:
        for i, c in enumerate(client):
            overall[i] += c

    return res, overall, totalClients

def normalizeList(dataDis):
    tempDataSize = sum(dataDis)
    return dit.ScalarDistribution(np.arange(len(dataDis)), [c / float(tempDataSize) for c in dataDis])

def measureDistance(individual, overall):
    return jensen_shannon_divergence([individual, overall])

def pickSubset(clientSampleList, numOfSample):
    subsetSample = [0 for i in range(len(clientSampleList[0]))]

    for c in range(numOfSample):
        for i, cl in enumerate(clientSampleList[c]):
            subsetSample[i] += cl

    return subsetSample

def openImg():
    # 507
    clientSampleList, allSamples, totalClients = read_client_samples("openImg_size.txt", 507)

    dis = []
    sampleRs = []
    for i in range(numOfTries):
        print('...current openImg trial ' + str(i))
        distances, sampleRatios = draw_rs(clientSampleList, allSamples, totalClients)
        dis.append(distances)
        sampleRs.append(sampleRatios)

    #distances, sampleRatios = draw_ss(clientSampleList, allSamples, totalClients)
    return dis, sampleRs
    #plot_line([distances], [sampleRatios], [''], "Ratio of Clients Sampled", "Divergence to Overall", "diffallImg.pdf")

def quickDraw():
    clientSampleList, allSamples, totalClients = read_client_samples("quickdraw_size.txt", 30000)

    dis = []
    sampleRs = []
    for i in range(numOfTries):
        print('...current quickDraw trial ' + str(i))
        distances, sampleRatios = draw_rs(clientSampleList, allSamples, totalClients)
        dis.append(distances)
        sampleRs.append(sampleRatios)

    #distances, sampleRatios = draw_ss(clientSampleList, allSamples, totalClients)
    return dis, sampleRs

def blog():
    clientSampleList, allSamples, totalClients = readFromSerialized("blog_data_dist.txt", 1000)

    dis = []
    sampleRs = []
    for i in range(numOfTries):
        print('...current blog trial ' + str(i))
        distances, sampleRatios = draw_rs(clientSampleList, allSamples, totalClients)
        dis.append(distances)
        sampleRs.append(sampleRatios)

    #distances, sampleRatios = draw_ss(clientSampleList, allSamples, totalClients)
    return dis, sampleRs
    #plot_line([distances], [sampleRatios], [''], "Ratio of Clients Sampled", "Divergence to Overall", "diffallBlog.pdf")

def email():
    clientSampleList, allSamples, totalClients = readFromSerialized("email_data_dist.txt", 30000)

    dis = []
    sampleRs = []
    for i in range(numOfTries):
        print('...current email trial ' + str(i))
        distances, sampleRatios = draw_rs(clientSampleList, allSamples, totalClients)
        dis.append(distances)
        sampleRs.append(sampleRatios)

    #distances, sampleRatios = draw_ss(clientSampleList, allSamples, totalClients)
    return dis, sampleRs
    #plot_line([distances], [sampleRatios], [''], "Ratio of Clients Sampled", "Divergence to Overall", "diffallEmail.pdf")

def readFromSerialized(file, threshold=99999999):
    infile = open(file, "rb")
    dist = pickle.load(infile)

    threshold = min(threshold, len(dist))
    overall = [0 for x in range(len(dist[0]))]

    dist = sorted(dist, key=lambda k: sum(k), reverse=True)

    for i in range(len(dist)):
        if sum(dist[i]) > 0:
            for index, c in enumerate(dist[i]):
                overall[index] += c
        else:
            threshold = i
            break

    return dist[:threshold], overall, threshold

def draw_ss(clientSampleList, allSamples, totalClients, figCaption="diffall.pdf"):
    # normalize lists
    distances = []
    nallSamples = normalizeList(allSamples)

    for clientSample in clientSampleList:
        #random.shuffle(clientSampleList)
        nclientSamples = normalizeList(clientSample)
        distances.append(measureDistance(nclientSamples, nallSamples))

    return distances, None#, array(numOfSamples)/float(totalClients)


def draw_rs(clientSampleList, allSamples, totalClients, figCaption="diffall.pdf"):

    random.shuffle(clientSampleList)

    numOfSamples = [0.01] + [i * 0.02 for i in range(1, 7)] + [0.15, 0.2, 0.25, 0.3, 0.35, 0.4]

    #numOfSamples = [i for i in range(int(totalClients*0.01), int(totalClients * 0.4), int(totalClients*0.02))]

    # normalize lists
    distances = []
    nallSamples = normalizeList(allSamples)

    for numOfSample in numOfSamples:
        random.shuffle(clientSampleList)
        nsubsetSamples = normalizeList(pickSubset(clientSampleList, int(math.ceil(numOfSample * totalClients))))
        distances.append(measureDistance(nsubsetSamples, nallSamples))

    return distances, numOfSamples #array(numOfSamples)/float(totalClients)


ds = []
ss = []

# for i in range(numOfTries):

#     dE, sE = email()
#     print('...current email trial ' + str(i))
#     dB, sB = blog()
#     print('...current blog trial ' + str(i))
#     dO, sO = openImg()
#     print('...current openImg trial ' + str(i))
#     dQ, sQ = quickDraw()
#     print('...current quickDraw trial ' + str(i))

dE, sE = email()
dB, sB = blog()
dO, sO = openImg()
dQ, sQ = quickDraw()

ds.append([dE, dB, dO, dQ])
ss.append([sE, sB, sO, sQ])

with open("MultiRs", 'wb') as fout:
    pickle.dump(ds, fout)
    pickle.dump(ss, fout)


#plot_line([dE, dB, dO, dQ], [sE, sB, sO, sQ], ['email', 'blog', 'openImg', 'quickDraw'], "Ratio of Clients Sampled", "Divergence to Overall", "diffall.pdf")


# dE = email()
# dB = blog()
# dO = openImg()
# dQ = quickDraw()
# plot_cdf([dE, dB, dO, dQ], ['email', 'blog', 'openImg', 'quickDraw'], "Data Divergence to Overall", "CDF", "cdiffall.pdf")
