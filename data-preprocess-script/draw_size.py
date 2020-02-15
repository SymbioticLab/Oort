import matplotlib.pyplot as plt
import numpy as np
from numpy import *
import re, math
#import seaborn as sns
from matplotlib import rcParams
import pandas as pd
import matplotlib, csv
from matplotlib import rc
#from pyemd import emd
import dit
#from dit.divergences import jensen_shannon_divergence
import pickle
#from data_div import measureAllDistance

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
    _fontsize = 10
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

divBigger1 = read_div("blog_top1000_data_size.txt")
divBigger2 = read_div("email_data_size.txt")
divBiggerOpenImg = read_from_sum("openImg_size.txt", 507)
divBiggerQuickDraw = read_from_sum("quickdraw_size.txt")
   


plot_cdf([divBigger1, divBigger2, divBiggerOpenImg, divBiggerQuickDraw], ['email', 'blog', 'openImg', 'quickDraw'], "Data Size across Clients", "CDF", "cSample.pdf")


# sumOfSamples = [407151, 377247, 353653, 311266, 260730, 259152, 258209, 253638, 247461, 246345, 245879, 244401, 236422, 235054, 217480, 211138, 209066, 208276, 206766, 203002, 202481, 198784, 195878, 195048, 183976, 181233, 180153, 179526, 178319, 169386, 165010, 163446, 162984, 162706, 159890, 157723, 148684, 147361, 147361, 147080, 146925, 146901, 146814, 146296, 145306, 143345, 143299, 140875, 140871, 139694, 134818, 134793, 134438, 134231, 133890, 130389, 129520, 129165, 127506, 126671, 122998, 122866, 121828, 121508, 121089, 121089, 120636, 120071, 119419, 118880, 118143, 117796, 117456, 117241, 116581, 116501, 115285, 114011, 113888, 113667, 113650, 113380, 113376, 113198, 113125, 112058, 111988, 111685, 111518, 110979, 110642, 110246, 109337, 108992, 106986, 106826, 106215, 104506, 104264, 104261, 103897, 103807, 103257, 103005, 102391, 102379, 100794, 100758, 100516, 99601, 99511, 99168, 98622, 98091, 96844, 96598, 96372, 96098, 95984, 95755, 95737, 95026, 94630, 94346, 92981, 92668, 92321, 92276, 92123, 92113, 91930, 91865, 91688, 91461, 90632, 90581, 89552, 89417, 88623, 88470, 88470, 88405, 88306, 88000, 87963, 87637, 87191, 87024, 86687, 86115, 86031, 85763, 85349, 84970, 84823, 84736, 84505, 83663, 83661, 83466, 83426, 83107, 83051, 82868, 82836, 82688, 82404, 82403, 82338, 81731, 81584, 81412, 80838, 80406, 80269, 79797, 79696, 78911, 78829, 78795, 78507, 78116, 78000, 77531, 77402, 77226, 76890, 76719, 76394, 76085, 75741, 75702, 75542, 75507, 75507, 75242, 75197, 74996, 74913, 74854, 74658, 74584, 74559, 74559, 74521, 74418, 74356, 74287, 74201, 74037, 73784, 73161, 73047, 72953, 72898, 72302, 72090, 72042, 72028, 71820, 71536, 71413, 71223, 71206, 70906, 70881, 70878, 70782, 70324, 70163, 70146, 70056, 70037, 69599, 69394, 69247, 69088, 68961, 68906, 68776, 68457, 68311, 68302, 67967, 67814, 67729, 67583, 67059, 67057, 67010, 66978, 66655, 66636, 65756, 65618, 65618, 64803, 64634, 64497, 64349, 63865, 63663, 63613, 63548, 63484, 63253, 63065, 62960, 62625, 62579, 62351, 62304, 62282, 62195, 61835, 61671, 61604, 61354, 61319, 61230, 61141, 61102, 60903, 60863, 60860, 60767, 60726, 60531, 60413, 60013, 59934, 59851, 59472, 59263, 59118, 59095, 59062, 58878, 58867, 58724, 58108, 57888, 57763, 57387, 57366, 57366, 57366, 57282, 56997, 56997, 56925, 56853, 56794, 56702, 56682, 56636, 56092, 56030, 55713, 55574, 55429, 55122, 54991, 54803, 54767, 54594, 54231, 54188, 54095, 54050, 53976, 53975, 53930, 53774, 53762, 53742, 53684, 53680, 53647, 53437, 53392, 53272, 53197, 53103, 53085, 52917, 52807, 52535, 51999, 51981, 51972, 51250, 51114, 51099, 51063, 50972, 50901, 50868, 50813, 50799, 50532, 50484, 50470, 50440, 50439, 50342, 50339, 50333, 50229, 50209, 50167, 50113, 50008, 49973, 49973, 49934, 49652, 49536, 49464, 49354, 49353, 49217, 48893, 48721, 48676, 48402, 48154, 48128, 47825, 47812, 47741, 47675, 47540, 47388, 47341, 47109, 46928, 46892, 46884, 46686, 46685, 46670, 46663, 46534, 46441, 46360, 46353, 46349, 46316, 46240, 46192, 46185, 46155, 46033, 46013, 45933, 45864, 45781, 45642, 45642, 45633, 45624, 45569, 45509, 45465, 45412, 45371, 45357, 45213, 45106, 45082, 45061, 45057, 44943, 44912, 44832, 44827, 44810, 44765, 44576, 44455, 44388, 44344, 44344, 44323, 44232, 44217, 44194, 44130, 44061, 44022, 43984, 43918, 43892, 43891, 43826, 43824, 43794, 43629, 43313, 43084, 43083, 43069, 42987, 42947, 42940, 42934, 42794, 42779, 42553, 42544, 42418, 42392, 42307, 42285, 42165, 42080, 42000, 41985, 41915, 41836, 41795, 41734, 41706, 41704, 41637, 41621, 41583, 41582, 41559, 41510, 41443, 41372, 41369, 41326, 41286, 41284, 41233, 41196, 41160]

# sumOfSamples = [i/100 for i in sumOfSamples]
# plot_cdf([sumOfSamples], [''], "\# of Samples per Client", "CDF", "sizeImg.pdf")
