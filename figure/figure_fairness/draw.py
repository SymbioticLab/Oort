import matplotlib.pyplot as plt 
import numpy as np
from scipy import stats

def read_file(filename):
    with open(filename, 'r') as f:
        data = f.readlines()
    
    accuracy = []
    size = []
    for d in data:
        temp = d.replace(',','').split()
        size.append(int(temp[temp.index('len:')+1]))
        accuracy.append(float(temp[-1]) * 100)
    return accuracy, size

def draw(names, accuracy_list):
    fig, ax = plt.subplots() 
    boxes = []
    for name, acc in zip(names, accuracy_list):

        boxes.append({
            'label' : name,
            'whislo': np.percentile(acc, 5),    # Bottom whisker position
            'q1'    : np.percentile(acc, 25),    # First quartile (25th percentile)
            'med'   : np.median(acc),    # Median         (50th percentile)
            'q3'    : np.percentile(acc, 75),    # Third quartile (75th percentile)
            'whishi': np.percentile(acc, 95),    # Top whisker position
            'fliers': []        # Outliers
        })
        print(name)
        print(np.percentile(acc, 95) - np.percentile(acc, 5))

    ax.bxp(boxes, showfliers=False)
    ax.set_ylabel("Accuracy")
    plt.savefig("boxplot.png")
    plt.close()


if __name__ == "__main__":
    files = ['prox', 'prox', 'yogi', 'kuiper_yogi']
    names = ['Prox', 'Kuiper + Prox', 'YoGi', 'Kuiper + YoGi']

    accuracy_list = []
    size_list = []
    for f in files:
        result = read_file(f)
        accuracy_list.append(result[0])
        size_list.append(result[1])
    accuracy_list = np.array(accuracy_list)
    size_list = np.array(size_list)

    # for s, a in zip(size_list, accuracy_list):
    #     mean = np.sum(s * a)/ np.sum(s)
    #     print(mean)
    #     print(np.mean(a))
    #     var = np.mean(abs(a - mean)**2)
    #     print(var)
    draw(names, accuracy_list)