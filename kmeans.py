import scipy.cluster.vq as vq
from random import random

#This class implements k-means clustering
def import_data(d):
    f = open(d,'r')
    classes = []
    datapoints = []
    for line in f.readlines():
        split = line.split(',')
        classes.append(split[4])
        datapoints.append(tuple([float(x) for x in split[:4]]))
    f.close()
    return datapoints,classes

def import_data_a1(): #Get Artificial dataset 1
    datapoints = [[random() * 2 -1,random() * 2 -1] for i in range(0,400)]
    classes = [1 if ((d[0] >= 0.7 or d[0] <= 0.3) and d[1] > -0.2 * d[0]) else 0 for d in datapoints]
    return datapoints, classes

def main():
    datapoints,classes = import_data_a1()#('irisdata/iris.data')
    clustering = vq.kmeans(datapoints,4)
    print(clustering)

if(__name__ == '__main__'):
    main()
