from random import random

def update(location,velocity, localOptimum,swarmOptimum):
    w = 2 #Inertia wieght
    c1 = 1 #Acceleration constant 1
    c2 = 1 #Acceleration constant 2
    r1 = random()
    r2 = random()
    return w * velocity + r1 * c1 * (localOptimum-location) + r2 * c2 * (swarmOptimum-location)

def cluster(datapoints,classes):
    ndim = len(datapoints[0])
    particles = [[tuple([random(0,1)*10 for i in range(0,ndim)]),] for i in range(0,len(classes))] #TODO fix maximum and minimum

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

def main():
    datapoints,classes = import_data('irisdata/iris.data')
    print(datapoints[0])
    print(classes[0])
    #cluster(data)

if(__name__ == '__main__'):
    main()