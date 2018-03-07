from random import random
import numpy as np
import math

def update(location,velocity, localOptimum,swarmOptimum):
    w = 2 #Inertia wieght
    c1 = 1 #Acceleration constant 1
    c2 = 1 #Acceleration constant 2
    r1 = random()
    r2 = random()
    result = w * velocity + r1 * c1 * (localOptimum-tuple(location)) + r2 * c2 * (swarmOptimum-tuple(location))
    return result.tolist()

def initialize(particles,datapoints,ndim):
    averages = []
    for i in range(0,ndim):
        mean = sum([x[i] for x in datapoints]) / float(len([x[i] for x in datapoints]))
        std = sum([abs(mean - x[i]) for x in datapoints]) / float(len([x[i] for x in datapoints]))
        averages.append((mean,std))
    for p in particles:
        for centroid in p:
            for i in range(0,ndim):
                avg = averages[i][0]
                std = averages[i][1]
                centroid[0][i] = avg - 2 * std + random() * 4 * std #Initialize location as random number within mean-2std to mean+2std
                centroid[1][i] = -1 + random() * 2 #Initialize speed as a random float between -1 and 1
    return particles

def getDistances(datapoints,particle):
    dists = [] #List of lists for the distances of all data points to each centroids
    for d in datapoints:
        centroid_dists = [] #Distances of all centroids to this specific datapoint
        for i in range(0,len(particle)):
            centroid_dists.append(np.linalg.norm(np.array(d)-np.array(particle[i][0])))
        dists.append(centroid_dists)
    return dists

def getFitness(particle,assignedcentroids):
    return None

def cluster(datapoints,classes,n_particles):
    ndim = len(datapoints[0])
    n_clusters = len(set(classes))

    #Initialize the culster locations and velocities
    #First tuple represents the location of the particle, the second tuple the velocity in each dimension
    particles = [[[[0 for d in range(0,ndim)],[0 for d in range(0,ndim)]]
                  for c in range(0,n_clusters)] for p in range(0,n_particles)]
    particles = initialize(particles,datapoints,ndim)

    #Iterate until the
    done = False
    while(done != True):
        for p in particles:
            #Calculate distances between cluster centroids and all datapoints
            distances = getDistances(datapoints,p)
            #Assign each datapoint the closest centroid by adding its index
            assignedcentroids = []
            for ds in distances:
                assignedcentroids.append(ds.index(min(ds)))
            fitness = getFitness(p,assignedcentroids)

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
    cluster(datapoints,classes,3)
    #cluster(data)

if(__name__ == '__main__'):
    main()