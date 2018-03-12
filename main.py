from random import random
from scipy.spatial import distance
import numpy as np
import math

def plus(t1,t2):
    return tuple([t1[i] + t2[i] for i in range(0,len(t1))])

def minus(t1,t2):
    return tuple([t1[i] - t2[i] for i in range(0,len(t1))])

def scalar_mult(s,t1):
    return tuple([s * t1[i] for i in range(0,len(t1))])

def update(location,velocity, localOptimum,swarmOptimum):
    w = 2 #Inertia wieght
    c1 = 1 #Acceleration constant 1
    c2 = 1 #Acceleration constant 2
    r1 = random()
    r2 = random()
    result = plus(plus(scalar_mult(w,velocity), scalar_mult(r1 * c1,minus(tuple(localOptimum),tuple(location)))),scalar_mult(r2 * c2,minus(tuple(swarmOptimum),tuple(location))))
    print(result)
    print(location)
    return [list(plus(tuple(location),result)),list(result)]

def printCentroids(particles):
    output = ''
    for i in range(0,len(particles)):
        output += 'Particle ' + str(i) + ':\n'
        for j in range(0,len(particles[0])):
            output += 'Cluster ' + str(j) + ':' + str(particles[i][j]) + '\n'
    print(output)

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

def getFitness(particle,datapoints,assignedcentroids):
    fitness = 0
    for i in range(0,len(particle)):
        centroid = particle[i]
        centroid_max = 0
        cluster_indices = [x for x in range(0,len(assignedcentroids)) if assignedcentroids[x] == i]
        for j in cluster_indices:
            centroid_max += distance.euclidean(tuple(centroid[0]),tuple(datapoints[j]))
        fitness += centroid_max / (len(cluster_indices) + 0.01) #Smoothed by 0.01
    return fitness / len(particle)

def centroidFitness(particle,datapoints,assignedcentroids):
    output = []
    for i in range(0,len(particle)):
        centroid = particle[i]
        centroid_max = 0
        cluster_indices = [x for x in range(0,len(assignedcentroids)) if assignedcentroids[x] == i]
        for j in cluster_indices:
            centroid_max += distance.euclidean(tuple(centroid[0]),tuple(datapoints[j]))
        output.append(centroid_max / (len(cluster_indices) + 0.01)) #Smoothed by 0.01
    return output

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
            #Define local and global best
            globalBest = [0.0 for x in range(0,ndim)]
            globalBestFitness = 999999.9
            localBest = [[0.0 for x in range(0,ndim)] for y in range(0,n_clusters)]
            localBestFitness = [999999.9 for x in range(0,n_clusters)]

            #Calculate distances between cluster centroids and all datapoints
            distances = getDistances(datapoints,p)
            #Assign each datapoint the closest centroid by adding its index
            assignedcentroids = []
            for ds in distances:
                assignedcentroids.append(ds.index(min(ds)))

            #Calculate fitnesses and update local and global best
            fitness = getFitness(p,datapoints,assignedcentroids)
            if (fitness < globalBestFitness):
                dummy = (sum([x[0][0] for x in p]),sum([x[0][1] for x in p]),sum([x[0][2] for x in p]),sum([x[0][3] for x in p]))
                globalBest = tuple([x / n_clusters for x in dummy])
                globalBestFitness = fitness
            cfitness = centroidFitness(p,datapoints,assignedcentroids)
            for i in range(0,n_clusters):
                if(cfitness[i] < localBestFitness[i]):
                    localBest[i] = p[i][0]
            localBestFitness = centroidFitness([[localBest[i],0] for i in range(0,n_clusters)],datapoints,assignedcentroids)

            #Update centroid locations and velocities
            for i in range(0,n_clusters):
                p[i] = update(p[i][0],p[i][1],localBest[i],globalBest)

        #Print the location of the centroids
        printCentroids(particles)

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