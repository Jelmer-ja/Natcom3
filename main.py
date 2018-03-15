from random import random
from scipy.spatial import distance
import numpy as np
import matplotlib.pyplot as plt
import math

def plus(t1,t2):
    return tuple([t1[i] + t2[i] for i in range(0,len(t1))])

def minus(t1,t2):
    return tuple([t1[i] - t2[i] for i in range(0,len(t1))])

def scalar_mult(s,t1):
    return tuple([s * t1[i] for i in range(0,len(t1))])

def update(location,velocity, localOptimum,swarmOptimum):
    w = 0.8 #Inertia weight
    c1 = 2 #Acceleration constant 1
    c2 = 2 #Acceleration constant 2
    r1 = random()
    r2 = random()
    result = plus(plus(scalar_mult(w,velocity), scalar_mult(r1 * c1,minus(tuple(localOptimum),tuple(location)))),scalar_mult(r2 * c2,minus(tuple(swarmOptimum),tuple(location))))
    return [list(plus(tuple(location),result)),list(result)]

def printCentroids(particles):
    output = ''
    for i in range(0,len(particles)):
        output += 'Particle ' + str(i) + ':\n'
        for j in range(0,len(particles[0])):
            output += 'Cluster ' + str(j) + ':' + str(particles[i][j]) + '\n'
    print(output)

#Initialize the centroids in random positions between two standard deviations from the mean for each dimension
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

#Calculate the global fitness for a particle as defined by equation 8 in the paper.
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

def globalFitness(p): #Global fitness implemented as the average distance between centroids
    distances = [distance.euclidean(tuple(x[0]),tuple(y[0])) for x in p for y in p if x != y]
    return -sum(distances) / len(distances)

#Calculate the fitness for individual centroids
def cluster(datapoints,classes,n_particles,n_iterations):
    ndim = len(datapoints[0])
    n_clusters = len(set(classes))

    #Initialize the culster locations and velocities
    #First tuple represents the location of the particle, the second tuple the velocity in each dimension
    particles = [[[[0 for d in range(0,ndim)],[0 for d in range(0,ndim)]]
                  for c in range(0,n_clusters)] for p in range(0,n_particles)]
    particles = initialize(particles,datapoints,ndim)

    #Iterate until until n_interations has been reached
    #Define local and global bests
    globalBest = [[0.0 for x in range(0,ndim)] for i in range(0,n_particles)]
    globalBestFitness = [999999.9 for i in range(0,n_particles)]
    localBest = [[[0.0 for x in range(0,ndim)] for y in range(0,n_clusters)] for i in range(0,n_particles)]
    localBestFitness = [[999999.9 for x in range(0,n_clusters)] for i in range(0,n_particles)]
    fitnesses = [[],[],[]]
    for k in range(0,n_iterations):
        for p in particles:
            #Calculate distances between cluster centroids and all datapoints
            distances = getDistances(datapoints,p)
            #Assign each datapoint the closest centroid by adding its index
            assignedcentroids = []
            for ds in distances:
                assignedcentroids.append(ds.index(min(ds)))

            #Calculate fitnesses and update local and global best
            p_index = particles.index(p)
            fitness = getFitness(p,datapoints,assignedcentroids) #Overall fitness of the particle
            gfitness = globalFitness(p) #Average distance between clusters
            fitnesses[p_index].append(fitness)
            if (gfitness < globalBestFitness[p_index]):
                dummy = (sum([x[0][i] for x in p]) for i in range(0,ndim))
                globalBest[p_index] = tuple([x / n_clusters for x in dummy])
                globalBestFitness[p_index] = gfitness
            cfitness = centroidFitness(p,datapoints,assignedcentroids) #Fitness for individual centroids
            for i in range(0,n_clusters):
                if(cfitness[i] < localBestFitness[p_index][i]):
                    localBest[p_index][i] = p[i][0]
            localBestFitness[p_index] = centroidFitness([[localBest[p_index][i],0] for i in range(0,n_clusters)],datapoints,assignedcentroids)

            #Update centroid locations and velocities
            for i in range(0,n_clusters):
                p[i] = update(p[i][0],p[i][1],localBest[p_index][i],globalBest[p_index])

    #Print the location of the centroids and plot the fitness
    printCentroids(particles)
    plt.plot(range(0,n_iterations),fitnesses[0])
    plt.plot(range(0,n_iterations),fitnesses[1])
    plt.plot(range(0,n_iterations),fitnesses[2])
    plt.title('Fitness over time (Artificial Dataset 1)')
    plt.xlabel('Iterations')
    plt.ylabel('Fitness')
    plt.show()

def import_data(d): #Get Iris dataset
    f = open(d,'r')
    classes = []
    datapoints = []
    for line in f.readlines():
        split = line.split(',')
        classes.append(split[4])
        datapoints.append(tuple([float(x) for x in split[:4]]))
    f.close()
    return datapoints,classes

def import_data_a1(): #Get Artificial dataset 1 as defined in the paper
    datapoints = [[random() * 2 -1,random() * 2 -1] for i in range(0,400)]
    classes = [1 if ((d[0] >= 0.7 or d[0] <= 0.3) and d[1] > -0.2 * d[0]) else 0 for d in datapoints]
    return datapoints, classes

def main():
    datapoints,classes = import_data('irisdata/iris.data')
    #datapoints,classes = import_data_a1()
    cluster(datapoints,classes,3,500)

if(__name__ == '__main__'):
    main()