import scipy.cluster.vq as vq

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

def main():
    datapoints,classes = import_data('irisdata/iris.data')
    clustering = vq.kmeans(datapoints,4)
    print(clustering)

if(__name__ == '__main__'):
    main()
