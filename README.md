# Parallel K-Means Clustering
Parallel version of K-means clustering (C++, Nvidia CUDA).

## Algorithm
The aim of this project was to compare diferent approaches of K-Means clustering parallelization. 

K-Means Clustering has two main stages:
1. Assigning new centroids (clusters' representants) to objects.
2. Calculating new centroids. 

There are four different alogrithms implemented:
* first - completely sequential,
* second - stage 1: parallel, stage 2: sequential,
* third and fourth - both stages paralleled, but the algorithms differ. 

You can find a comprehensive descripton of all algorithms in my [report](KMeans_description.pdf) (in Polish). 

The results (time comparisons and graphs) can be seen in [this file](KMeans_results_comparison.xlsx). 

## Input and output data
This program assumes the input data is in *data.txt* file in solution directory.

Input data format:
```
 number of object
 number of objects' cooridantes
 number of clusters
 objects (one object per line - coordinates separated by a space)
```

You can generate random input data using IO Generator.

Output data is saved in .txt files in solution directory. Each centroid per line, coordinates separeted by a space.

## Running different algorithms
To run a choosen algorithm you need to uncomment an appropriate line in main function ([main.cpp](KMeans2/main.cpp) file). The algorithms are implemented in modules named as follow:
* NoCudaModule - sequential version,
* CudaModule2 -  stage 1: parallel, stage 2: sequential,
* CudaModule & CudaModule3 - both stages paralleled, but the algorithms differ (read about the differences [here](KMeans_description.pdf)).

## Test data
In folders *n_centroids_data*, *N_objects_data* and *test_data* you can find some random test input data. First two folders contains data files with dfferernt program parameters (number of coordinates or objects) and the last folder - data allowing to easily check how the algorithm works. 
