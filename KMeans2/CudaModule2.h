#ifndef CUDAMODULE2
#define CUDAMODULE2

#include "device_launch_parameters.h"
#include "vector_functions.hpp"
#include <cuda.h>
#include "cuda_runtime.h"
#include <curand_kernel.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/constant_iterator.h>


class CudaModule2
{
private:

	int N; // number of objects
	int n; // number of coordinates
	int k; // number of centroids

	const float threshold = 0.0001;

	int centroid_blocks;
	int centroid_threads;

	int objects_blocks;
	int objects_threads;

	float* h_objects;
	float* d_objects; // object cooridiantes: x0, y0, z0, x1, y1, z1, ...

	int* d_objectsToCentroids; // number of centroid that respons to object with ID in d_objectIDs array

	int* h_objectsToCentroids; // number of centroid that respons to object with ID in d_objectIDs array

	float* h_sums;
	float* h_counts;
	float* h_avgs;

	float* d_centroids; // centroid cooridnates  x0, y0, z0, x1, y1, z1, ... (index of array is an id of a centroid)

	int* d_ifObjectChangedCentroid; // if centroid of object (index is a number ob object) changed in subsequent iteration

	
public:
	CudaModule2(float* d_objects, int N, int n, int k);

	void KMeans();
};

#endif