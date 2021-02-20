#ifndef CUDAMODULE
#define CUDAMODULE

#include "device_launch_parameters.h"
#include "vector_functions.hpp"
#include <cuda.h>
#include "cuda_runtime.h"
#include <curand_kernel.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/constant_iterator.h>

struct compareInt2
{
	__host__ __device__ bool operator() (const int2& a, const int2& b)
	{
		return a.y < b.y;
	}
};

struct equalInt2
{
	__host__ __device__ bool operator() (const int2& a, const int2& b)
	{
		return a.x == b.x;
	}
};

class CudaModule
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

	int* d_objectsIDs; // id of each object
	int* d_objectsToCentroids; // number of centroid that respons to object with ID in d_objectIDs array

	int* d_centroidStartInd; // index  of centroid start in d_objectsToCentroids array (after sorting)
	int* d_centroidEndInd; // index  of centroid end in d_objectsToCentroids array (after sorting)

	float* d_centroids; // centroid cooridnates  x0, y0, z0, x1, y1, z1, ... (index of array is an id of a centroid)

	int* d_ifObjectChangedCentroid; // if centroid of object (index is a number ob object) changed in subsequent iteration

	int2* d_toReduceKeys; // keys to make reduction by key (to calculate sums of each cooridate in each centrois) - key - the number - means the number of coordinate
	float* d_toReduceVals; // values to make reduction by key (to calculate sums of each cooridate in each centrois) - 

	int2* d_reducedKeys; // keys after reduction: x_c0, y_c0 z_c0, x_c1, y_c1, z_c1,  ... where x_c0 means sum od x coordinates in 0th centroid
	float* d_reducedVals; // values after reduction: x_c0, y_c0 z_c0, x_c1, y_c1, z_c1,  ... where x_c0 means the value of sum od x coordinates in 0th centroid

public:
	CudaModule(float* d_objects, int N, int n, int k);

	void KMeans();
};

#endif