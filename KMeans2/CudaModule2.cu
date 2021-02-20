#include "CudaModule2.h"
#include "IOManager.h"
#include <stdio.h>
//#define TM
//#define GTM
#define LTM
using namespace std;
// error handling
#define CC(x) do {\
		cudaError_t err = x;\
		if (err != cudaSuccess) {\
			const char* errname = cudaGetErrorName(err);\
			const char* errdesc = cudaGetErrorString(err);\
			printf("ERROR, file: %s, line: %d: Cuda call failed: %s: %s \n", __FILE__, __LINE__, errname, errdesc);\
			exit(-1);\
		}\
	}\
	while(0);\


// helper functions for debug purposes only

__host__ __device__ void PrintArray_2(float* arr, int size, int n)
{
	for (int i = 0; i < size; i++)
	{
		printf("%d. ", i);
		for (int j = 0; j < n; j++)
		{
			printf("%f ", arr[n*i + j]);
		}
		printf("\n");
	}
}

__host__ __device__ void PrintArrayInt_2(int* arr, int size)
{
	for (int i = 0; i < size; i++)
	{
		printf("%d. %d\n", i, arr[i]);
	}
}

__host__ __device__ void PrintArrayInt2_2(int2* arr, int size)
{
	for (int i = 0; i < size; i++)
	{
		printf("%d. %d %d \n", i, arr[i].x, arr[i].y);
	}
}

__device__ float distSquare_2(float* a, float* b, int n)
{
	float ret = 0.0;
	for (int i = 0; i < n; i++)
	{
		ret += (a[i] - b[i])*(a[i] - b[i]);
	}
	return ret;
}

__global__ void print_kernel_2(float* arr, int size, int n)
{
	PrintArray_2(arr, size, n);
}

__global__ void print_int2_kernel_2(int2* arr, int size)
{
	PrintArrayInt2_2(arr, size);
}

__global__ void print_int_kernel_2(int* arr, int size)
{
	PrintArrayInt_2(arr, size);
}

// cuda kernels

__global__ void init_objectsIDs_kernel_2(int* d_objectIDs, int N)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= N)
		return;

	d_objectIDs[i] = i;
}

__global__ void set_random_centroids_kernel_2(float* d_centroids, int k, int n)
{
	curandState state;
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= k)
		return;

	curand_init(1234, i, 0, &state);

	for (int j = 0; j < n; j++)
	{
		d_centroids[i*n + j] = curand_uniform(&state) * 10;
	}
}

__global__ void init_obectsIDs_kernel_2(int* d_objectsIDs, int N)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= N)
		return;

	d_objectsIDs[i] = i;
}

__global__ void calc_nearest_centroids_kernel_2(float* d_objects, int* d_objectsToCentroids, int* d_ifObjectChangedCentroid, int N, float* d_centroids, int k, int n)
{
	// object id (no sorting - no need to store objectIDs array)
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= N)
		return;

	

	int minIdx = 0;
	int minDistSquare = distSquare_2(d_objects + i * n, d_centroids, n);
	for (int j = 1; j < k; j++)
	{
		float d = distSquare_2(d_objects + i * n, d_centroids + j * n, n);
		if (d < minDistSquare)
		{
			minDistSquare = d;
			minIdx = j;
		}
	}

	d_ifObjectChangedCentroid[i] = 0;
	if (d_objectsToCentroids[i] != minIdx)
	{
		d_ifObjectChangedCentroid[i] = 1;
		d_objectsToCentroids[i] = minIdx;
	}
}

__global__ void calcul_starts_ends_kernel_2(int* d_objectsToCentroids, int N, int* d_centroidStartInd, int* d_centroidEndInd, int k)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= N)
		return;

	if (i == 0 || d_objectsToCentroids[i - 1] != d_objectsToCentroids[i])
	{
		d_centroidStartInd[d_objectsToCentroids[i]] = i;
	}
	if (i == N - 1 || d_objectsToCentroids[i + 1] != d_objectsToCentroids[i])
	{
		d_centroidEndInd[d_objectsToCentroids[i]] = i;
	}
}

__global__ void calcul_toReduce_kernel_2(int* d_objectsIDs, float* d_objects, int* d_objectsToCentroids, int N, int* d_centroidStartInd, int* d_centroidEndInd, int2* d_toReduceKeys, float* d_toReduceVals, int k, int n)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x; // index form d_objectsIDs, d_objectToCentroids
	if (i >= N)
		return;

	int centroidID = d_objectsToCentroids[i];
	int startIdx = d_centroidStartInd[centroidID]; // index where centroid d_objectsToCentroids[i] starts in d_objectsToCentroids arrray
	int endIdx = d_centroidEndInd[centroidID]; // index where centroid d_objectsToCentroids[i] ends in d_objectsToCentroids arrray

	int toWriteIndex =
		n * startIdx // elements from previous centroids
		+ (i - startIdx); // elements from current centroid (only 0th coordinates)

	int c = endIdx - startIdx + 1; // number of elements in given centroid 

	for (int j = 0; j < n; j++)
	{
		d_toReduceKeys[toWriteIndex + c * j].x = j;
		d_toReduceKeys[toWriteIndex + c * j].y = centroidID + 1; // NOTE: ID of centroid is increased by one, in order to distinguish from the case when there's no objects with the centorid (0 then)
		d_toReduceVals[toWriteIndex + c * j] = d_objects[d_objectsIDs[i] * n + j];
	}
}


__global__ void update_centroids_kernel_2(float* d_reducedVals, int2* d_reducedKeys, int* d_centroidStartInd, int* d_centroidEndInd, float* d_centroids, int k, int n)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x; // index of d_reducedKeys array
	if (idx >= k)
		return;

	int i = d_reducedKeys[idx*n].y;
	if (i == 0) // no objects in centroid - no need to update
		return;

	i--;

	int count = d_centroidEndInd[i] - d_centroidStartInd[i] + 1;

	for (int j = 0; j < n; j++)
	{
		d_centroids[i*n + j] = d_reducedVals[idx*n + j] / (float)count;
	}

}


CudaModule2::CudaModule2(float* h_objects, int N, int n, int k)
{
	this->N = N;
	this->n = n;
	this->k = k;

	this->h_objects = h_objects;

	centroid_blocks = k % 512 != 0 ? k / 512 + 1 : k / 512;
	centroid_threads = k < 512 ? k : 512;

	objects_blocks = N % 512 != 0 ? N / 512 + 1 : N / 512;
	objects_threads = N < 512 ? N : 512;

	CC(cudaMalloc((void**)&d_objects, N * n * sizeof(float)));

	CC(cudaMalloc((void**)&d_objectsToCentroids, N * sizeof(int)));

	h_objectsToCentroids = new int[N];

	CC(cudaMalloc((void**)&d_centroids, k * n * sizeof(float)));

	CC(cudaMalloc((void**)&d_ifObjectChangedCentroid, N * sizeof(int)));

	h_sums = new float[n*k];
	for (int i = 0; i < n*k; i++)
		h_sums[i] = 0.0;
	h_counts = new float[k];
	for (int i = 0; i < k; i++)
		h_counts[i] = 0;
	h_avgs = new float[n*k];

	CC(cudaMemcpy(d_objects, h_objects, N * n * sizeof(float), cudaMemcpyHostToDevice));
}

void CudaModule2::KMeans()
{
#ifdef TM
	cudaEvent_t start, stop;
	CC(cudaEventCreate(&start));
	CC(cudaEventCreate(&stop));
	float milliseconds = 0;
#endif

#ifdef GTM
	cudaEvent_t g_start, g_stop;
	CC(cudaEventCreate(&g_start));
	CC(cudaEventCreate(&g_stop));
	float g_milliseconds = 0;
	CC(cudaEventRecord(g_start));
#endif

#ifdef LTM
	cudaEvent_t l_start, l_stop;
	CC(cudaEventCreate(&l_start));
	CC(cudaEventCreate(&l_stop));
	float l_milliseconds = 0;
#endif

#ifdef TM
	CC(cudaEventRecord(start));
#endif
	set_random_centroids_kernel_2 << <centroid_blocks, centroid_threads >> > (d_centroids, k, n);
	CC(cudaMemcpy(h_avgs, d_centroids, k*n * sizeof(float), cudaMemcpyDeviceToHost));
#ifdef TM
	CC(cudaEventRecord(stop));
	CC(cudaEventSynchronize(stop));
	milliseconds = 0;
	CC(cudaEventElapsedTime(&milliseconds, start, stop));
	std::cout << "set_random_centroids_kernel: " << milliseconds << " ms" << endl;
#endif

	int changedCount = N;
	int iter = 0;
	while ((float)changedCount / (float)N > threshold && iter < 600)
	{
#ifdef LTM
		CC(cudaEventRecord(l_start));
#endif
		iter++;
#ifdef TM
		CC(cudaEventRecord(start));
#endif
		calc_nearest_centroids_kernel_2 << <objects_blocks, objects_threads >> > (d_objects, d_objectsToCentroids, d_ifObjectChangedCentroid, N, d_centroids, k, n);
#ifdef TM
		CC(cudaEventRecord(stop));
		CC(cudaEventSynchronize(stop));
		milliseconds = 0;
		CC(cudaEventElapsedTime(&milliseconds, start, stop));
		std::cout << "calc_nearest_centroids_kernel: " << milliseconds << " ms" << endl;
#endif

#ifdef TM
		CC(cudaEventRecord(start));
#endif
		thrust::device_ptr<int> d_ptr_ifObjectChangedCentroid(d_ifObjectChangedCentroid);
		changedCount = thrust::reduce(thrust::device, d_ptr_ifObjectChangedCentroid, d_ptr_ifObjectChangedCentroid + N, 0);
#ifdef TM
		CC(cudaEventRecord(stop));
		CC(cudaEventSynchronize(stop));
		milliseconds = 0;
		CC(cudaEventElapsedTime(&milliseconds, start, stop));
		std::cout << "reduce to find number of changed: " << milliseconds << " ms" << endl;
#endif

#ifdef TM
		CC(cudaEventRecord(start));
#endif
		CC(cudaMemcpy(h_objectsToCentroids, d_objectsToCentroids, N * sizeof(int), cudaMemcpyDeviceToHost));

		for (int i = 0; i < N; i++)
		{
			int centroidIdx = h_objectsToCentroids[i];
			h_counts[centroidIdx]++;
			for (int j = 0; j < n; j++)
			{
				h_sums[centroidIdx*n + j] += h_objects[i*n + j];
			}
		}

		for (int i = 0; i < k; i++)
		{
			if (h_counts[i] > 0)
			{
				for (int j = 0; j < n; j++)
				{
					h_avgs[i*n + j] = h_sums[i*n + j] / h_counts[i];
					h_sums[i*n + j] = 0.0;
				}
				h_counts[i] = 0;
			}
		}

		CC(cudaMemcpy(d_centroids, h_avgs, k*n * sizeof(float), cudaMemcpyHostToDevice));

#ifdef TM
		CC(cudaEventRecord(stop));
		CC(cudaEventSynchronize(stop));
		milliseconds = 0;
		CC(cudaEventElapsedTime(&milliseconds, start, stop));
		std::cout << "update_centroids sequential: " << milliseconds << " ms" << endl;
#endif	

#ifdef LTM
		CC(cudaEventRecord(l_stop));
		CC(cudaEventSynchronize(l_stop));
		l_milliseconds = 0;
		CC(cudaEventElapsedTime(&l_milliseconds, l_start, l_stop));
		std::cout << "1loop time: " << l_milliseconds << " ms" << endl;
#endif
	}

#ifdef GTM
	CC(cudaEventRecord(g_stop));
	CC(cudaEventSynchronize(g_stop));
	g_milliseconds = 0;
	CC(cudaEventElapsedTime(&g_milliseconds, g_start, g_stop));
	std::cout << "1. sum time: " << g_milliseconds << " ms" << endl;
	std::cout << "2. iter: " << iter << " ms" << endl;
#endif
	float* finalCentroids = new float[n*k];
	CC(cudaMemcpy(finalCentroids, d_centroids, k*n * sizeof(float), cudaMemcpyDeviceToHost));

	IOManager::WriteResults("results_CudaModule2.txt", n, k, finalCentroids);
}



