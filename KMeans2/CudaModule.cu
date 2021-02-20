#include "cudaModule.h"
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
__host__ __device__ void PrintArray(float* arr, int size, int n)
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

__host__ __device__ void PrintArrayInt(int* arr, int size)
{
	for (int i = 0; i < size; i++)
	{
		printf("%d. %d\n", i, arr[i]);
	}
}

__host__ __device__ void PrintArrayInt2(int2* arr, int size)
{
	for (int i = 0; i < size; i++)
	{
		printf("%d. %d %d \n", i, arr[i].x, arr[i].y);
	}
}

__device__ float distSquare(float* a, float* b, int n)
{
	float ret = 0.0;
	for (int i = 0; i < n; i++)
	{
		ret += (a[i] - b[i])*(a[i] - b[i]);
	}
	return ret;
}

__global__ void print_kernel(float* arr, int size, int n)
{
	PrintArray(arr, size, n);
}

__global__ void print_int2_kernel(int2* arr, int size)
{
	PrintArrayInt2(arr, size);
}

__global__ void print_int_kernel(int* arr, int size)
{
	PrintArrayInt(arr, size);
}

// cuda kernels
__global__ void init_objectsIDs_kernel(int* d_objectIDs, int N)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= N)
		return;

	d_objectIDs[i] = i;
}

__global__ void set_random_centroids_kernel(float* d_centroids, int k, int n)
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

__global__ void init_obectsIDs(int* d_objectsIDs, int N)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= N)
		return;

	d_objectsIDs[i] = i;
}

// NOTE: number of centorids must be <= number of threads, otherwise use calc_nearest_centroids_with
__global__ void calc_nearest_centroids_with_shared_centroids_kernel(float* d_objects, int* d_objectsIDs, int* d_objectsToCentroids, int* d_ifObjectChangedCentroid, int N, float* d_centroids, int k, int n)
{
	// index of objectID in d_objectsIds table
    // NOTE objectId is use do get coordinates from d_objects. d_objectIDs[idx] and d_objectsToCentroids[idx] are connected with the same  object (the object with coordinates starting at d_objectsIDs[idx]*n
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= N)
		return;

	// object id
	int i = d_objectsIDs[idx];

	extern __shared__  char sharedMemory[];
	float *objects_shared = (float *)sharedMemory;

	for (int p = 0; p < n; p++)
	{
		// when more centroids than threads
		if(threadIdx.x<k) objects_shared[blockDim.x*n + threadIdx.x*n + p] = d_centroids[threadIdx.x*n + p]; // each object write one centroid to shared
	}
	__syncthreads();

	for (int p = 0; p < n; p++)
	{
		objects_shared[threadIdx.x*n + p] = d_objects[i*n + p]; //each thread write its object to shared
	}

	int minIdx = 0;
	int minDistSquare = distSquare(objects_shared + threadIdx.x*n, objects_shared + blockDim.x*n, n);
	for (int j = 1; j < k; j++)
	{
		float d = distSquare(objects_shared + threadIdx.x*n, objects_shared + blockDim.x*n + j*n, n);
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

__global__ void calc_nearest_centroids_kernel(float* d_objects, int* d_objectsIDs, int* d_objectsToCentroids, int* d_ifObjectChangedCentroid, int N, float* d_centroids, int k, int n)
{
	// index of objectID in d_objectsIds table
	// NOTE objectId is used do get coordinates from d_objects. d_objectIDs[idx] and d_objectsToCentroids[idx] are connected with the same  object (the object with coordinates starting at d_objectsIDs[idx]*n)
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= N)
		return;

	// object id
	int i = d_objectsIDs[idx];

	extern __shared__  char sharedMemory[];
	float *objects_shared = (float *)sharedMemory;

	for (int p = 0; p < n; p++)
	{
		objects_shared[threadIdx.x*n + p] = d_objects[i*n + p];
	}

	int minIdx = 0;
	int minDistSquare = distSquare(objects_shared + threadIdx.x*n, d_centroids, n);
	for (int j = 1; j < k; j++)
	{
		float d = distSquare(objects_shared + threadIdx.x*n, d_centroids + j * n, n);
		if (d < minDistSquare)
		{
			minDistSquare = d;
			minIdx = j;
		}
	}

	d_ifObjectChangedCentroid[i] = 0;
	if (d_objectsToCentroids[idx] != minIdx)
	{
		d_ifObjectChangedCentroid[i] = 1;
		d_objectsToCentroids[idx] = minIdx;
	}
}


__global__ void calcul_starts_ends_kernel(int* d_objectsToCentroids, int N, int* d_centroidStartInd, int* d_centroidEndInd, int k)
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


__global__ void calcul_toReduce_kernel(int* d_objectsIDs, float* d_objects, int* d_objectsToCentroids, int N, int* d_centroidStartInd, int* d_centroidEndInd, int2* d_toReduceKeys, float* d_toReduceVals, int k, int n)
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


	extern __shared__  char sharedMemory[];
	float *objects_shared = (float *)sharedMemory;
	for (int j = 0; j < n; j++)
	{
		objects_shared[threadIdx.x*n + j] = d_objects[d_objectsIDs[i] * n + j];
	}

	for (int j = 0; j < n; j++)
	{
		d_toReduceKeys[toWriteIndex + c * j].x = j;
		d_toReduceKeys[toWriteIndex + c * j].y = centroidID + 1; // NOTE: ID of centroid is increased by one, in order to distinguish from the case when there's no objects with the centorid (0 then)
		d_toReduceVals[toWriteIndex + c * j] = objects_shared[threadIdx.x*n + j];
	}
}


__global__ void update_centroids_kernel(float* d_reducedVals, int2* d_reducedKeys, int* d_centroidStartInd, int* d_centroidEndInd, float* d_centroids, int k, int n)
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


CudaModule::CudaModule(float* h_objects, int N, int n, int k)
{
	this->N = N;
	this->n = n;
	this->k = k;

	this->h_objects = h_objects;

	centroid_blocks =  k % 512!=0 ? k/512 +1 : k/512;
	centroid_threads = k < 512 ? k : 512;

	objects_blocks = N % 512 != 0 ? N / 512 + 1 : N / 512;
	objects_threads = N < 512 ? N : 512;

	CC(cudaMalloc((void**)&d_objects, N * n * sizeof(float)));

	CC(cudaMalloc((void**)&d_objectsIDs, N * sizeof(int)));
	CC(cudaMalloc((void**)&d_objectsToCentroids, N * sizeof(int)));

	CC(cudaMalloc((void**)&d_centroidStartInd, k * sizeof(int)));
	CC(cudaMalloc((void**)&d_centroidEndInd, k * sizeof(int)));

	CC(cudaMalloc((void**)&d_centroids, k * n * sizeof(float)));

	CC(cudaMalloc((void**)&d_ifObjectChangedCentroid, N * sizeof(int)));

	CC(cudaMalloc((void**)&d_toReduceKeys, N * n * sizeof(int2)));
	CC(cudaMalloc((void**)&d_toReduceVals, N * n * sizeof(float)));

	CC(cudaMalloc((void**)&d_reducedKeys, k * n * sizeof(int2)));
	CC(cudaMalloc((void**)&d_reducedVals, k * n * sizeof(float)));

	CC(cudaMemcpy(d_objects, h_objects, N * n * sizeof(float), cudaMemcpyHostToDevice));
}

void CudaModule::KMeans()
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
	set_random_centroids_kernel <<<centroid_blocks, centroid_threads>>> (d_centroids, k, n);
#ifdef TM
	CC(cudaEventRecord(stop));
	CC(cudaEventSynchronize(stop));
	milliseconds = 0;
	CC(cudaEventElapsedTime(&milliseconds, start, stop));
	std::cout << "1. set_random_centroids: " << milliseconds << " ms" << endl;
#endif

#ifdef TM
	CC(cudaEventRecord(start));
#endif

	init_obectsIDs <<<objects_blocks, objects_threads>>> (d_objectsIDs, N);

#ifdef TM
	CC(cudaEventRecord(stop));
	CC(cudaEventSynchronize(stop));
	milliseconds = 0;
	CC(cudaEventElapsedTime(&milliseconds, start, stop));
	std::cout << "1. init_obectsIDs: " << milliseconds << " ms" << endl;
#endif

	int changedCount = N;
	int iter = 0;
	while ((float)changedCount / (float)N > threshold && iter<600)
	{
#ifdef LTM
		CC(cudaEventRecord(l_start));
#endif
		iter++;
#ifdef TM
		CC(cudaEventRecord(start));
#endif
		if(k >= 20 && k <= objects_threads)
			calc_nearest_centroids_with_shared_centroids_kernel<<<objects_blocks, objects_threads, 2 * objects_threads * 4 * n>>>(d_objects, d_objectsIDs, d_objectsToCentroids, d_ifObjectChangedCentroid, N, d_centroids, k, n);
		else
			calc_nearest_centroids_kernel<<<objects_blocks, objects_threads, objects_threads * 4 * n>>>(d_objects, d_objectsIDs, d_objectsToCentroids, d_ifObjectChangedCentroid, N, d_centroids, k, n);
		
#ifdef TM
		CC(cudaEventRecord(stop));
		CC(cudaEventSynchronize(stop));
		milliseconds = 0;
		CC(cudaEventElapsedTime(&milliseconds, start, stop));
		std::cout << "calc_nearest_centroids: " << milliseconds << " ms" << endl;
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
		thrust::device_ptr<int> d_ptr_objectsIDs(d_objectsIDs);
		thrust::device_ptr<int> d_ptr_objectsToCentroids(d_objectsToCentroids);
		thrust::sort_by_key(thrust::device, d_ptr_objectsToCentroids, d_ptr_objectsToCentroids + N, d_ptr_objectsIDs);
#ifdef TM
		CC(cudaEventRecord(stop));
		CC(cudaEventSynchronize(stop));
		milliseconds = 0;
		CC(cudaEventElapsedTime(&milliseconds, start, stop));
		std::cout << "sort d_objectsIDs by d_objectsToCentroids: " << milliseconds << " ms" << endl;
#endif

#ifdef TM
		CC(cudaEventRecord(start));
#endif
		calcul_starts_ends_kernel <<<objects_blocks, objects_threads >>> (d_objectsToCentroids, N, d_centroidStartInd, d_centroidEndInd, k);
#ifdef TM
		CC(cudaEventRecord(stop));
		CC(cudaEventSynchronize(stop));
		milliseconds = 0;
		CC(cudaEventElapsedTime(&milliseconds, start, stop));
		std::cout << "calcul_starts_ends_kernel: " << milliseconds << " ms" << endl;
#endif

#ifdef TM
		CC(cudaEventRecord(start));
#endif
		calcul_toReduce_kernel <<<objects_blocks, objects_threads ,  objects_threads * 4 * n  >>> (d_objectsIDs, d_objects, d_objectsToCentroids, N, d_centroidStartInd, d_centroidEndInd, d_toReduceKeys, d_toReduceVals, k, n);
#ifdef TM
		CC(cudaEventRecord(stop));
		CC(cudaEventSynchronize(stop));
		milliseconds = 0;
		CC(cudaEventElapsedTime(&milliseconds, start, stop));
		std::cout << "calcul_toReduce_kernel: " << milliseconds << " ms" << endl;
#endif

#ifdef TM
		CC(cudaEventRecord(start));
#endif
		thrust::device_ptr<int2> d_ptr_toReduceKeys(d_toReduceKeys);
		thrust::device_ptr<float> d_ptr_toReduceVals(d_toReduceVals);
		thrust::device_ptr<int2> d_ptr_reducedKeys(d_reducedKeys);
		thrust::device_ptr<float> d_ptr_reducedVals(d_reducedVals);

		thrust::reduce_by_key(thrust::device, d_ptr_toReduceKeys, d_ptr_toReduceKeys + N *n, d_ptr_toReduceVals, d_ptr_reducedKeys, d_ptr_reducedVals, equalInt2());
#ifdef TM
		CC(cudaEventRecord(stop));
		CC(cudaEventSynchronize(stop));
		milliseconds = 0;
		CC(cudaEventElapsedTime(&milliseconds, start, stop));
		std::cout << "reduce_by_key - calculate new centroids: " << milliseconds << " ms" << endl;
#endif		

#ifdef TM
		CC(cudaEventRecord(start));
#endif
		update_centroids_kernel << <centroid_blocks, centroid_threads >> > (d_reducedVals, d_reducedKeys, d_centroidStartInd, d_centroidEndInd, d_centroids, k, n);
#ifdef TM
		CC(cudaEventRecord(stop));
		CC(cudaEventSynchronize(stop));
		milliseconds = 0;
		CC(cudaEventElapsedTime(&milliseconds, start, stop));
		std::cout << "update_centroids_kernel: " << milliseconds << " ms" << endl;
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
	std::cout << "2. iter: " << iter << endl;
#endif
	float* finalCentroids = new float[n*k];
	CC(cudaMemcpy(finalCentroids, d_centroids, k*n * sizeof(float), cudaMemcpyDeviceToHost));

	IOManager::WriteResults("results_CudaModule.txt", n, k, finalCentroids);

}



