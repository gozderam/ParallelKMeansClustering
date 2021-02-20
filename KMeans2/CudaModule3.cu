#include "CudaModule3.h"
#include "IOManager.h"
#include <stdio.h>
//#define TM
//#define GTM
#define LTM

#define tc 512

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
__host__ __device__ void PrintArray_3(float* arr, int size, int n)
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

__host__ __device__ void PrintArrayInt_3(int* arr, int size)
{
	for (int i = 0; i < size; i++)
	{
		printf("%d. %d\n", i, arr[i]);
	}
}

__host__ __device__ void PrintArrayInt2_3(int2* arr, int size)
{
	for (int i = 0; i < size; i++)
	{
		printf("%d. %d %d \n", i, arr[i].x, arr[i].y);
	}
}

__device__ float distSquare_3(float* a, float* b, int n)
{
	float ret = 0.0;
	for (int i = 0; i < n; i++)
	{
		ret += (a[i] - b[i])*(a[i] - b[i]);
	}
	return ret;
}

__global__ void print_kernel_3(float* arr, int size, int n)
{
	PrintArray_3(arr, size, n);
}

__global__ void print_int2_kernel_3(int2* arr, int size)
{
	PrintArrayInt2_3(arr, size);
}

__global__ void print_int_kernel_3(int* arr, int size)
{
	PrintArrayInt_3(arr, size);
}

// cuda kernels
__global__ void set_random_centroids_kernel_3(float* d_centroids, int k, int n)
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


__global__ void calc_nearest_centroids_kernel_3(float* d_objects, int* d_objectsToCentroids, int* d_ifObjectChangedCentroid, int N, float* d_centroids, int k, int n)
{

	// object id (no sorting - no need to store objectIDs array)
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= N)
		return;

	extern __shared__  char sharedMemory[];
	float *objects_shared = (float *)sharedMemory;

	for (int p = 0; p < n; p++)
	{
		objects_shared[threadIdx.x*n + p] = d_objects[i*n + p];
	}

	int minIdx = 0;
	int minDistSquare = distSquare_3(objects_shared + threadIdx.x*n, d_centroids, n);
	for (int j = 1; j < k; j++)
	{
		float d = distSquare_3(objects_shared + threadIdx.x*n, d_centroids + j * n, n);
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

__global__ void calc_sums_counts_kernel_3(float* d_objects, int* d_objectsToCentroids, int N, float* d_toRedSums, int* d_toRedCounts, int k, int n)
{

	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= tc)
		return;

	int part = N / tc;
	for (int ii = idx*part; ii < (idx+1)*part; ii++)
	{
		int centroidIdx = d_objectsToCentroids[ii];
		d_toRedCounts[centroidIdx*tc+idx]++;
		for (int jj = 0; jj < n; jj++)
		{
			d_toRedSums[centroidIdx*tc*n + jj*tc + idx] += d_objects[ii*n + jj];
		}
	}
}

__global__ void reduce_sums_kernel_3(float* d_toRedSums, float* d_reducedSums, int k, int n)
{

	int sumIdx = threadIdx.x + blockIdx.x * blockDim.x;
	if (sumIdx >= k * n* tc)
		return;

	extern __shared__  char sharedMemory[];
	float *redSums_shared = (float *)sharedMemory;

	int innerSumIdx = threadIdx.x;
	redSums_shared[innerSumIdx] = d_toRedSums[sumIdx];
	__syncthreads();

	for (int s = tc / 2; s > 0; s >>= 1)
	{
		if (innerSumIdx < s)
			redSums_shared[innerSumIdx] += redSums_shared[innerSumIdx + s];
		__syncthreads();
	}
	if (innerSumIdx == 0)
	{
		d_reducedSums[blockIdx.x] = redSums_shared[innerSumIdx];
	}
	d_toRedSums[sumIdx] = 0;
}

__global__ void reduce_counts_kernel_3(int* d_toRedCounts, int* d_reducedCounts, int k, int n)
{

	int countIdx = threadIdx.x + blockIdx.x * blockDim.x;
	if (countIdx >= k * tc)
		return;
	
	extern __shared__  char sharedMemory[];
	int *redCounts_shared = (int *)sharedMemory;

	int innerCountIdx = threadIdx.x;
	redCounts_shared[innerCountIdx] = d_toRedCounts[countIdx];
	__syncthreads();

	for (int s = tc / 2; s > 0; s >>= 1)
	{
		if (innerCountIdx < s)
		{
			redCounts_shared[innerCountIdx] += redCounts_shared[innerCountIdx + s];
		}
		__syncthreads();
	}
	if (innerCountIdx == 0)
	{
		d_reducedCounts[blockIdx.x] = redCounts_shared[innerCountIdx];
	}
	d_toRedCounts[countIdx] = 0;
}

__global__ void update_centroids_kernel_3(float* d_centroids, float* d_reducedSums, int* d_reducedCounts, int N, int k, int n)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= k)
		return;

	if (d_reducedCounts[i] > 0)
	{
		for (int j = 0; j < n; j++)
		{
			d_centroids[i*n + j] = d_reducedSums[i*n + j] / (float)d_reducedCounts[i];
			d_reducedSums[i*n + j] = 0.0;
		}
		d_reducedCounts[i] = 0;
	}

}

CudaModule3::CudaModule3(float* h_objects, int N, int n, int k)
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
	h_counts = new int[k];
	for (int i = 0; i < k; i++)
		h_counts[i] = 0;
	h_avgs = new float[n*k];

	CC(cudaMalloc((void**)&d_toRedSums, tc* n* k * sizeof(float)));
	CC(cudaMalloc((void**)&d_toRedCounts, tc * k* sizeof(int)));

	CC(cudaMalloc((void**)&d_reducedSums,  k * n * sizeof(float)));
	CC(cudaMalloc((void**)&d_reducedCounts,  k * sizeof(int)));

	CC(cudaMalloc((void**)&d_avgs, k * sizeof(float)));

	CC(cudaMemcpy(d_objects, h_objects, N * n * sizeof(float), cudaMemcpyHostToDevice));

	cudaStreamCreate(&stream1);
	cudaStreamCreate(&stream2);

}

void CudaModule3::KMeans()
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
	int sharedMemoryBytes = tc * sizeof(float);
	cudaDeviceProp deviceProp;
	int deviceNum;
	cudaGetDevice(&deviceNum);
	cudaGetDeviceProperties(&deviceProp, deviceNum);

	if (sharedMemoryBytes > deviceProp.sharedMemPerBlock || objects_threads * 4 * n > deviceProp.sharedMemPerBlock) {
		printf("Not enough shared memory");
		return;
	}

	set_random_centroids_kernel_3 << <centroid_blocks, centroid_threads >> > (d_centroids, k, n);
	CC(cudaMemcpy(h_avgs, d_centroids, k*n * sizeof(float), cudaMemcpyDeviceToHost));

#ifdef TM
	CC(cudaEventRecord(stop));
	CC(cudaEventSynchronize(stop));
	milliseconds = 0;
	CC(cudaEventElapsedTime(&milliseconds, start, stop));
	std::cout << "set_random_centroids: " << milliseconds << " ms" << endl;
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
		calc_nearest_centroids_kernel_3 << <objects_blocks, objects_threads, objects_threads * 4 * n >> > (d_objects, d_objectsToCentroids, d_ifObjectChangedCentroid, N, d_centroids, k, n);
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
		calc_sums_counts_kernel_3 << <1, tc >> > (d_objects, d_objectsToCentroids, N, d_toRedSums, d_toRedCounts, k, n);		
#ifdef TM
		CC(cudaEventRecord(stop));
		CC(cudaEventSynchronize(stop));
		milliseconds = 0;
		CC(cudaEventElapsedTime(&milliseconds, start, stop));
		std::cout << "calc_sums_counts_kernel_3: " << milliseconds << " ms" << endl;
#endif	

#ifdef TM
		CC(cudaEventRecord(start));
#endif
		reduce_sums_kernel_3<<<n*k, tc, tc * sizeof(float) >>>( d_toRedSums,  d_reducedSums, k, n);
		reduce_counts_kernel_3<<<k, tc, tc * sizeof(float) >>>(d_toRedCounts, d_reducedCounts, k, n);
		cudaStreamSynchronize(stream1);
		cudaStreamSynchronize(stream2);		
#ifdef TM
		CC(cudaEventRecord(stop));
		CC(cudaEventSynchronize(stop));
		milliseconds = 0;
		CC(cudaEventElapsedTime(&milliseconds, start, stop));
		std::cout << "reduce_sums_counts_kernel: " << milliseconds << " ms" << endl;
#endif	

#ifdef TM
		CC(cudaEventRecord(start));
#endif
		update_centroids_kernel_3 << <centroid_blocks, centroid_threads >> > (d_centroids, d_reducedSums, d_reducedCounts, N, k, n);
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
		std::cout << "loop time: " << l_milliseconds << " ms" << endl;
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

	IOManager::WriteResults("results_CudaModule3.txt", n, k, finalCentroids);
	
}



