#include "NoCudaModule.h"
#include "IOManager.h"

#include <time.h>
#include <stdlib.h>  
#include <iostream>
#include <time.h>
#include <algorithm>
#include <chrono>
//#define GTM
#define LTM

float noCudaDistSquare(float* a, float* b, int n)
{
	float ret = 0.0;
	for (int i = 0; i < n; i++)
	{
		ret += (a[i] - b[i])*(a[i] - b[i]);
	}
	return ret;
}

void calculateNearestCentroids(float* centroids, float* centroidSums, int* centroidCounts, int k, float * objects, int* objectsToCentroids, int N, int n, int* numberOfChangedCentroids)
{
	for (int i = 0; i < N; i++)
	{
		int minIdx = 0;
		int minDistSquare = noCudaDistSquare(objects + i * n, centroids, n);
		for (int j = 1; j < k; j++)
		{
			float d = noCudaDistSquare(objects + i * n, centroids + j * n, n);
			if (d < minDistSquare)
			{
				minDistSquare = d;
				minIdx = j;
			}
		}

		if (objectsToCentroids[i] != minIdx)
		{
			(*numberOfChangedCentroids)++;
			objectsToCentroids[i] = minIdx;
		}

		for (int j = 0; j < n; j++)
		{
			centroidSums[minIdx*n + j] += objects[i*n + j];
		}
		centroidCounts[minIdx]++;
	}
}

NoCudaModule::NoCudaModule(float* objects, int N, int n, int k)
{
	this->N = N;
	this->n = n;
	this->k = k;

	this->objects = objects;
	this->centroids = new float[k*n];
	this->objectsToCentroids = new int[N];
	this->centroidSums = new float[k*n];
	this->centroidCounts = new int[k];
}

void NoCudaModule::KMeans()
{
#ifdef GTM
	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	begin = std::chrono::steady_clock::now();
#endif

	srand(1234);
	// set random centroids
	for (int i = 0; i < n*k;i++)
	{
		centroids[i] = rand() % 10;
	}

	int changedCount = N;
	int iter = 0;
	while ((float)changedCount / (float)N > threshold)
	{
#ifdef LTM
		std::chrono::steady_clock::time_point begin_l = std::chrono::steady_clock::now();
		std::chrono::steady_clock::time_point end_l = std::chrono::steady_clock::now();
		begin_l = std::chrono::steady_clock::now();
#endif

		iter++;
		for (int i = 0; i < k; i++)
		{
			for (int p = 0; p < n; p++)
			{
				centroidSums[i*n + p] = 0.0;
			}
			centroidCounts[i] = 0;
		}

		changedCount = 0;
		calculateNearestCentroids(centroids, centroidSums, centroidCounts, k, objects, objectsToCentroids, N, n, &changedCount);

		for (int i = 0; i < k; i++)
		{
			if(centroidCounts[i]!=0)
				for (int p = 0; p < n; p++)
				{
					centroids[i*n + p] = centroidSums[i*n + p] / (float)centroidCounts[i];
				}
		}

#ifdef LTM
		end_l = std::chrono::steady_clock::now();
		std::cout << "loop time: " << (double)std::chrono::duration_cast<std::chrono::microseconds>(end_l - begin_l).count() / 1000.0 << " ms" << std::endl;
#endif

	}

#ifdef GTM
	end = std::chrono::steady_clock::now();
	std::cout << "summary: " << (double)std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1000.0 << " ms" << std::endl;
	std::cout << "iter: " << iter << std::endl;
#endif

	IOManager::WriteResults("results_NoCudaModule.txt", n, k, centroids);
}

