

class NoCudaModule
{
	int N;
	int n;
	int k;

	float* objects;
	float* centroids;
	int* objectsToCentroids;
	float* centroidSums;
	int* centroidCounts;
	float threshold = 0.0001;

public:
	NoCudaModule(float* objects, int N, int n, int k);
	void KMeans();

};