#include <iostream>
#include <fstream>
#include "CudaModule.h"
#include "CudaModule2.h"
#include "CudaModule3.h"
#include "NoCudaModule.h"
#include "IOManager.h"
using namespace std;




void printData(int N, int n, int k, float* objects)
{
	cout << "N: " << N << endl;
	cout << "n " << n << endl;
	cout << "k: " << k << endl;

	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < n; j++)
			cout << objects[i*n + j] << " ";
		cout << endl;
	}
}

int main()
{

	int N, n, k;
	float* objects = nullptr;
	float* centroids = nullptr;

	//IOManager::Generate("data.txt", 1024*512, 10, 50, 0, 10);

	IOManager::ReadData("data.txt", &N, &n, &k, &objects, &centroids);

	
	CudaModule cudaModule(objects, N, n, k);
	CudaModule2 cudaModule2(objects, N, n, k);
	CudaModule3 cudaModule3(objects, N, n, k);
	NoCudaModule noCudaModule(objects, N, n, k);

//////////////////////////////////////////////////////////
	//UNCOMMENT THE MODULE YOU WANT TO RUN
//////////////////////////////////////////////////////////

	cout << "===========\tcudaModule\t===========" << endl;
	cudaModule.KMeans();

	/*cout << "===========\tcudaModule2\t===========" << endl;
	cudaModule2.KMeans();*/

//////////////////////////////////////////////////////////
	//SPECIFY TC IN DEFINE - CUDAMODULE.CPP FILE (MAX 512, have to be less or equal to N)
//////////////////////////////////////////////////////////
	/*cout << "===========\tcudaModule3\t===========" << endl;
	cudaModule3.KMeans();

	cout << "===========\tnoCudaModule\t===========" << endl;
	noCudaModule.KMeans();*/


}