#include <iostream>
#include <fstream>
#include <time.h>
#include "IOManager.h"

using namespace std;

void IOManager::ReadData(string filename, int* N, int* n, int* k, float** objects, float** centroids)
{
	string file = "..\\" + filename;
	fstream dataFile(file, std::ios_base::in);

	if (dataFile.is_open())
	{
		dataFile >> *N;
		dataFile >> *n;
		dataFile >> *k;

		*objects = new float[(*N)*(*n)];
		*centroids = new float[(*k)*(*n)];

		for (int i = 0; i < *N; i++)
		{
			for (int j = 0; j < *n; j++)
			{
				float number;
				dataFile >> number;
				(*objects)[i*(*n) + j] = number;
			}
		}
	}
	else
	{
		cout << "Could not open the file" << endl;
	}
}

void IOManager::WriteResults(string filename, int n, int k, float* centroids)
{
	fstream dataFile("..\\" + filename, std::ios_base::out);

	if (dataFile.is_open())
	{
		for (int i = 0; i < k; i++)
		{
			for (int j = 0; j < n; j++)
			{
				dataFile << centroids[i*n + j]<< " ";
			}
			dataFile << endl;
		}
	}
	else
	{
		cout << "Could not create the file" << endl;
	}
}

void IOManager::Generate(string filename, int N, int n, int k, float minVal, float maxVal)
{
	srand(time(NULL));

	fstream dataFile("..\\" + filename, std::ios_base::out);

	if (dataFile.is_open())
	{
		dataFile << N << endl;
		dataFile << n << endl;
		dataFile << k << endl;

		for (int i = 0; i <N; i++)
		{
			for (int j = 0; j <n; j++)
			{
				float number = rand()%(int)(maxVal-minVal+1)+minVal;
				dataFile << number << " ";
			}
			dataFile << endl;
		}
	}
	else
	{
		cout << "Could not create the file" << endl;
	}
}

void IOManager::GenerateConstant(string filename, int N, int n, int k)
{
	fstream dataFile("..\\" + filename, std::ios_base::out);

	if (dataFile.is_open())
	{
		dataFile << N << endl;
		dataFile << n << endl;
		dataFile << k << endl;

		for (int i = 0; i < N; i++)
		{
			for (int j = 0; j < n; j++)
			{
				float number = j;
				dataFile << number << " ";
			}
			dataFile << endl;
		}
	}
	else
	{
		cout << "Could not create the file" << endl;
	}
}