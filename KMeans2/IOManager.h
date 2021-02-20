#ifndef IOMANGER
#define IOMANGER

#include <iostream>
using namespace std; 

static class IOManager
{
public:
	static void Generate(string filename, int N, int n, int k, float minVal, float maxVal);
	static void GenerateConstant(string filename, int N, int n, int k);
	static void ReadData(string filename, int* N, int* n, int* k, float** objects, float** centroids);
	static void WriteResults(string filename, int n, int k, float* centroids);
};

#endif