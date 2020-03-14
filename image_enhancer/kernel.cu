#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <opencv2/opencv.hpp>

#define MIN(x,y)  y ^ ((x ^ y) & -(x < y))  // calculating minimum of two unsigned integers without branching
#define MAX(x,y)  x ^ ((x ^ y) & -(x < y))  // calculating maximum of two unsigned integers without branching

#define LOG false  // for debugging

using namespace std;
using namespace cv;

__global__ void tmp(unsigned int* dev_min, unsigned int* dev_max, unsigned int num_channel);

__global__ void initialize(unsigned int* dev_min, unsigned int* dev_max, unsigned int num_channel);

template <unsigned int block_size>
__global__ void reduction(unsigned char* dev_vec, unsigned int* dev_min,
	unsigned int* dev_max, unsigned int size, unsigned int channel);

__global__ void enhance(unsigned char* dev_vec, unsigned int* dev_min, unsigned int* dev_max,
	unsigned int size, unsigned int num_channel);



int main()
{
	cudaError_t cudaStatus;

	Mat img = imread("original.jpg");  // Mat object for loading the input image
	unsigned int size = img.total();  // size of each channel (row*col)
	unsigned int num_channel = img.channels();  // number of channels
	unsigned int tot_size = size * num_channel;
	Mat* BGR = new Mat[num_channel];  // Mat object for splitting channels
	split(img, BGR);
	
	unsigned char *dev_vec;  // device array containing pixel values
	unsigned int *dev_min, * dev_max; // device arrays for containing minimum/maximum values of channels
	unsigned char* vec = new unsigned char[tot_size];  // host array containing pixel values

	// copying pixel values of channels to vec
	for (unsigned int i = 0; i < num_channel; i++)
	{
		for (unsigned int j = 0; j < size; j++)
		{
			vec[i * size + j] = (unsigned char)BGR[i].data[j];
		}
	}

	cudaStatus = cudaMalloc((void**)&dev_vec, tot_size);
	if (cudaStatus != cudaSuccess) { cerr << "memory allocation for dev_vec failed"; goto Error1; }
	cudaStatus = cudaMalloc((void**)&dev_min, num_channel * sizeof(unsigned int));
	if (cudaStatus != cudaSuccess) { cerr << "memory allocation for dev_min failed"; goto Error2; }
	cudaStatus = cudaMalloc((void**)&dev_max, num_channel * sizeof(unsigned int));
	if (cudaStatus != cudaSuccess) { cerr << "memory allocation for dev_max failed"; goto Error3; }

	cudaStatus = cudaMemcpy(dev_vec, vec, tot_size, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { cerr << "memory copying to device failed"; goto Error3; }
	initialize << <1, 32 >> > (dev_min, dev_max, num_channel);  // initalizing min and max arrays
	cudaDeviceSynchronize();
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) { cerr << "initializing min/max arrays failed"; goto Error3; }

	unsigned int size_grid_reduction(2), size_block_reduction(128);

	// each stream is responsible for finding min/max for one channel
	const unsigned int num_streams = 3;  
	if (num_streams < num_channel) {
		cerr << "num_stream must be equal or greater than num_channel" << endl; goto Error3;
	}

	cudaStream_t streams[num_streams];
	for (unsigned int i = 0; i < num_streams; i++) {
		cudaStreamCreate(&streams[i]);
	}

	// reduction kernel
	for (unsigned int i = 0; i < num_channel; i++)
	{
		switch (size_block_reduction)
		{
		case 1024:
			reduction<1024> << <size_grid_reduction, size_block_reduction, size_block_reduction, streams[i] >> > 
				(&dev_vec[i*size], dev_min, dev_max, size, i); break;
		case 512:
			reduction<512> << <size_grid_reduction, size_block_reduction, size_block_reduction, streams[i] >> >
				(&dev_vec[i * size], dev_min, dev_max, size, i); break;
		case 256:
			reduction<256> << <size_grid_reduction, size_block_reduction, size_block_reduction, streams[i] >> >
				(&dev_vec[i * size], dev_min, dev_max, size, i); break;
		case 128:
			reduction<128> << <size_grid_reduction, size_block_reduction, size_block_reduction, streams[i] >> >
				(&dev_vec[i * size], dev_min, dev_max, size, i); break;
		case 64:
			reduction<64> << <size_grid_reduction, size_block_reduction, size_block_reduction, streams[i] >> >
				(&dev_vec[i * size], dev_min, dev_max, size, i); break;
		case 32:
			reduction<32> << <size_grid_reduction, size_block_reduction, size_block_reduction, streams[i] >> >
				(&dev_vec[i * size], dev_min, dev_max, size, i); break;
		}
	}
	cudaDeviceSynchronize();
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) { cerr << "reduction failed"; goto Error3; }

	if (LOG) tmp<<<1,32>>>(dev_min,  dev_max, num_channel);

	unsigned int size_grid_enhance(2), size_block_enhance(128);

	// enhancing pixel values
	enhance << <size_grid_enhance, size_block_enhance >> >
		(dev_vec, dev_min, dev_max, size, num_channel);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) { cerr << "enhance"; goto Error3; }

	cudaStatus = cudaMemcpy(vec, dev_vec, tot_size, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) { cerr << "memory copying to host failed"; goto Error3; }

	for (unsigned int i = 0; i < num_channel; i++)
	{
		for (unsigned int j = 0; j < size; j++)
		{
			 BGR[i].data[j] = (uint8_t)vec[i * size + j];
		}
	}

	// merging all channels to a single Mat object
	merge(BGR, num_channel, img);
	imwrite("enhanced.jpg", img);

	for (unsigned int i = 0; i < num_streams; i++) {
		cudaStreamDestroy(streams[i]);
	}

Error3:
	cudaFree(dev_max);
Error2:
	cudaFree(dev_min);
Error1:
	cudaFree(dev_vec);
	delete[] BGR;
	delete[] vec;

	return 0;
}




__global__ void initialize(unsigned int* dev_min, unsigned int* dev_max, unsigned int num_channel)
{
	unsigned int gid = threadIdx.x + blockDim.x * gridDim.x;
	while (gid < num_channel)
	{
		dev_min[gid] = 255u;
		dev_max[gid] = 0;
		gid += blockDim.x * gridDim.x;
	}

}





template <unsigned int block_size>
__global__ void reduction(unsigned char* dev_vec, unsigned int* dev_min,
	unsigned int* dev_max, unsigned int size, unsigned int channel)
{
	extern __shared__ unsigned char SHARED_MIN[];
	unsigned char* SHARED_MAX = &SHARED_MIN[block_size];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * block_size + tid;
	unsigned int grid_size = block_size * gridDim.x;
	SHARED_MIN[tid] = 255u;
	SHARED_MAX[tid] = 0;

	while (i < size)
	{
		SHARED_MIN[tid] = MIN(SHARED_MIN[tid], dev_vec[i]);
		SHARED_MAX[tid] = MAX(SHARED_MAX[tid], dev_vec[i]);
		i += grid_size;
	}
	__syncthreads();

	if (block_size >= 512)
	{
		if (tid < 256)
		{
			SHARED_MIN[tid] = MIN(SHARED_MIN[tid], SHARED_MIN[tid + 256]);
			SHARED_MAX[tid] = MAX(SHARED_MAX[tid], SHARED_MAX[tid + 256]);
		}
		__syncthreads();
	}

	if (block_size >= 256)
	{
		if (tid < 128)
		{
			SHARED_MIN[tid] = MIN(SHARED_MIN[tid], SHARED_MIN[tid + 128]);
			SHARED_MAX[tid] = MAX(SHARED_MAX[tid], SHARED_MAX[tid + 128]);
		}
		__syncthreads();
	}

	if (block_size >= 128)
	{
		if (tid < 64)
		{
			SHARED_MIN[tid] = MIN(SHARED_MIN[tid], SHARED_MIN[tid + 64]);
			SHARED_MAX[tid] = MAX(SHARED_MAX[tid], SHARED_MAX[tid + 64]);
		}
		__syncthreads();
	}

	if (block_size >= 64)
	{
		if (tid < 32)
		{
			SHARED_MIN[tid] = MIN(SHARED_MIN[tid], SHARED_MIN[tid + 32]);
			SHARED_MAX[tid] = MAX(SHARED_MAX[tid], SHARED_MAX[tid + 32]);
		}
		__syncthreads();
	}

	if (tid < 32)
	{
		SHARED_MIN[tid] = MIN(SHARED_MIN[tid], SHARED_MIN[tid + 16]);
		SHARED_MIN[tid] = MIN(SHARED_MIN[tid], SHARED_MIN[tid + 8]);
		SHARED_MIN[tid] = MIN(SHARED_MIN[tid], SHARED_MIN[tid + 4]);
		SHARED_MIN[tid] = MIN(SHARED_MIN[tid], SHARED_MIN[tid + 2]);
		SHARED_MIN[tid] = MIN(SHARED_MIN[tid], SHARED_MIN[tid + 1]);

		SHARED_MAX[tid] = MAX(SHARED_MAX[tid], SHARED_MAX[tid + 16]);
		SHARED_MAX[tid] = MAX(SHARED_MAX[tid], SHARED_MAX[tid + 8]);
		SHARED_MAX[tid] = MAX(SHARED_MAX[tid], SHARED_MAX[tid + 4]);
		SHARED_MAX[tid] = MAX(SHARED_MAX[tid], SHARED_MAX[tid + 2]);
		SHARED_MAX[tid] = MAX(SHARED_MAX[tid], SHARED_MAX[tid + 1]);
	}
	if (tid == 0)
	{
		atomicMin(&dev_min[channel], (unsigned int) SHARED_MIN[0]);
		atomicMax(&dev_max[channel], (unsigned int) SHARED_MAX[0]);
	}
}





__global__ void enhance(unsigned char* dev_vec, unsigned int* dev_min, unsigned int* dev_max,
	unsigned int size, unsigned int num_channel)
{
	unsigned int grid_size = blockDim.x * gridDim.x;
	for (unsigned int i = 0; i < num_channel; i++)
	{
		unsigned int gid = threadIdx.x + blockIdx.x * blockDim.x;
		unsigned char MIN_PIXEL = dev_min[i];
		unsigned char RANGE_PIXEL = (dev_max[i] - dev_min[i]);
		while (gid < size)
		{
			dev_vec[i * size + gid] = (dev_vec[i * size + gid] - MIN_PIXEL) / (float)RANGE_PIXEL * 255;
			gid += grid_size;
		}
	}
}



__global__ void tmp(unsigned int* dev_min, unsigned int* dev_max, unsigned int num_channel)
{
	unsigned int i = threadIdx.x;
	if (threadIdx.x < num_channel)
	{
		printf("min[%u]= %u,    max[%u]= %u\n", i, dev_min[i], i, dev_max[i]);
	}
}