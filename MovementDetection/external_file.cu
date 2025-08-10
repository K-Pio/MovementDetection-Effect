#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <opencv2/opencv.hpp>

__global__ void substractKernel( unsigned char* f1, unsigned char* f2, unsigned char* nf, int width, int height );

cudaError_t substractWithCuda( int width, int height, cv::Mat frame1, cv::Mat frame2, cv::Mat* matrix );