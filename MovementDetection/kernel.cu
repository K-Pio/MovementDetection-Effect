
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <iostream>
#include <opencv2/opencv.hpp>


/**
*   \brief Base subtraction - allocate data on device and do calculations
*
*   More complete description...
*
*   \param width image width.
*	\param height image height.
*	\param[in] frame1 first frame to subtract - minuend
*	\param[in] frame1 second frame to subtract - subtrahend
*	\param[out] matrix pointer to image matrix
*   \return cudaError_t
*
**/
cudaError_t subtractWithCuda( int width, int height, cv::Mat frame1, cv::Mat frame2, cv::Mat* matrix );
/**
*   \brief Base subtraction - allocate data on device and do calculations
*
*   Method fitted for grayscale frames
*
*   \param width image width.
*	\param height image height.
*	\param[in] frame1 first frame to subtract - minuend
*	\param[in] frame1 second frame to subtract - subtrahend
*	\param[out] matrix pointer to image matrix
*   \return cudaError_t
*
**/
cudaError_t subtractWithCuda_BnW( int width, int height, cv::Mat frame1, cv::Mat frame2, cv::Mat* matx );
/**
*   \brief Base subtraction - allocate data on device and do calculations
*
*   Method fitted for colorscale frames.
*	can operate in 2D and 3D grid structure
*
*   \param width image width.
*	\param height image height.
*	\param[in] frame1 first frame to subtract - minuend
*	\param[in] frame1 second frame to subtract - subtrahend
*	\param[out] matrix pointer to image matrix
*	\param type3d switch to select grid type
*   \return cudaError_t
*
**/
cudaError_t subtractWithCuda_Color( int width, int height, cv::Mat frame1, cv::Mat frame2, cv::Mat* matx, bool type3d = false );

__global__ void subtractKernel( unsigned char* f1, unsigned char* f2, unsigned char* nf, int width, int height );
__global__ void color_subtractKernel2D( unsigned char* f1, unsigned char* f2, unsigned char* nf, int width, int height );
__global__ void color_subtractKernel3D( unsigned char* f1, unsigned char* f2, unsigned char* nf, int width, int height );


void print_frame_info( cv::Mat frame ) {
	std::cout << "Matrix size: " << frame.rows << "x" << frame.cols << std::endl;
	std::cout << "channels: " << frame.channels() << std::endl;
	std::cout << "Data type: " << frame.type() << std::endl;
	std::cout << "Frame size:" << frame.size() << std::endl;
}
void set_frame( cv::VideoCapture* cap, int frameNumber ) {
	// Skip to frameNumber frame
	// int frameNumber = 10;
	cap->set( cv::CAP_PROP_POS_FRAMES, frameNumber - 1 ); // Set frame position (index begins from 0)
}
void Subtract_with_cv2( cv::Mat grayFrame1, cv::Mat grayFrame2, cv::Mat* image ) {
	cv::subtract( grayFrame1, grayFrame2, *image );
}


void blackNwhiteSubtract( cv::Mat frame1, cv::Mat frame2, cv::Mat* newFrame, int rows, int cols );
void colorSubtract( cv::Mat frame1, cv::Mat frame2, cv::Mat* newFrame, int rows, int cols, bool type3d = false );


int main( int argc, char* argv[] )
{

	std::string input_video_path;
	std::string output_video_path; // = "C:\\Users\\kaspi\\Desktop\\share\\new4.mp4";
	bool saveOutput;
	bool colorScale = false;
	bool method3d = true;
	float transparencyFactor = 0.0;

	std::map<std::string, std::string> inputMap;

	// input decoding
	if( argc % 2 == 0 )
	{
		std::cerr << "Incorrect input map\n";
		return 1;
	}
	for( int i = 1; i < argc; i += 2 )
	{
		std::string key = argv[i];
		std::string value = argv[i + 1];
		inputMap[key] = value;
	}
	if( inputMap.find( "input" ) != inputMap.end() )
	{
		input_video_path = inputMap["input"];
	}
	else
	{
		std::cerr << "Missing input path\n";
		return 1;
	}
	if( inputMap.find( "output" ) != inputMap.end() )
	{
		saveOutput = true;
		output_video_path = inputMap["output"];
	}
	else
	{
		saveOutput = false;
	}
	if( inputMap.find( "color" ) != inputMap.end() )
	{
		if( inputMap["color"] == "color" )
		{
			colorScale = true;
		}
		else
		{
			colorScale = false;
		}
	}
	if( inputMap.find( "method" ) != inputMap.end() )
	{
		if( inputMap["method"] == "3d" )
		{
			method3d = true;
		}
		else if( inputMap["method"] == "2d" )
		{
			method3d = false;
		}
		else
		{
			std::cout << "error";
		}
	}
	if( inputMap.find( "transparency" ) != inputMap.end() )
	{
		float transparency;
		try
		{
			transparency = std::stof( inputMap["transparency"] );
		}
		catch( const std::invalid_argument& )
		{
			std::cerr << "Invalid agument\n";
			return 1;
		}

		if( transparency > 0.0 && transparency < 1.0 )
		{
			transparencyFactor = transparency;
		}
		else
		{
			std::cerr << "Invalid value\n";
			return 1;
		}
	}

	/*if( argc > 1 )
	{
		input_video_path = argv[1];
		if( "color" == "colorr" )
		{
			colorScale = true;
		}
		else
		{
			colorScale = false;
		}
		output_video_path = "C:\\Users\\kaspi\\Desktop\\share\\new4.mp4";
		transparency = 0.85;
		method3d = false;
	}
	else
	{
		std::cerr << "No input provided" << std::endl;
		return -1;
	}*/

	// const std::string input_video_path = "C:\\Users\\kaspi\\Downloads\\unexpected_shit.mp4";

	// Open video file and 
	cv::VideoCapture cap( input_video_path );
	if( !cap.isOpened() )
	{
		std::cerr << "Cannot open video file!" << std::endl;
		return -1;
	}

	// Get width, height and framer per seconds
	const int width = static_cast<int>(cap.get( cv::CAP_PROP_FRAME_WIDTH ));
	const int height = static_cast<int>(cap.get( cv::CAP_PROP_FRAME_HEIGHT ));
	const double fps = cap.get( cv::CAP_PROP_FPS );

	std::cout << "fps: " << fps << std::endl;
	
	cv::VideoWriter writer(output_video_path,
		cv::VideoWriter::fourcc( 'm', 'p', '4', 'v' ), // Codec
		fps,
		cv::Size( width, height), 
		colorScale );

	if( !writer.isOpened() )
	{
		std::cerr << "Nie można otworzyć pliku wyjściowego!" << std::endl;
		return -1;
	}
	
	// debug | set frame
	//set_frame(&cap, 213);

	cv::Mat frame1, frame2, frame, newFrame, image;
	int frameCount = 0;

	// debug
	/*int ct = 0;
	double totalTime = 0.0;
	int frameCunt = 0;*/
	
	cap >> frame2;
	const cv::Size s = frame2.size();
	const int rows = s.height;
	const int cols = s.width;

	while( true )
	{
		// auto start = std::chrono::high_resolution_clock::now(); // time debug

		// Capture two next frames
		frame1 = frame2.clone();
		cap >> frame2;
		frameCount += 2;

		if( frame1.empty() || frame2.empty() ) {
			break; // no more frames / end of video
		}

		if( colorScale )
		{
			if( transparencyFactor > 0.0)
			{
				frame2.convertTo( frame, -1, transparencyFactor, 0 );
				colorSubtract( frame1, frame, &image, rows, cols, method3d );
			}
			else
			{
				colorSubtract( frame1, frame2, &image, rows, cols, method3d );
			}
		}
		else
		{
			blackNwhiteSubtract( frame1, frame2, &image, rows, cols );
		}
		// save frame to new file
		writer.write( image );

		// debug | show old and new frame
		/*cv::imshow( "old", frame1 );
		cv::imshow( "new", image );*/
		// debug | fast break
		/*if( ct == 100 )
		{
			break;
		}
		else
		{
			ct++;
		}*/
		// debug | execution time
		/*auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsed = end - start;
		std::cout << "Frame processing time: " << elapsed.count() << " sec" << std::endl;
		totalTime += elapsed.count();
		frameCunt++;*/
		
		if( cv::waitKey( 30 ) == 'q' ) 
		{
			break;
		}
	}

	// debug | execution time
	/*if( frameCount > 0 )
	{
		double averageTime = totalTime / frameCount;
		std::cout << "Total processing time: " << totalTime << " sec" << std::endl;
		std::cout << "Avarage frame processing time: " << averageTime << " sec" << std::endl;
	}*/

	writer.release();
	std::cout << "writer released" << std::endl;
	cap.release();

	if( cv::waitKey( 3000 ) == 'q' ) {
		cv::destroyAllWindows();
	}
	
	// debug
	// cv::destroyAllWindows(); 

	return 0;
}

void blackNwhiteSubtract(cv::Mat frame1, cv::Mat frame2, cv::Mat* newFrame, int rows, int cols)
{
	cv::Mat grayFrame1, grayFrame2;
	cv::cvtColor( frame1, grayFrame1, cv::COLOR_BGR2GRAY );
	cv::cvtColor( frame2, grayFrame2, cv::COLOR_BGR2GRAY );
	*newFrame = cv::Mat::zeros( rows, cols, CV_8UC1 );
	cudaError_t cudaStatus = subtractWithCuda( cols, rows, grayFrame1, grayFrame2, newFrame );
}

void colorSubtract( cv::Mat frame1, cv::Mat frame2, cv::Mat* newFrame, int rows, int cols, bool type3d )
{
	*newFrame = cv::Mat::zeros( rows, cols, CV_8UC3 );
	cudaError_t cudaStatus = subtractWithCuda_Color( cols, rows, frame1, frame2, newFrame, type3d );
}


cudaError_t subtractWithCuda_BnW( int width, int height, cv::Mat frame1, cv::Mat frame2, cv::Mat* matx )
{
	cudaError_t cudaStatus;
	uchar* d_frame1;
	uchar* d_frame2;
	uchar* d_frame_new;

	const size_t size = width * height * sizeof( uchar );

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice( 0 );
	if( cudaStatus != cudaSuccess )
	{
		fprintf( stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?" );
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc( (void**)&d_frame1, size );
	if( cudaStatus != cudaSuccess )
	{
		fprintf( stderr, "cudaMalloc failed!" );
		goto Error;
	}
	cudaStatus = cudaMalloc( (void**)&d_frame2, size );
	if( cudaStatus != cudaSuccess )
	{
		fprintf( stderr, "cudaMalloc failed!" );
		goto Error;
	}
	cudaStatus = cudaMalloc( (void**)&d_frame_new, size );
	if( cudaStatus != cudaSuccess )
	{
		fprintf( stderr, "cudaMalloc failed!" );
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy( d_frame1, frame1.data, size, cudaMemcpyHostToDevice );
	if( cudaStatus != cudaSuccess )
	{
		fprintf( stderr, "cudaMemcpy failed!" );
		goto Error;
	}
	cudaStatus = cudaMemcpy( d_frame2, frame2.data, size, cudaMemcpyHostToDevice );
	if( cudaStatus != cudaSuccess )
	{
		fprintf( stderr, "cudaMemcpy failed!" );
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	dim3 threadsPerBlock( 32, 32 ); // Block size: 32x32 threads | 256
	dim3 numBlocks( (width + threadsPerBlock.x - 1) / threadsPerBlock.x,
		(height + threadsPerBlock.y - 1) / threadsPerBlock.y );

	subtractKernel << <numBlocks, threadsPerBlock >> > (d_frame1, d_frame2, d_frame_new, width, height);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if( cudaStatus != cudaSuccess )
	{
		fprintf( stderr, "aubKernel launch failed: %s\n", cudaGetErrorString( cudaStatus ) );
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if( cudaStatus != cudaSuccess )
	{
		fprintf( stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus );
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy( matx->data, d_frame_new, size, cudaMemcpyDeviceToHost );
	if( cudaStatus != cudaSuccess )
	{
		fprintf( stderr, "cudaMemcpy failed!" );
		goto Error;
	}

Error:
	cudaFree( d_frame1 );
	cudaFree( d_frame2 );
	cudaFree( d_frame_new );

	return cudaStatus;
}

cudaError_t subtractWithCuda_Color( int width, int height, cv::Mat frame1, cv::Mat frame2, cv::Mat* matx , bool type3d )
{
	cudaError_t cudaStatus;
	uchar* d_frame1;
	uchar* d_frame2;
	uchar* d_frame_new;

	const size_t size = 3 *width * height * sizeof( uchar );

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice( 0 );
	if( cudaStatus != cudaSuccess )
	{
		fprintf( stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?" );
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output).
	cudaStatus = cudaMalloc( (void**)&d_frame1, size );
	if( cudaStatus != cudaSuccess )
	{
		fprintf( stderr, "cudaMalloc failed!" );
		goto Error;
	}
	cudaStatus = cudaMalloc( (void**)&d_frame2, size );
	if( cudaStatus != cudaSuccess )
	{
		fprintf( stderr, "cudaMalloc failed!" );
		goto Error;
	}
	cudaStatus = cudaMalloc( (void**)&d_frame_new, size );
	if( cudaStatus != cudaSuccess )
	{
		fprintf( stderr, "cudaMalloc failed!" );
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy( d_frame1, frame1.data, size, cudaMemcpyHostToDevice );
	if( cudaStatus != cudaSuccess )
	{
		fprintf( stderr, "cudaMemcpy failed!" );
		goto Error;
	}
	cudaStatus = cudaMemcpy( d_frame2, frame2.data, size, cudaMemcpyHostToDevice );
	if( cudaStatus != cudaSuccess )
	{
		fprintf( stderr, "cudaMemcpy failed!" );
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	if( type3d )
	{
		dim3 threadsPerBlock( 16, 16, 3 ); // 16x16 threads + 3 layers for BGR
		dim3 numBlocks( (width + threadsPerBlock.x - 1) / threadsPerBlock.x,
						(height + threadsPerBlock.y - 1) / threadsPerBlock.y );
		color_subtractKernel3D << <numBlocks, threadsPerBlock >> > (d_frame1, d_frame2, d_frame_new, width, height);
	}
	else
	{
		dim3 threadsPerBlock( 32, 32 ); // Block size: 32x32 threads | 256
		dim3 numBlocks( (3 * width + threadsPerBlock.x - 1) / threadsPerBlock.x,
						(height + threadsPerBlock.y - 1) / threadsPerBlock.y );
		color_subtractKernel2D << <numBlocks, threadsPerBlock >> > (d_frame1, d_frame2, d_frame_new, width, height);
	}

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if( cudaStatus != cudaSuccess )
	{
		fprintf( stderr, "aubKernel launch failed: %s\n", cudaGetErrorString( cudaStatus ) );
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if( cudaStatus != cudaSuccess )
	{
		fprintf( stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus );
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy( matx->data, d_frame_new, size, cudaMemcpyDeviceToHost );
	if( cudaStatus != cudaSuccess )
	{
		fprintf( stderr, "cudaMemcpy failed!" );
		goto Error;
	}

Error:
	cudaFree( d_frame1 );
	cudaFree( d_frame2 );
	cudaFree( d_frame_new );

	return cudaStatus;
}

cudaError_t subtractWithCuda( int width, int height, cv::Mat frame1, cv::Mat frame2, cv::Mat* matx )
{
	cudaError_t cudaStatus;
	uchar* d_frame1;
	uchar* d_frame2;
	uchar* d_frame_new;

	const size_t size = width * height * sizeof( uchar );

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice( 0 );
	if( cudaStatus != cudaSuccess ) {
		fprintf( stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?" );
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc( (void**)&d_frame1, size );
	if( cudaStatus != cudaSuccess ) {
		fprintf( stderr, "cudaMalloc failed!" );
		goto Error;
	}
	cudaStatus = cudaMalloc( (void**)&d_frame2, size );
	if( cudaStatus != cudaSuccess ) {
		fprintf( stderr, "cudaMalloc failed!" );
		goto Error;
	}
	cudaStatus = cudaMalloc( (void**)&d_frame_new, size );
	if( cudaStatus != cudaSuccess ) {
		fprintf( stderr, "cudaMalloc failed!" );
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy( d_frame1, frame1.data, size, cudaMemcpyHostToDevice );
	if( cudaStatus != cudaSuccess ) {
		fprintf( stderr, "cudaMemcpy failed!" );
		goto Error;
	}
	cudaStatus = cudaMemcpy( d_frame2, frame2.data, size, cudaMemcpyHostToDevice );
	if( cudaStatus != cudaSuccess ) {
		fprintf( stderr, "cudaMemcpy failed!" );
		goto Error;
	}

	// std::cout << "rodeo\n";

	// Launch a kernel on the GPU with one thread for each element.
	dim3 threadsPerBlock( 32, 32 ); // Block size: 32x32 threads | 256
	dim3 numBlocks( (width + threadsPerBlock.x - 1) / threadsPerBlock.x,
		(height + threadsPerBlock.y - 1) / threadsPerBlock.y );

	subtractKernel << <numBlocks, threadsPerBlock >> > (d_frame1, d_frame2, d_frame_new, width, height);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if( cudaStatus != cudaSuccess ) {
		fprintf( stderr, "aubKernel launch failed: %s\n", cudaGetErrorString( cudaStatus ) );
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if( cudaStatus != cudaSuccess ) {
		fprintf( stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus );
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy( matx->data, d_frame_new, size, cudaMemcpyDeviceToHost );
	if( cudaStatus != cudaSuccess ) {
		fprintf( stderr, "cudaMemcpy failed!" );
		goto Error;
	}

Error:
	cudaFree( d_frame1 );
	cudaFree( d_frame2 );
	cudaFree( d_frame_new );

	return cudaStatus;
}

// Kernels
__global__ void subtractKernel( unsigned char* f1, unsigned char* f2, unsigned char* nf, int width, int height ) {

	int x = blockIdx.x * blockDim.x + threadIdx.x; // X index in grid
	int y = blockIdx.y * blockDim.y + threadIdx.y; // Y index in grid
	int i = y * width + x; // Transformation to 1D 

	if( x < width && y < height ) // Fit border 
	{
		int diff = f2[i] - f1[i];

		// nf[i] = (diff > 0) ? diff : 0; // Subtraction with non negative 
		nf[i] = (diff > 0) ? diff : -diff; // absolute value from difference
	}
}

__global__ void color_subtractKernel2D( unsigned char* f1, unsigned char* f2, unsigned char* nf, int width, int height )
{

	int x = blockIdx.x * blockDim.x + threadIdx.x; // X index in grid
	int y = blockIdx.y * blockDim.y + threadIdx.y; // Y index in grid
	int i = y * width + x; // Transformation to 1D 
	int idx = 3 * i;

	if( x < width && y < height ) // Fit border 
	{
		// subtract for different channels: B, G, R
		int diffB = f2[idx] - f1[idx];         // B
		int diffG = f2[idx + 1] - f1[idx + 1]; // G
		int diffR = f2[idx + 2] - f1[idx + 2]; // R

		// Save abs diff for each channels
		nf[idx] = abs( diffB );
		nf[idx + 1] = abs( diffG );
		nf[idx + 2] = abs( diffR );
	}
}

__global__ void color_subtractKernel3D( unsigned char* f1, unsigned char* f2, unsigned char* nf, int width, int height )
{

	int x = blockIdx.x * blockDim.x + threadIdx.x; // X index in grid
	int y = blockIdx.y * blockDim.y + threadIdx.y; // Y index in grid
	int c = threadIdx.z;
	int i = y * width + x; // Transformation to 1D 
	int idx = 3 * i + c;

	if( x < width && y < height ) // Fit border 
	{
		nf[idx] = abs( f2[idx] - f1[idx] );
	}
}
