
#include <stdio.h>
#include <include/labwork.h>
#include <cuda_runtime_api.h>
#include <omp.h>
#include <math.h>

#define ACTIVE_THREADS 4


int main(int argc, char **argv) {
    printf("USTH ICT Master 2017, Advanced Programming for HPC.\n");
    if (argc < 2) {
        printf("Usage: labwork <lwNum> <inputImage>\n");
        printf("   lwNum        labwork number\n");
        printf("   inputImage   the input file name, in JPEG format\n");
        return 0;
    }

    int lwNum = atoi(argv[1]);
    std::string inputFilename;

    // pre-initialize CUDA to avoid incorrect profiling
    printf("Warming up...\n");
    char *temp;
    cudaMalloc(&temp, 1024);

    Labwork labwork;
    if (lwNum != 2 ) {
        inputFilename = std::string(argv[2]);
        labwork.loadInputImage(inputFilename);
    }

    printf("Starting labwork %d\n", lwNum);
    Timer timer;
    timer.start();
    switch (lwNum) {
        case 1:
            labwork.labwork1_CPU();
            labwork.saveOutputImage("labwork2-cpu-out.jpg");
            printf("labwork 1 CPU ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            timer.start();
	    labwork.labwork1_OpenMP();
            labwork.saveOutputImage("labwork2-openmp-out.jpg");
            printf("labwork 1 OpenMP ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec()); 
            break;
        case 2:
            labwork.labwork2_GPU();
            break;
        case 3:
            labwork.labwork3_GPU();
            labwork.saveOutputImage("labwork3-gpu-out.jpg");
            break;
        case 4:
            labwork.labwork4_GPU();
            labwork.saveOutputImage("labwork4-gpu-out.jpg");
            break;
        case 5:
            labwork.labwork5_CPU();
            labwork.saveOutputImage("labwork5-cpu-out.jpg");
            labwork.labwork5_GPU();
            labwork.saveOutputImage("labwork5-gpu-out.jpg");
            break;
        case 6:
            labwork.labwork6_GPU();
            labwork.saveOutputImage("labwork6-gpu-out.jpg");
            break;
        case 7:
            labwork.labwork7_GPU();
            labwork.saveOutputImage("labwork7-gpu-out.jpg");
            break;
        case 8:
            labwork.labwork8_GPU();
            labwork.saveOutputImage("labwork8-gpu-out.jpg");
            break;
        case 9:
            labwork.labwork9_GPU();
            labwork.saveOutputImage("labwork9-gpu-out.jpg");
            break;
        case 10:
            labwork.labwork10_GPU();
            labwork.saveOutputImage("labwork10-gpu-out.jpg");
            break;
    }
    printf("labwork %d ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
}

void Labwork::loadInputImage(std::string inputFileName) {
    inputImage = jpegLoader.load(inputFileName);
}

void Labwork::saveOutputImage(std::string outputFileName) {
    jpegLoader.save(outputFileName, outputImage, inputImage->width, inputImage->height, 90);
}

void Labwork::labwork1_CPU() {
    int pixelCount = inputImage->width * inputImage->height;
    outputImage = static_cast<char *>(malloc(pixelCount * 3));
    for (int j = 0; j < 100; j++) {		// let's do it 100 times, otherwise it's too fast!
        for (int i = 0; i < pixelCount; i++) {
            outputImage[i * 3] = (char) (((int) inputImage->buffer[i * 3] + (int) inputImage->buffer[i * 3 + 1] +
                                          (int) inputImage->buffer[i * 3 + 2]) / 3);
            outputImage[i * 3 + 1] = outputImage[i * 3];
            outputImage[i * 3 + 2] = outputImage[i * 3];
        }    }
}

void Labwork::labwork1_OpenMP() {
    int pixelCount = inputImage->width * inputImage->height;
    outputImage = static_cast<char *>(malloc(pixelCount * 3));
    omp_set_num_threads(12); // blocs size
    #pragma omp parallel for schedule(dynamic, 3)
    for (int j = 0; j < 100; j++) {             // let's do it 100 times, otherwise it's too fast!
        for (int i = 0; i < pixelCount; i++) {
            outputImage[i * 3] = (char) (((int) inputImage->buffer[i * 3] + (int) inputImage->buffer[i * 3 + 1] +
                                          (int) inputImage->buffer[i * 3 + 2]) / 3);
            outputImage[i * 3 + 1] = outputImage[i * 3];
            outputImage[i * 3 + 2] = outputImage[i * 3];
        }
    }
}

int getSPcores(cudaDeviceProp devProp) {
    int cores = 0;
    int mp = devProp.multiProcessorCount;
    switch (devProp.major) {
        case 2: // Fermi
            if (devProp.minor == 1) cores = mp * 48;
            else cores = mp * 32;
            break;
        case 3: // Kepler
            cores = mp * 192;
            break;
        case 5: // Maxwell
            cores = mp * 128;
            break;
        case 6: // Pascal
            if (devProp.minor == 1) cores = mp * 128;
            else if (devProp.minor == 0) cores = mp * 64;
            else printf("Unknown device type\n");
            break;
        default:
            printf("Unknown device type\n");
            break;
    }
    return cores;
}

void Labwork::labwork2_GPU() {

    int devCount;
    cudaGetDeviceCount(&devCount);
    printf("Device number : %d\n",devCount);
    for(int i = 0; i < devCount; ++i)
    {
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, i);
        printf("Major : %d\n",props.major);
        printf("Total global memory : %zu\n",props.totalGlobalMem);
        printf("Shared memory peu block : %zu\n",props.sharedMemPerBlock);
        //printf("%s\n",props.totalConstMem);
        printf("Registers per block : %d\n",props.regsPerBlock);
	printf("Clock rate : %d\n",props.clockRate);
	printf("Multiprocessor count : %d\n",props.multiProcessorCount);
	printf("Memory Bus Width : %d\n",props.memoryBusWidth);
	printf("  Peak Memory Bandwidth (GB/s): %f\n\n", 2.0*props.memoryClockRate*(props.memoryBusWidth/8)/1.0e6);// source :devblogs.nvidia
        printf("Warp size : %d\n",props.warpSize);
        printf("Max threads per blocks : %d\n",props.maxThreadsPerBlock);
        printf("Max threads dimension :\n 1 : %7d\n2 : %7d\n3 : %7d\n",props.maxThreadsDim[0],props.maxThreadsDim[1],props.maxThreadsDim[2]);
        printf("Max grid size :\n 1 : %7d\n2 : %7d\n3 : %7d\n",props.maxGridSize[0],props.maxGridSize[1],props.maxGridSize[2]);
        printf("\n\n\n");
    }   
}


__global__ void imageComputeLab3(uchar3 *devImage, uchar3 *devOutputImage){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
            devOutputImage[tid].x = (char) (((int) devImage[tid].x + (int) devImage[tid].y +
                                          (int) devImage[tid].z) / 3);
            devOutputImage[tid].y = devOutputImage[tid].x;
            devOutputImage[tid].z = devOutputImage[tid].x;
}

void Labwork::labwork3_GPU() {
            uchar3 *devImage;
            uchar3 *devOutputImage;
            uchar3 *hostOutputImage;
            int pixelCount =inputImage->width *inputImage->height;
            int blockSize = 1024;
            int numBlock = pixelCount / blockSize;

            cudaMalloc(&devImage, pixelCount * 3);
            cudaMalloc(&devOutputImage, pixelCount * 3);
            hostOutputImage = (uchar3 *) malloc(pixelCount * 3);

            cudaMemcpy(devImage, inputImage->buffer,pixelCount * sizeof(uchar3),cudaMemcpyHostToDevice); // Memory transfert

            imageComputeLab3<<<numBlock, blockSize>>>(devImage,devOutputImage); // Kernel

            cudaMemcpy(hostOutputImage, devOutputImage,pixelCount * sizeof(uchar3),cudaMemcpyDeviceToHost);
	    outputImage = (char *)hostOutputImage;
            cudaFree(devImage);   
	    cudaFree(devOutputImage);
}

__global__ void imageComputeLab4(uchar3 *devImage, uchar3 *devOutputImage,int width){
            int x = threadIdx.x + blockIdx.x * blockDim.x;
	    int y = threadIdx.y + blockIdx.y * blockDim.y;
	    int tid = y * width +x;
            devOutputImage[tid].x = (char) ((int) (ceil((float) devImage[tid].x) + (int) ceil((float) devImage[tid].y) +
                                         (int) ceil((float) devImage[tid].z)) / 3);
            devOutputImage[tid].y = devOutputImage[tid].x;
            devOutputImage[tid].z = devOutputImage[tid].x;
}

void Labwork::labwork4_GPU() {
            uchar3 *devImage;
            uchar3 *devOutputImage;
            uchar3 *hostOutputImage;
            dim3 blockSize = dim3(32,32);
            int pixelCount =inputImage->width *inputImage->height;	   
	    int width = inputImage->width;
	    dim3 gridSize = dim3(inputImage->width/blockSize.x,inputImage->height/blockSize.y);

            cudaMalloc(&devImage, pixelCount * 3);
            cudaMalloc(&devOutputImage, pixelCount * 3);
            hostOutputImage = (uchar3 *) malloc(pixelCount * 3);

            cudaMemcpy(devImage, inputImage->buffer,pixelCount * sizeof(uchar3),cudaMemcpyHostToDevice); // Memory transfert

            imageComputeLab4<<<gridSize, blockSize>>>(devImage,devOutputImage,width); // Kernel
	
            cudaMemcpy(hostOutputImage, devOutputImage,pixelCount * sizeof(uchar3),cudaMemcpyDeviceToHost);
            outputImage = (char *)hostOutputImage;
            cudaFree(devImage);   
	    cudaFree(devOutputImage);
}



// CPU implementation of Gaussian Blur
void Labwork::labwork5_CPU() {
    int kernel[] = { 0, 0, 1, 2, 1, 0, 0,
                     0, 3, 13, 22, 13, 3, 0,
                     1, 13, 59, 97, 59, 13, 1,
                     2, 22, 97, 159, 97, 22, 2,
                     1, 13, 59, 97, 59, 13, 1,
                     0, 3, 13, 22, 13, 3, 0,
                     0, 0, 1, 2, 1, 0, 0 };
    int pixelCount = inputImage->width * inputImage->height;
    outputImage = (char*) malloc(pixelCount * sizeof(char) * 3);
    for (int row = 0; row < inputImage->height; row++) {
        for (int col = 0; col < inputImage->width; col++) {
            int sum = 0;
            int c = 0;
            for (int y = -3; y <= 3; y++) {
                for (int x = -3; x <= 3; x++) {
                    int i = col + x;
                    int j = row + y;
                    if (i < 0) continue;
                    if (i >= inputImage->width) continue;
                    if (j < 0) continue;
                    if (j >= inputImage->height) continue;
                    int tid = j * inputImage->width + i;
                    unsigned char gray = (inputImage->buffer[tid * 3] + inputImage->buffer[tid * 3 + 1] + inputImage->buffer[tid * 3 + 2])/3;
                    int coefficient = kernel[(y+3) * 7 + x + 3];
                    sum = sum + gray * coefficient;
                    c += coefficient;
                }
            }
            sum /= c;
            int posOut = row * inputImage->width + col;
            outputImage[posOut * 3] = outputImage[posOut * 3 + 1] = outputImage[posOut * 3 + 2] = sum;
        }
    }
}


__global__ void filterLab5(uchar3 *filterDevOutputImage, uchar3 *devOutputImage,int width, int height){
            int x = threadIdx.x + blockIdx.x * blockDim.x;
            int y = threadIdx.y + blockIdx.y * blockDim.y;
            int tid = y * width +x;
	    int filter[] = { 0, 0, 1, 2, 1, 0, 0,
                     0, 3, 13, 22, 13, 3, 0,
                     1, 13, 59, 97, 59, 13, 1,
                     2, 22, 97, 159, 97, 22, 2,
                     1, 13, 59, 97, 59, 13, 1,
                     0, 3, 13, 22, 13, 3, 0,
                     0, 0, 1, 2, 1, 0, 0 };


	    float outputPixel = 0;
	    for (int i = -3; i <= 3; i++)
	    {
		for (int j = -3; j <= 3; j++)
		{
		    if ( (x + i) < 0)	// left side
			continue;
		    if ( (x + i) >= width )	// right side
		        continue;
		    if ((y + j) < 0)	// top side
			continue;
		    if ((y + j) >= height )	// bottom side
			continue;
		    int localtid = (x+i)+ (y+j)*width;
		    unsigned char grey = (devOutputImage[localtid].x + devOutputImage[localtid].y + devOutputImage[localtid].z)/3;
                    int coefficient = filter[(j+3) * 7 + i + 3];
		    outputPixel += coefficient * grey;
		   
	        }
	    }
            filterDevOutputImage[tid].x = outputPixel/1003;
	    filterDevOutputImage[tid].y = filterDevOutputImage[tid].z = filterDevOutputImage[tid].x;
}

__global__ void greyScalingLab5(uchar3 *devImage, uchar3 *devOutputImage, int width){
            int x = threadIdx.x + blockIdx.x * blockDim.x;
            int y = threadIdx.y + blockIdx.y * blockDim.y;
            int tid = y * width + x;

            devOutputImage[tid].x = (char) ((int) (ceil((float) devImage[tid].x) + (int) ceil((float) devImage[tid].y) +
                                         (int) ceil((float) devImage[tid].z)) / 3);
            devOutputImage[tid].y = devOutputImage[tid].x;
            devOutputImage[tid].z = devOutputImage[tid].x;

}


void Labwork::labwork5_GPU() {
/*
	    float GaussianFilter[7][7] ={
		{1,4,7,10,7,4,1},
		{4,12,26,33,26,12,4},
		{7,26,55,71,55,26,7},
		{10,33,71,91,71,33,10},
	        {7,26,55,71,55,26,7},
		{4,12,26,33,26,12,4},
		{1,4,7,10,7,4,1},
	    }; // Sum equal to 1115
*/
/*	    int filter[] = { 0, 0, 1, 2, 1, 0, 0,
                     0, 3, 13, 22, 13, 3, 0,
                     1, 13, 59, 97, 59, 13, 1,
                     2, 22, 97, 159, 97, 22, 2,
                     1, 13, 59, 97, 59, 13, 1,
                     0, 3, 13, 22, 13, 3, 0,
                     0, 0, 1, 2, 1, 0, 0 };
 */     
	    uchar3 *devImage;
	    uchar3 *devOutputImage;
            uchar3 *hostOutputImageFilter;
	    uchar3 *filterDevOutputImage;

            dim3 blockSize = dim3(256,256);
            int pixelCount =inputImage->width *inputImage->height;
            int width = inputImage->width;
	    int height = inputImage->height;
            dim3 gridSize = dim3(inputImage->width/blockSize.x,inputImage->height/blockSize.y);

	    hostOutputImageFilter = (uchar3 *) malloc(pixelCount*3);
            cudaMalloc(&devImage, pixelCount * 3);
            cudaMalloc(&devOutputImage, pixelCount*3);      
	    cudaMalloc(&filterDevOutputImage, pixelCount*3);

            cudaMemcpy(devImage, inputImage->buffer,pixelCount * sizeof(uchar3),cudaMemcpyHostToDevice); // Memory transfert

            greyScalingLab5<<<gridSize, blockSize>>>(devImage,devOutputImage,width); // Kernel greyscaling

            filterLab5<<<gridSize, blockSize>>>(filterDevOutputImage,devOutputImage,width, height); // Kernel
	    
            cudaMemcpy(hostOutputImageFilter, filterDevOutputImage, pixelCount*3, cudaMemcpyDeviceToHost);

	    outputImage = (char *)hostOutputImageFilter;
            cudaFree(devImage);
            cudaFree(devOutputImage);
            cudaFree(filterDevOutputImage);

}




__global__ void binarisationLab6(uchar3 *binarisationDevOutputImage, uchar3 *devOutputImage,int width){
            int x = threadIdx.x + blockIdx.x * blockDim.x;
            int y = threadIdx.y + blockIdx.y * blockDim.y;
            int tid = y * width + x;
	    int therehold = 256/2;
	    unsigned char grey = devOutputImage[tid].x;

	    if(grey<therehold){
		binarisationDevOutputImage[tid].x = 0;
	    }
	    else{
		binarisationDevOutputImage[tid].x = 255;
	    }

	    binarisationDevOutputImage[tid].y = binarisationDevOutputImage[tid].z = binarisationDevOutputImage[tid].x;
	    
}

__global__ void brightnessLab6(uchar3 *binarisationDevOutputImage, uchar3 *devOutputImage,int width){
            int x = threadIdx.x + blockIdx.x * blockDim.x;
            int y = threadIdx.y + blockIdx.y * blockDim.y;
            int tid = y * width + x;
            unsigned char grey = devOutputImage[tid].x;

            binarisationDevOutputImage[tid].x = grey + (int) 0.2*grey;
            binarisationDevOutputImage[tid].y = binarisationDevOutputImage[tid].z = binarisationDevOutputImage[tid].x;

}



__global__ void greyScalingLab6(uchar3 *devImage, uchar3 *devOutputImage, int width){
            int x = threadIdx.x + blockIdx.x * blockDim.x;
            int y = threadIdx.y + blockIdx.y * blockDim.y;
            int tid = y * width + x;

            devOutputImage[tid].x = (char) ((int) (ceil((float) devImage[tid].x) + (int) ceil((float) devImage[tid].y) +
                                         (int) ceil((float) devImage[tid].z)) / 3);
            devOutputImage[tid].y = devOutputImage[tid].x;
            devOutputImage[tid].z = devOutputImage[tid].x;
}


void Labwork::labwork6_GPU() {
            uchar3 *devImage;
            uchar3 *devOutputImage;
            uchar3 *hostOutputImageFilter;
            uchar3 *binarisationDevOutputImage;

            dim3 blockSize = dim3(32,32);
            int pixelCount =inputImage->width *inputImage->height;
            int width = inputImage->width;
            dim3 gridSize = dim3(inputImage->width/blockSize.x,inputImage->height/blockSize.y);

            hostOutputImageFilter = (uchar3 *) malloc(pixelCount*3);
            cudaMalloc(&devImage, pixelCount * 3);
            cudaMalloc(&devOutputImage, pixelCount*3);
            cudaMalloc(&binarisationDevOutputImage, pixelCount*3);


            cudaMemcpy(devImage, inputImage->buffer,pixelCount * sizeof(uchar3),cudaMemcpyHostToDevice); // Memory transfert

            greyScalingLab6<<<gridSize, blockSize>>>(devImage,devOutputImage,width); // Kernel greyscaling

            binarisationLab6<<<gridSize, blockSize>>>(binarisationDevOutputImage,devOutputImage,width); // Kernel
	    
	    //brightnessLab6<<<gridSize, blockSize>>>(binarisationDevOutputImage,devOutputImage,width); // Kernel
            
	    cudaMemcpy(hostOutputImageFilter, binarisationDevOutputImage, pixelCount*3, cudaMemcpyDeviceToHost);

            outputImage = (char *)hostOutputImageFilter;
            cudaFree(devImage);
            cudaFree(devOutputImage);
            cudaFree(binarisationDevOutputImage);

}






void Labwork::labwork7_GPU() {

}

void Labwork::labwork8_GPU() {

}

void Labwork::labwork9_GPU() {

}

void Labwork::labwork10_GPU() {

}
