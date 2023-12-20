#include "cuda_runtime.h"
#include <cuda.h>
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <omp.h>

#include "gpu.h"

__global__ void matrixMultiplication_kernel(float* A, float* B, float* C, int M, int N, int K) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;


    if (row < M && col < K) {
        for (int i = 0; i < N; i++) {
            C[row * K + col] += A[row * N + i] * B[i * K + col];
        }
    }

}
__global__ void matrixMultiplication_optimized_kernel(float* A, float* B, float* C, int M, int N, int K) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    int BLOCK_SIZE = blockDim.x;

    float res = 0;
    __shared__ float A_block[16 * 16];
    __shared__ float B_block[16 * 16];

    if (row < M && col < K) {
        for (int i = 0; i < N; i += BLOCK_SIZE) {
            A_block[threadIdx.y * BLOCK_SIZE + threadIdx.x] = A[row * N + (i + threadIdx.x)];
            B_block[threadIdx.y * BLOCK_SIZE + threadIdx.x] = B[(i + threadIdx.y) * K + col];
            __syncthreads();
            for (int j = 0; j < BLOCK_SIZE; j++) {
                res += A_block[threadIdx.y * BLOCK_SIZE + j] * B_block[j * BLOCK_SIZE + threadIdx.x];
            }
            __syncthreads();
        }
        C[row * K + col] += res;
    }
}



void matrixMultiplication_gpu(float* A, float* B, float* C, int M, int N, int K) {
    float* A_gpu;
    float* B_gpu;
    float* C_gpu;
    cudaError_t cudaStatus;


    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .

    cudaStatus = cudaMalloc((void**)&A_gpu, M * N * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&B_gpu, N * K * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&C_gpu, M * K * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(A_gpu, A, M * N * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(B_gpu, B, N * K * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(C_gpu, C, M * K * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    double begin, end;

    dim3 blockSize(16, 16);
    dim3 gridSize((M + blockSize.x - 1) / blockSize.x, (K + blockSize.y - 1) / blockSize.y);
    begin = omp_get_wtime();
    matrixMultiplication_kernel << <gridSize, blockSize >> > (A_gpu, B_gpu, C_gpu, M, N, K);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "matrixMultiplication_gpu launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    end = omp_get_wtime();

    printf("GPU time: %f\n", end - begin);

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching saxpy_gpu!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(C, C_gpu, M * K * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(A_gpu);
    cudaFree(B_gpu);
    cudaFree(C_gpu);

}

void matrixMultiplication_gpu_optimized(float* A, float* B, float* C, int M, int N, int K) {
    float* A_gpu;
    float* B_gpu;
    float* C_gpu;
    cudaError_t cudaStatus;


    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .

    cudaStatus = cudaMalloc((void**)&A_gpu, M * N * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&B_gpu, N * K * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&C_gpu, M * K * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(A_gpu, A, M * N * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(B_gpu, B, N * K * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(C_gpu, C, M * K * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    double begin, end;

    dim3 blockSize(16, 16);
    dim3 gridSize((M + blockSize.x - 1) / blockSize.x, (K + blockSize.y - 1) / blockSize.y);
    begin = omp_get_wtime();
    matrixMultiplication_optimized_kernel << <gridSize, blockSize >> > (A_gpu, B_gpu, C_gpu, M, N, K);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "matrixMultiplication_gpu_optimized launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    end = omp_get_wtime();

    printf("GPU optimized time: %f\n", end - begin);

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching saxpy_gpu!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(C, C_gpu, M * K * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(A_gpu);
    cudaFree(B_gpu);
    cudaFree(C_gpu);

}