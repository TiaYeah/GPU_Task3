#pragma once

void matrixMultiplication_gpu(float* A, float* B, float* C, int M, int N, int K);
void matrixMultiplication_gpu_optimized(float* A, float* B, float* C, int M, int N, int K);