#pragma once
#include <omp.h>
#include <stdio.h>

void matrixMultiplication_sequential(float* A, float* B, float* C, int M, int N, int K) {
    double begin, end;

    begin = omp_get_wtime();
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            for (int k = 0; k < N; k++) {
                C[i * K + j] += A[i * N + k] * B[k * K + j];
            }
        }
    }
    end = omp_get_wtime();
    printf("CPU time: %f\n", end - begin);
}