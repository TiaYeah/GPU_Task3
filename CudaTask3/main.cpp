#include <cstdio>
#include <iostream>
#include <vector>
#include <cassert>

#include "omp.h"
#include "gpu.h"
#include "sequential.h"

using namespace std;

bool compareArrays(float* C, float* res, int size) {
    for (int i = 0; i < size; i++) {
        if (C[i] != res[i]) {
            return false;
        }
    }
    return true;
}

void printMatrix(float* A, int x, int y) {
    for (int i = 0; i < y; i++) {
        for (int j = 0; j < x; j++) {
            cout << A[i * x + j] << " ";
        }
        cout << endl;
    }
}

int main() {
    int M = 1024;
    int N = 512;
    int K = 1024;
    float* A, * B, * C, * res;
    A = new float[M * N];
    B = new float[N * K];
    C = new float[M * K];
    res = new float[M * K];

    for (int i = 0; i < M * K; i++) {
        C[i] = 0.0f;
        if (i % 2 == 0) {
            res[i] = (float) 3 * N;
        }
        else {
            res[i] = (float) 1.5 * N;
        }
    }
    for (int i = 0; i < M * N; i++) {
        A[i] = 1.0f;
    }
    for (int i = 0; i < M * N; i++) {
        if (i % 2 == 0) {
            B[i] = 3.0f;
        }
        else {
            B[i] = 1.5f;
        }
    }

    matrixMultiplication_sequential(A, B, C, M, N, K);
    assert(compareArrays(C, res, M * K));

    delete[] C;
    C = new float[M * K];
    for (int i = 0; i < M * K; i++) {
        C[i] = 0.0f;
    }

    matrixMultiplication_omp(A, B, C, M, N, K);
    assert(compareArrays(C, res, M * K));

    delete[] C;
    C = new float[M * K];
    for (int i = 0; i < M * K; i++) {
        C[i] = 0.0f;
    }

    matrixMultiplication_gpu(A, B, C, M, N, K);
    assert(compareArrays(C, res, M * K));

    delete[] C;
    C = new float[M * K];
    for (int i = 0; i < M * K; i++) {
        C[i] = 0.0f;
    }

    matrixMultiplication_gpu_optimized(A, B, C, M, N, K);
    assert(compareArrays(C, res, M * K));


    return 0;
}