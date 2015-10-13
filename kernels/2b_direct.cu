#include <cuda.h>
#include <cuda_runtime_api.h>

__device__ void evaluate_2b_direct(double * a, double * x, double * g, double * energy_buffer) {
    double p[1153];

#ifndef DISABLEPOLYGPU

#include "2b_direct_polynomial.cu"

    energy_buffer[0] = energy;
#endif
}

__global__ void evaluate_2b_direct_many(double * a, double * x, double * g, double * energy_buffer, int n) {
    for (int i=0; i<n; i++) {
        evaluate_2b_direct(a, x, g, energy_buffer);
    }
}

void launch_evaluate_2b_direct(double * a, double * x, double * g, double * e) {
    evaluate_2b_direct_many<<<1,1>>>(a, x, g, e, 10000);
    cudaDeviceSynchronize();
}
