// #define DEBUG

__device__ void evaluate_2b_maple(
    double * a, double * x, double * g, double * energy_buffer
)
{

#include "2b_maple_polynomial.cu"

energy_buffer[0] = energy;
}


__global__ void evaluate_2b_maple_many(double * a, double * x, double * g, double * energy_buffer, int n) {
    for (int i=0; i<n; i++) {
        evaluate_2b_maple(a, x, g, energy_buffer);
    }
}

void launch_evaluate_2b_maple(double * a, double * x, double * g, double * e) {
    evaluate_2b_maple_many<<<1,1>>>(a, x, g, e, 10000);
    cudaDeviceSynchronize();
}
