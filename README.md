MBPol Polynomials benchmark
===========================

MBPol is a new force field for chemically accurate water simulations.
Paesani's group developed the theory and provided a C++ implementation as a [plugin for OpenMM](https://github.com/paesanilab/mbpol_openmm_plugin).

We are now working on the CUDA implementation, and we have performance issues.

## Algorithm description

The key element of the MBPol force field is the evaluation of a 4th order polynomial with thousands of terms for each pair of molecules. Production simulations molecules have generally 512 or more molecules, so within our cutoff we have ~30000 pairs.
Therefore there is no need to parallelize the polynomial evaluation, each of them should be executed by a CUDA thread.

## Complexity of the polynomials

Polynomials are just multiplications and sums, a polynomial is made up of 1153 terms, just to give a feeling, here is how a standard term looks like:

    p[70] = x[13]*x[20]*x[25]+x[12]*x[19]*x[26]+x[13]*x[20]*x[26]+x[12]*x[21]*x[26]+x[10]*x[15]*x[23]+x[11]*x[18]*x[23]+x[13]*x[22]*x[25]+x[11]*x[16]*x[23]+x[13]*x[22]*x[26]+x[10]*x[17]*x[24]+x[11]*x[18]*x[24]+x[10]*x[15]*x[24]+x[12]*x[21]*x[25]+x[11]*x[16]*x[24]+ x[12]*x[19]*x[25]+x[10]*x[17]*x[23];
    
* `x` is the input variable of size 31
* `p` is a temporary double array that holds the 1153 terms
* output `energy` is computed multiplying `p` by constant coefficients `a` (size 1153 as well)
* `g` are 31 gradients of the energy and are very complicated terms of `x` and `a`

We call this version of the polynomials, available in `cpu_direct.cpp`, "Direct".

First we made a rewrite of the polynomials to CUDA C, here is some statistics about the code:

```
kernels/2b_direct_polynomial.cu
Lines 1191
Characters 964997
+ 52569
* 107433
= 1186
pow 5575
```

## Automatic optimization with Maple

Polynomials terms have some repeated components, so we used the [`codegen` package from Maple](http://www.maplesoft.com/support/help/maple/view.aspx?path=codegen) to automatically factor out common terms and automatically generate a optimized C code.

"Maple" version of the polynomials for CPU is available in `cpu_maple.cpp`.

Here are some statistics about the Maple version of the code ported to CUDA:

```
kernels/2b_maple_polynomial.cu
Lines 13836
Characters 440555
+ 19894
* 10707
= 12361
pow 0
```
## Benchmark results

We have compared the CPU and CUDA implementations and notice that when we ran 10000 times on the same input, CPU is 45 times for Direct and 95 times for Maple faster than CUDA, with Intel(R) Xeon(R) CPU E5640 @ 2.67GHz vs Tesla k40.

See results below:

```
CPU Direct
 0.837584s wall, 0.830000s user + 0.000000s system = 0.830000s CPU (99.1%)

CPU Maple
 0.255172s wall, 0.260000s user + 0.000000s system = 0.260000s CPU (101.9%)

CUDA Direct
 36.322605s wall, 31.390000s user + 4.930000s system = 36.320000s CPU (100.0%)

CUDA Maple
 23.394607s wall, 20.230000s user + 3.160000s system = 23.390000s CPU (100.0%)
```

## Details about the test code

`run_test.cpp` code runs all 4 cases. Input variables:

* `double x[31]` variable will be different for every pair of molecules
* `const double a[1153]` this is always constant, we have not tried using constant memory in CUDA with this

The code uses `boost::timer` for benchmarking, goes through the 4 test cases and runs each 10000 times.

## Requirements

* CUDA
* `boost::timer` and `boost::system`
* `CMake`
* GCC

### How to build and run

* Create a `build` folder inside the source folder
* `cmake ..`
* `make`
* `./run_test`
