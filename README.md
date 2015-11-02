PB:


x2b-v6x-direct.cpp  contains the explicit 2B water-water polynomials as in MBpol, evaluated directly (not using the maple generated code, without gradients).
I used the maple code and manually converted the polynomial terms to cpp format.

The maple generated code without gradients can be found here /home/pbajaj/codes/mb-md/fitting/src/potential/poly-2b-v6x-nogrd.cpp for comparison.

```
kernels/2b_direct_polynomial.cu
Lines 1191
Characters 964997
+ 52569
* 107433
= 1186
pow 5575

kernels/2b_maple_polynomial.cu
Lines 13836
Characters 440555
+ 19894
* 10707
= 12361
pow 0
```
