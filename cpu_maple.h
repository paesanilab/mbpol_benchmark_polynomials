#ifndef POLY_2B_V6X_H
#define POLY_2B_V6X_H


//
// this is the polynomial used by x2b_v6<4> (including gradients)
//

double eval_cpu_maple(const double a[1153],
                         const double x[31],
                               double g[31]);

#endif // POLY_2B_V6X_H
