#include <math.h>
#include <stdio.h>
#include "num.h"

/*** INTERPOLATION METHODS ***/

void linear_itp(long double out[], long double xs[], long double ys[], int n)
{
    int i;
    for (i = 0; i < n; i++) {
        out[i] = 0.5 * (ys[i+1] + ys[i]);
    }
}

void cubic_itp(long double out[], long double xs[], long double ys[], int n)
{
    int i;
    for (i = 0; i < n; i++) {
        // Forward quadratic
        if (i == 0)
            out[i] = (3.0 * ys[i] + 6.0 * ys[i+1] - ys[i+2]) / 8.0;

        // Backward quadratic
        else if (i == n - 1)
            out[i] = (-ys[i-1] + 6.0 * ys[i] + 3.0 * ys[i+1]) / 8.0;

        // Full cubic
        else
            out[i] = (9.0 * (ys[i+1] + ys[i]) - (ys[i+2] + ys[i-1])) / 16.0;
    }
}

void spline_itp(long double out[], long double xs[], long double ys[], int n) 
{
    int i;

    // step zero
    long double a[n+1], b[n+1], c[n+1], d[n+1];
    for (i = 0; i < n + 1; i++)
        a[i] = ys[i];

    // step one
    long double h[n];
    for (i = 0; i < n; i++)
        h[i] = xs[i+1] - xs[i];

    // step two
    long double alpha[n];
    for (i = 1; i < n; i++)
        alpha[i] = 3 * (ys[i+1] - ys[i]) / h[i] - 3 * (ys[i] - ys[i-1]) / h[i-1];
        
    // step three
    long double l[n+1], mu[n+1], z[n+1];
    l[0] = 1;
    mu[0] = 0;
    z[0] = 0;

    // step four
    for (i = 1; i < n; i++) {
        l[i] = 2 * (xs[i+1] - xs[i-1]) - h[i-1] * mu[i-1];
        mu[i] = h[i] / l[i];
        z[i] = (alpha[i] - h[i-1] * z[i-1]) / l[i];
    }

    // step five
    l[n] = 1;
    z[n] = 0;
    c[n] = 0;

    // step six
    for (i = n - 1; i >= 0; i--) {
        c[i] = z[i] - mu[i] * c[i+1];
        b[i] = (a[i+1] - a[i]) / h[i] - h[i] * (c[i+1] + 2 * c[i]) / 3;
        d[i] = (c[i+1] - c[i]) / (3 * h[i]);
    }

    // now write the contents of `out`
    long double x;
    for (i = 0; i < n; i++) {
        x = 0.5 * (xs[i+1] - xs[i]); // midpoint of bin i - x[i]
        out[i] = a[i] + b[i] * x + c[i] * x*x + d[i] * x*x*x;
    }
}


/*** DIFFERENTIATION METHODS ***/

void linear_diff(long double out[], long double xs[], long double ys[], int n)
{
    int i;
    for (i = 0; i < n; i++)
        out[i] = (ys[i+1] - ys[i]) / (xs[i+1] - xs[i]);
}

void cubic_diff(long double out[], long double xs[], long double ys[], int n)
{
    int i;
    for (i = 0; i < n; i++) {
        if (i == 0 || i == n - 1)
            out[i] = (ys[i+1] - ys[i]) / (xs[i+1] - xs[i]);
        else
            out[i] = (27.0 * (ys[i+1] - ys[i]) - (ys[i+2] - ys[i-1])) 
                     / (24.0 * (xs[i+1] - xs[i]));
    }
}

void spline_diff(long double out[], long double xs[], long double ys[], int n)
{
    int i;

    // step zero
    long double a[n+1], b[n+1], c[n+1], d[n+1];
    for (i = 0; i < n + 1; i++)
        a[i] = ys[i];

    // step one
    long double h[n];
    for (i = 0; i < n; i++)
        h[i] = xs[i+1] - xs[i];

    // step two
    long double alpha[n];
    for (i = 1; i < n; i++)
        alpha[i] = 3 * (ys[i+1] - ys[i]) / h[i] - 3 * (ys[i] - ys[i-1]) / h[i-1];
        
    // step three
    long double l[n+1], mu[n+1], z[n+1];
    l[0] = 1;
    mu[0] = 0;
    z[0] = 0;

    // step four
    for (i = 1; i < n; i++) {
        l[i] = 2 * (xs[i+1] - xs[i-1]) - h[i-1] * mu[i-1];
        mu[i] = h[i] / l[i];
        z[i] = (alpha[i] - h[i-1] * z[i-1]) / l[i];
    }

    // step five
    l[n] = 1;
    z[n] = 0;
    c[n] = 0;

    // step six
    for (i = n - 1; i >= 0; i--) {
        c[i] = z[i] - mu[i] * c[i+1];
        b[i] = (a[i+1] - a[i]) / h[i] - h[i] * (c[i+1] + 2 * c[i]) / 3;
        d[i] = (c[i+1] - c[i]) / (3 * h[i]);
    }

    // now write the contents of `out`
    long double x;
    for (i = 0; i < n; i++) {
        x = 0.5 * (xs[i+1] - xs[i]); // midpoint of bin i - x[i]
        out[i] = b[i] + 2.0 * c[i] * x + 3.0 * d[i] * x*x;
    }
}


/*** INTEGRATION METHODS ***/

void euler(
    long double yout[],
    long double xs[], long double ys[], int n,
    long double dt,
    struct Parameters p,
    void (*rhs)(long double[], long double[], long double[], int, struct Parameters)
) {
    int i;
    long double dydt[n];
    (*rhs)(dydt, xs, ys, n, p);
    for (i = 0; i < n; i++)
        yout[i] = ys[i] + dt * dydt[i];
}

void rk2(
    long double yout[],
    long double xs[], long double ys[], int n,
    long double dt,
    struct Parameters p,
    void (*rhs)(long double[], long double[], long double[], int, struct Parameters)
) {
    int i;
    long double dydt[n], k1[n], k2[n], yt[n];

    // Compute k1
    (*rhs)(dydt, xs, ys, n, p);
    for (i = 0; i < n; i++)
        k1[i] = dt * dydt[i];

    // Compute k2
    for (i = 0; i < n; i++)
        yt[i] = ys[i] + 0.5 * k1[i];
    (*rhs)(dydt, xs, yt, n, p);
    for (i = 0; i < n; i++)
        k2[i] = dt * dydt[i];

    // Compute the updated y values
    for (i = 0; i < n; i++)
        yout[i] = ys[i] + k2[i];
}

void rk4(
    long double yout[],
    long double xs[], long double ys[], int n,
    long double dt,
    struct Parameters p,
    void (*rhs)(long double[], long double[], long double[], int, struct Parameters)
) {
    int i;
    long double dydt[n], k1[n], k2[n], k3[n], k4[n], yt[n];

    // Compute k1
    (*rhs)(dydt, xs, ys, n, p);
    for (i = 0; i < n; i++)
        k1[i] = dt * dydt[i];

    // Compute k2
    for (i = 0; i < n; i++)
        yt[i] = ys[i] + 0.5 * k1[i];
    (*rhs)(dydt, xs, yt, n, p);
    for (i = 0; i < n; i++)
        k2[i] = dt * dydt[i];

    // Compute k3
    for (i = 0; i < n; i++)
        yt[i] = ys[i] + 0.5 * k2[i];
    (*rhs)(dydt, xs, yt, n, p);
    for (i = 0; i < n; i++)
        k3[i] = dt * dydt[i];

    // Compute k4
    for (i = 0; i < n; i++)
        yt[i] = ys[i] + k3[i];
    (*rhs)(dydt, xs, yt, n, p);
    for (i = 0; i < n; i++)
        k4[i] = dt * dydt[i];

    // Compute the updated y values
    long double one_sixth = 1.0 / 6.0;
    for (i = 0; i < n; i++)
        yout[i] = ys[i] + one_sixth * (k1[i] + 2 * (k2[i] + k3[i]) + k4[i]);
}

