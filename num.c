#include <math.h>
#include <stdio.h>

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


/**
 * Returns the value of the Lagrange polynomial, defined as follows.
 * L_{n,k} := \prod_{j = 0, j \neq k}^{n} (x - x_j) / (x_k - x_j)
 * n, k, and x match the arguments to the function. x_j = xs[j].
 * Note that xs must be of length n + 1.
 */
long double lagrange_poly(int k, int n, long double x, long double xs[])
{
    int j;
    long double ell = 1.0;
    for (j = 0; j <= n; j++) { // includes n
        if (j == k) continue;
        ell *= (x - xs[j]) / (xs[k] - xs[j]);
    }
    return ell;
}

void quad_itp(long double out[], long double xs[], long double ys[], int n) 
{
    // For each interval, use Lagrange interpolation to fit two quadratics: one
    // forward and one backward. i.e. for the interval specified by x[i] and
    // x[i+1], fit a quadratic using x[i-1], x[i], x[i+1] and one using x[i],
    // x[i+1], and x[i+2]. Average the results. (Use only forward for i = 0,
    // only backward for i = n-1.)

    int i, j, k;
    long double x_int[3], y_int[3], x, ell;

    for (i = 0; i < n; i++) {
        // Get midpoint of interval
        x = 0.5 * (xs[i] + xs[i+1]);

        // Get forward interval prediction
        long double forward = 0;
        if (i != n - 1) {
            for (j = 0; j < 3; j++) {
                x_int[j] = xs[i+j];
                y_int[j] = ys[i+j];
            }
            for (j = 0; j < 3; j++) {
                ell = lagrange_poly(j, 2, x, x_int);
                forward += y_int[j] * ell;
            }
        }
        
        // Get backward interval prediction
        long double backward = 0;
        if (i != 0) {
            for (j = 0; j < 3; j++) {
                x_int[j] = xs[i-1+j];
                y_int[j] = ys[i-1+j];
            }
            for (j = 0; j < 3; j++) {
                ell = lagrange_poly(j, 2, x, x_int);
                backward += y_int[j] * ell;
            }
        }

        if (i == 0)
            out[i] = forward;
        else if (i == n - 1)
            out[i] = backward;
        else 
            out[i] = 0.5 * (forward + backward);
    }

}

void four_point_itp(long double out[], long double xs[], long double ys[], int n) 
{
    int i;
    long double one_16th = 1.0 / 16.0;
    long double lo, hi;
    for (i = 0; i < n; i++) {
        // clamped padding
        lo = (i == 0)   ? ys[0] : ys[i-1];
        hi = (i == n-1) ? ys[n] : ys[i+2];
        out[i] = (9.0 * (ys[i+1] + ys[i]) - (hi + lo)) * one_16th;
    }
}

/*** DIFFERENTIATION METHODS ***/

void linear_diff(long double out[], long double xs[], long double ys[], int n)
{
    int i;
    for (i = 0; i < n; i++)
        out[i] = (ys[i+1] - ys[i]) / (xs[i+1] - xs[i]);
}

void four_point_diff(long double out[], long double xs[], long double ys[], int n)
{
    int i;
    long double one_48th = 1.0 / 48.0;
    long double lo, hi;
    for (i = 0; i < n; i++) {
        // clamped padding
        lo = (i == 0)   ? ys[0] : ys[i-1];
        hi = (i == n-1) ? ys[n] : ys[i+2];
        out[i] = (27.0 * (ys[i+1] - ys[i]) - (hi - lo)) * one_48th * 2.0 / (xs[i+1] - xs[i]);
    }
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

/*
void euler(long double y[], long double yout[], int n, long double dt,
           void (*rhs)(long double[]))
{
    int i;
    long double dydt[n];
    (*rhs)(dydt);
    for (i = 0; i < n; i++)
        yout[i] = y[i] + dt * dydt[i];
}

void euler_recurse(long double y[], long double yout[], int n, long double
                   dt, void (*rhs)(long double[]))
{
    er(y, yout, n, dt, rhs, 0);
}

void er(long double y[], long double yout[], int n, long double dt,
        void (*rhs)(long double[]), int depth)
{
    int i, ok = 0;

    // If at max recursion depth, just run Euler without running any deeper
    // recursion (even if it happens to end up negative)
    int MAX_DEPTH = 5;
    if (depth >= MAX_DEPTH) {
        euler(y, yout, n, dt, rhs);
        return;
    }
    
    // Run Euler with a test yout array so we don't potentially overwrite y
    // values
    long double test_yout[n];
    euler(y, test_yout, n, dt, rhs);

    // Check the test yout for negative values
    for (i = 0; i < n; i++) {
        if (test_yout[i] < 0) {
            ok = 1;
            break;
        }
    }

    // If there are no negatives, we are good
    if (ok == 0) {
        for (i = 0; i < n; i++)
            yout[i] = test_yout[i];
        return;
    }
    // If there are negatives, try to instead do two steps of half size
    else {
        er(y, test_yout, n, 0.5 * dt, rhs, depth + 1);
        er(test_yout, yout, n, 0.5 * dt, rhs, depth + 1);
    }
}
*/
