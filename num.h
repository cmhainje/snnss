#ifndef CH_NUM
#define CH_NUM

#include <stdbool.h>

/**
 * Provides a clean structure for passing parameters to methods.
 */
struct Parameters {
    long double kT;
    long double rho_N;
    long double Y_e;
    int n_type;
    void (*interp_f)(long double[], long double[], long double[], int);
    void (*deriv_f)(long double[], long double[], long double[], int);
    void (*deriv_I)(long double[], long double[], long double[], int);
    bool force_pos;
};

/**
 * INTERPOLATION METHODS
 * 
 * Given two arrays xs, representing sampling points, and ys, representing the
 * sampled values of some function f(x) at each point x, these methods
 * interpolate the function f(x) and return the interpolated value of f at the
 * (arithmetic) midpoint of each interval in x.
 * i.e. given x values x_0, x_1, ..., x_N and y values y_0, y_1, ..., y_N, these
 * methods will return f_i \approx f((x_0 + x_1) / 2) for i in [0, N-1].
 * 
 * API:
 * @param out   output array, length n
 * @param xs    sampling points, length n + 1
 * @param ys    sampled values,  length n + 1
 * @param n     number of intervals
 */

/**
 * Linearly interpolates to the midpoint of each bin.
 */
void linear_itp(long double out[], long double xs[], long double ys[], int n);

/**
 * For each interval, performs cubic Lagrange interpolation on the four points
 * surrounding the interval midpoint. 
 * 
 * Note: the first and last bins instead perform quadratic Lagrange
 * interpolation, taking the two endpoints of the interval and one additional
 * point either forward or backward from the interval.
 */
void cubic_itp(long double out[], long double xs[], long double ys[], int n);

/**
 * Uses cubic spline interpolation to interpolate to the midpoint of each bin.
 */
void spline_itp(long double out[], long double xs[], long double ys[], int n);


/**
 * DIFFERENTIATION METHODS
 * 
 * Given two arrays xs, representing sampling points, and ys, representing the
 * sampled values of some function f(x) at each point x, these methods
 * differentiate the function f(x) and return the interpolated values of df/dx
 * at the (arithmetic) midpoint of each interval in x.
 * i.e. given x values x_0, x_1, ..., x_N and y values y_0, y_1, ..., y_N, these
 * methods will return df_i \approx df/dx|_{x=(x_0 + x_1) / 2} for i in [0, N-1].
 * 
 * API:
 * @param out   output array, length n
 * @param xs    sampling points, length n + 1
 * @param ys    sampled values,  length n + 1
 * @param n     number of intervals
 */

/**
 * Simple two-point centered finite differencing.
 */
void linear_diff(long double out[], long double xs[], long double ys[], int n);

/**
 * For each interval, performs cubic Lagrange interpolation on the four points
 * surrounding the interval midpoint and returns the derivative of the
 * interpolant at the center.
 * 
 * For the first and last intervals, returns only a linear approximation of the
 * derivative (instead of employing padding). (This is equivalent to the method
 * used in cubic_itp where we perform quadratic Lagrange interpolation and
 * return the derivative of the interpolant. Interestingly, the third point
 * drops out of the derivative when on the midpoint of the interval.)
 */
void cubic_diff(long double out[], long double xs[], long double ys[], int n);

/**
 * Computes the derivative of a cubic spline interpolator at the bin midpoints.
 */
void spline_diff(long double out[], long double xs[], long double ys[], int n);


/**
 * INTEGRATION METHODS
 * 
 * API:
 * @param yout  updated values of y
 * @param xs    dimensionless energy zone centers
 * @param ys    input values of y = x^3 J
 * @param n     length of xs, ys, yout arrays
 * @param dt    time update
 * @param p     parameters
 * @param rhs   computes the RHS of the ODE
 */


/**
 * Implements an Euler step.
 */
void euler(
    long double yout[],
    long double xs[], long double ys[], int n,
    long double dt,
    struct Parameters p,
    void (*rhs)(long double[], long double[], long double[], int, struct Parameters)
);

/**
 * Implements a step via the second-order Runge-Kutta method.
 */
void rk2(
    long double yout[],
    long double xs[], long double ys[], int n,
    long double dt,
    struct Parameters p,
    void (*rhs)(long double[], long double[], long double[], int, struct Parameters)
);

/**
 * Implements a step via the fourth-order Runge-Kutta method.
 */
void rk4(
    long double yout[],
    long double xs[], long double ys[], int n,
    long double dt,
    struct Parameters p,
    void (*rhs)(long double[], long double[], long double[], int, struct Parameters)
);

#endif