/**
 * main.c
 * Connor Hainje (cmhainje@gmail.com)
 * Provides all of the Kompaneets functionality in a single C90-compatible file
 */

#include "main.h"

/*** UTILITIES ***/

/**
 * Provides a clean structure for passing parameters to integration methods.
 * Not to be used externally.
 */
struct Parameters {
    long double kT;
    long double rho_N;
    long double Y_e;
    int n_type;
    void (*interp_f)(long double[], long double[], long double[], int);
    void (*deriv_f)(long double[], long double[], long double[], int);
    void (*deriv_I)(long double[], long double[], long double[], int);
    int force_pos;
};


/*** NUMERICAL METHODS ***/

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
void linear_itp(long double out[], long double xs[], long double ys[], int n)
{
    int i;
    for (i = 0; i < n; i++) {
        out[i] = 0.5 * (ys[i+1] + ys[i]);
    }
}

/**
 * For each interval, performs cubic Lagrange interpolation on the four points
 * surrounding the interval midpoint. 
 * 
 * Note: the first and last bins instead perform quadratic Lagrange
 * interpolation, taking the two endpoints of the interval and one additional
 * point either forward or backward from the interval.
 */
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


/**
 * Uses cubic spline interpolation to interpolate to the midpoint of each bin.
 */
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
void linear_diff(long double out[], long double xs[], long double ys[], int n)
{
    int i;
    for (i = 0; i < n; i++)
        out[i] = (ys[i+1] - ys[i]) / (xs[i+1] - xs[i]);
}


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


/**
 * Computes the derivative of a cubic spline interpolator at the bin midpoints.
 */
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
) {
    int i;
    long double dydt[n];
    (*rhs)(dydt, xs, ys, n, p);
    for (i = 0; i < n; i++)
        yout[i] = ys[i] + dt * dydt[i];
}

/**
 * Implements a step via the second-order Runge-Kutta method.
 */
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

/**
 * Implements a step via the fourth-order Runge-Kutta method.
 */
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



/*** KOMPANEETS METHODS ***/


/**
 * Computes the physical coefficient (referred to as `alpha` in my paper).
 *
 * @param kT    nucleon temperature [MeV]
 * @param rho_N nucleon mass density [g cm^-3]
 */
long double compute_coeff(long double kT, long double rho_N)
{
    long double beta = 1 / kT;
    long double nN = rho_N * 1e3 * powl(c,2) / (m * e_unit) * powl(l_unit,3);
    return 2 * G2 * nN / (3 * pi * powl(beta,3) * m);
}


/**
 * Computes I_nu on energy zone boundaries using the x^3 J scheme.
 * 
 * Note that this does not include the physical coefficient:
 *     (2 n_N G^2) / (3 pi beta^3 m)
 * 
 * @param I_nu      array to be filled [length: n+1]
 * @param xs        dimensionless energy zone centers [length: n]
 * @param x3J       sampled values of x^3 J(x) [length: n]
 * @param n         number of energy zones
 * @param beta      inverse nucleon temperature [MeV]
 * @param Y_e       electron fraction
 * @param n_type    nucleon type (0: proton, 1: neutron, else: both)
 * @param interp    interpolation function to use on f (= J)
 * @param deriv     differentiation function to use on f (= J)
 */
void compute_Inu(
    long double I_nu[],
    long double xs[], long double x3J[], int n,
    struct Parameters p
) {
    int i;

    long double beta = 1 / p.kT;
    long double Y_e = p.Y_e;
    int n_type = p.n_type;

    void (*interp)(long double[], long double[], long double[], int) = p.interp_f;
    void (*deriv)(long double[], long double[], long double[], int) = p.deriv_f;

    long double logx[n], logx3J[n];
    for (i = 0; i < n; i++) {
        logx[i] = logl(xs[i]);
        logx3J[i] = logl(x3J[i]);
    }

    // Compute x^3 J and d(x^3 J)/d(log x) on inner bin edges
    long double itp_x3J[n-1], dx3J_dlogx[n-1];

    if (p.force_pos) {
        (*interp)(itp_x3J, logx, logx3J, n-1);
        (*deriv)(dx3J_dlogx, logx, logx3J, n-1);
        for (i = 0; i < n - 1; i++) {
            itp_x3J[i] = expl(itp_x3J[i]);
            dx3J_dlogx[i] = dx3J_dlogx[i] * itp_x3J[i];
        }
    }
    else {
        (*interp)(itp_x3J, logx, x3J, n-1);
        (*deriv)(dx3J_dlogx, logx, x3J, n-1);
    }

    // Compute I_nu on bin edges
    I_nu[0] = 0;
    I_nu[n] = 0;
    for (i = 0; i < n - 1; i++) {
        long double x, lambda_p, lambda_n, coeff_p, coeff_n, phi;
        x = sqrtl(xs[i] * xs[i+1]);
        lambda_p = ((4*x-14)*powl(Vp,2) + (28*x-86)*powl(Ap,2)) / (prot * beta * m);
        lambda_n = ((4*x-14)*powl(Vn,2) + (28*x-86)*powl(An,2)) / (neut * beta * m);
        coeff_p = (n_type != 1) * prot * Y_e * (1 - lambda_p); 
        coeff_n = (n_type != 0) * neut * (1 - Y_e) * (1 - lambda_n);
        I_nu[i+1] = (coeff_p + coeff_n) * (
            powl(x, 2) * dx3J_dlogx[i] - 3 * powl(x, 2) * itp_x3J[i] 
            + powl(x, 3) * itp_x3J[i] - powl(itp_x3J[i], 2)
        );
    }
}

/**
 * Computes the right-hand side of the ODE on the energy zone centers 
 * using the x^3 scheme.
 * 
 * @param out       array to be filled [length: n]
 * @param xs        dimensionless energy zone centers [length: n]
 * @param x3J       sampled values of x^3 J(x) [length: n]
 * @param n         number of energy zones
 * @param kT        nucleon temperature [MeV]
 * @param rho_N     nucleon mass density [g/cm^3]
 * @param Y_e       electron fraction
 * @param n_type    nucleon type (0: proton, 1: neutron, else: both)
 * @param interp_f  interpolation function to use on f
 * @param deriv_f   differentiation function to use on f
 * @param deriv_I   differentiation function to use on I_nu
 */
void compute_rhs(
    long double out[],
    long double xs[], long double x3J[], int n,
    struct Parameters p
) {
    int i;

    long double kT = p.kT;
    long double rho_N = p.rho_N;
    long double Y_e = p.Y_e;
    int n_type = p.n_type;
    void (*interp_f)(long double[], long double[], long double[], int) = p.interp_f;
    void (*deriv_f)(long double[], long double[], long double[], int) = p.deriv_f;
    void (*deriv_I)(long double[], long double[], long double[], int) = p.deriv_I;

    // Compute I_nu on the bin edges
    long double I_nu[n+1];
    // compute_Inu(I_nu, xs, x3J, n, 1/kT, Y_e, n_type, interp_f, deriv_f);
    compute_Inu(I_nu, xs, x3J, n, p);

    // Compute d(I_nu)/d(log x) on bin centers and write to out
    long double log_edges[n+1];
    long double bin_w = logl(xs[1]) - logl(xs[0]);
    for (i = 0; i < n; i++)
        log_edges[i] = logl(xs[i]) - 0.5 * bin_w;
    log_edges[n] = logl(xs[n-1]) + 0.5 * bin_w;

    (*deriv_I)(out, log_edges, I_nu, n);

    long double coeff = compute_coeff(kT, rho_N);
    for (i = 0; i < n; i++)
        out[i] *= coeff;
}

/**
 * Dev version of `compute_step()`. Provides ability to directly
 * specify the interpolation and differentiation methods.
 * 
 * Physical parameters
 * @param kT        nucleon temperature [MeV]
 * @param rho_N     nucleon mass density [g/cm^3]
 * @param Y_e       electron fraction
 * @param n_type    nucleon type (0: proton, 1: neutron, else: both)
 * 
 * Input
 * @param energies  energy bins (zone *centers*) [MeV]
 * @param Js        distribution function
 * @param n         number of bins (length of energies and Js arrays)
 * @param dt        time step size [s]
 * 
 * Output
 * @param Jout      updated J values [length: n]
 * @param I_nu      I_nu on the bin edges [length: n+1]
 * @param qdot      energy deposition spectrum [length: n]
 * @param Qdot      energy deposition rate [length: n]
 * 
 * Numerical methods
 * @param interp_f  interpolation function to use on f
 * @param deriv_f   differentiation function to use on f
 * @param deriv_I   differentiation function to use on I_nu
 * @param stepper   stepping function to use (0: euler, 1: rk2, 2: rk4)
 */
void compute_step_dev(
    long double kT, long double rho_N, long double Y_e, int n_type,
    long double energies[], long double Js[], int n, long double dt,
    long double Jout[], long double I_nu[], long double qdot[], long double Qdot[],
    void (*interp_f)(long double[], long double[], long double[], int),
    void (*deriv_f)(long double[], long double[], long double[], int),
    void (*deriv_I)(long double[], long double[], long double[], int),
    int stepper, int force_pos
) {
    int i;

    struct Parameters p;
    p.kT = kT;
    p.rho_N = rho_N;
    p.Y_e = Y_e;
    p.n_type = n_type;
    p.interp_f = interp_f;
    p.deriv_f = deriv_f;
    p.deriv_I = deriv_I;
    p.force_pos = force_pos;

    // Compute relevant parameters
    long double beta = 1 / kT;
    dt /= t_unit;

    // Compute dimensionless energy zone centers and x^3 J
    long double xs[n], x3J[n];
    for (i = 0; i < n; i++) {
        xs[i] = energies[i] * beta;
        x3J[i] = xs[i]*xs[i]*xs[i] * fmaxl(Js[i], 1e-30);
    }
    
    // Compute I_nu on the bin edges
    compute_Inu(I_nu, xs, x3J, n, p);

    // Compute update to x^3 J
    switch (stepper) {
        case 1:
            rk2(x3J, xs, x3J, n, dt, p, compute_rhs);
            break;
        case 2:
            rk4(x3J, xs, x3J, n, dt, p, compute_rhs);
            break;
        default:
            euler(x3J, xs, x3J, n, dt, p, compute_rhs);
            break;
    }

    // Update Js
    for (i = 0; i < n; i++)
        Jout[i] = fmaxl(x3J[i] / powl(xs[i], 3), 1e-30);

    // Compute q dot, the spectrum of energy deposition
    // \dot{q} =  kT (I_\nu - d/dx (x I_\nu))   <- (eq. 44)
    //         = -kT d(I_\nu)/d(log x)
    long double dInu_dlogx[n];
    compute_rhs(dInu_dlogx, xs, x3J, n, p);
    for (i = 0; i < n; i++) {
        qdot[i] = - kT * dInu_dlogx[i];
    }

    // Compute Q dot, the rate of energy deposition
    // \dot{Q} = (kT)^4 / (2 \pi^2 \hbar^3 c^3) \int dx I_\nu <- (eq. 45)
    long double coeff = compute_coeff(kT, rho_N);
    long double coeffQ = powl(kT, 4) / (2 * pi*pi * powl(hbar * c, 3));
    long double edges[n+1], bin_w = logl(xs[1]) - logl(xs[0]);
    for (i = 0; i < n; i++) 
        edges[i] = xs[i] * expl(-0.5 * bin_w);
    edges[n] = xs[n-1] * expl(0.5 * bin_w);
    for (i = 0; i < n; i++)
        Qdot[i] = 0.5 * coeffQ * coeff * (edges[i+1] - edges[i]) * (I_nu[i+1] + I_nu[i]);
}

/**
 * Advances the neutrino distribution function through a time step
 * of size dt. 
 * 
 * Physical parameters
 * @param kT        nucleon temperature [MeV]
 * @param rho_N     nucleon mass density [g/cm^3]
 * @param Y_e       electron fraction
 * @param n_type    nucleon type (0: proton, 1: neutron, else: both)
 * 
 * Input
 * @param energies  energy bins (zone *centers*) [MeV]
 * @param Js        distribution function
 * @param n         number of bins (length of energies and Js arrays)
 * @param dt        time step size [s]
 * 
 * Output
 * @param Jout      updated J values [length: n]
 * @param I_nu      I_nu on the bin edges [length: n+1]
 * @param qdot      energy deposition spectrum [length: n]
 * @param Qdot      energy deposition rate [length: n]
 */
void compute_step(
    long double kT, long double rho_N, long double Y_e, int n_type,
    long double energies[], long double Js[], int n, long double dt,
    long double Jout[], long double I_nu[], long double qdot[], long double Qdot[]
) {
    compute_step_dev(kT, rho_N, Y_e, n_type, energies, Js, n, dt, Jout, I_nu,
                     qdot, Qdot, cubic_itp, cubic_diff, linear_diff, 0, 1);
}


/**
 * Simple test
 */
int main(int argc, char const *argv[]) {
    int i; // for loops

    /** PHYSICAL PARAMETERS **/
    long double beta = 1 / 1.0; // MeV^-1
    long double rhoN = 1e12; // g cm^-3
    long double Y_e = 0.2;
    long double nN = rhoN * 1e3 * c*c / (m * e_unit) * powl(l_unit,3);

    /** SIMULATION PARAMETERS **/
    int n_bin = 12;
    int n_step = 5;
    long double dt = 1e-6;
    long double t = 0;

    /** SET UP BINNING **/
    long double min_E = 1.0; // MeV
    long double max_E = 100.0; // MeV
    long double min_x = min_E * beta;
    long double max_x = max_E * beta;
    long double bin_w = (logl(max_x) - logl(min_x)) / (n_bin - 1);

    long double es[n_bin], xs[n_bin], Js[n_bin], x3J[n_bin];
    for (i = 0; i < n_bin; i++) {
        es[i] = min_E * expl(i * bin_w);
        xs[i] = min_x * expl(i * bin_w);

        // Fill J with values from a Gaussian
        // parameters: mean 10, width 3, height 1/2
        Js[i] = 0.5 * expl(-0.5 * powl((xs[i] - 10.0) / 3.0, 2));
        x3J[i] = Js[i] * powl(xs[i], 3);
    }

    /** KOMPANEETS CALCULATIONS **/
    int nucleon_type = 2; // include proton and neutron contributions
    long double coeff = compute_coeff(1 / beta, rhoN);

    int step;
    for (step = 0; step < n_step + 1; step++) {
        long double depE[n_bin+1], deltax3J[n_bin];

        // Leverage the compute_step method for testing
        long double Jout[n_bin], Inu[n_bin], qdot[n_bin], Qdot[n_bin];
        compute_step(1.0 / beta, rhoN, Y_e, nucleon_type, es, Js, n_bin, dt, Jout, Inu, qdot, Qdot);

        // Compute deposited energy and update to Js
        long double bin_w = logl(xs[1]) - logl(xs[0]);
        for (i = 0; i < n_bin; i++) {
            depE[i] = -coeff / beta * Inu[i] / powl(xs[i] * expl(-bin_w * 0.5), 2);
            deltax3J[i] = powl(xs[i], 3) * (Jout[i] - Js[i]);
            Js[i] = Jout[i];
        }

        t = step * dt;

        int kill = 0;
        int neg = 0;
        for (i = 0; i < n_bin; i++) {
            if (Js[i] < 1e-30) {
                Js[i] = 1e-30;
                neg = 1;
            }
            if (isnan(Js[i])) {
                kill = 1;
                break;
            }
        }
        if (neg) {
            printf("Negatives on step %d\n", step);
        }
        if (kill) {
            printf("Killed by NaNs on step %d\n", step);
            break;
        }

        printf("Step %d:\n", step);
        printf("f-values: ");
        for (i = 0; i < n_bin - 1; i++)
            printf("%Le, ", Js[i]);
        printf("%Le", Js[n_bin-1]);
        printf("\n\n");
    }

}
