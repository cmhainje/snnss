#include <math.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#include "komp.h"
#include "num.h"

bool WRITE_RESULTS = false;

long double gaussian(long double x)
{
    return 0.5 * expl(-0.5 * powl((x - 10.0) / 3.0, 2));
}

void write_out(FILE *file, int step, long double t, int n,
               long double N_x[], long double N_y[],
               long double E_x[], long double E_y[],
               long double J_x[], long double J_y[])
{
    // Write step number and time
    fprintf(file, "%d %Lf\n", step, t);
    
    // Write N_x, N_y
    for (int i = 0; i < n; i++)
        fprintf(file, "%Le ", N_x[i]);
    fputc('\n', file);
    for (int i = 0; i < n; i++)
        fprintf(file, "%Le ", N_y[i]);
    fputs("\n\n", file);

    // Write E_x, E_y
    for (int i = 0; i < n; i++)
        fprintf(file, "%Le ", E_x[i]);
    fputc('\n', file);
    for (int i = 0; i < n; i++)
        fprintf(file, "%Le ", E_y[i]);
    fputs("\n\n", file);

    // Write J_x, J_y
    for (int i = 0; i < n; i++)
        fprintf(file, "%Le ", J_x[i]);
    fputc('\n', file);
    for (int i = 0; i < n; i++)
        fprintf(file, "%Le ", J_y[i]);
    fputs("\n", file);

    // Write a separator
    fputs("---\n\n", file);
}

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
 * Computes a step update using the default numerical methods:
 * cubic interpolation and differentiation on f(x),
 * linear interpolation on I_nu, and Euler stepping.
 * 
 * Physical parameters:
 * @param kT        nucleon temperature [MeV]
 * @param rho_N     nucleon mass density [g/cm^3]
 * @param Y_e       electron fraction
 * @param n_type    nucleon type (0: proton, 1: neutron, else: both)
 * 
 * Input
 * @param energies  energy zone centers [MeV]
 * @param Js        distribution function
 * @param n         number of bins (length of energies and Js arrays)
 * @param dt        time step size [s]
 * 
 * Output
 * @param Jout      length: n
 * @param I_nu      length: n+1
 * @param qdot      length: n
 * @param Qdot      length: n (for now)
 */
void compute_step(
    long double kT, long double rho_N, long double Y_e, int n_type,
    long double energies[], long double Js[], int n, long double dt,
    long double Jout[], long double I_nu[], long double qdot[], long double Qdot[]
) {
    compute_step_dev(kT, rho_N, Y_e, n_type, energies, Js, n, dt, Jout, I_nu,
                     qdot, Qdot, cubic_itp, cubic_diff, linear_diff, 0, true);
}


/**
 * Physical parameters:
 * @param kT        nucleon temperature [MeV]
 * @param rho_N     nucleon mass density [g/cm^3]
 * @param Y_e       electron fraction
 * @param n_type    nucleon type (0: proton, 1: neutron, else: both)
 * 
 * Input
 * @param energies  energy zone centers [MeV]
 * @param Js        distribution function
 * @param n         number of bins (length of energies and Js arrays)
 * @param dt        time step size [s]
 * 
 * Output
 * @param Jout      length: n
 * @param I_nu      length: n+1
 * @param qdot      length: n
 * @param Qdot      length: n (for now)
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
    int stepper, bool force_pos
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
    // compute_Inu(I_nu, xs, x3J, n, beta, Y_e, n_type, interp_f, deriv_f);
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


    /* 
    // compute_rhs(dInu_dlogx, xs, x3J, n, kT, rho_N, Y_e, n_type, interp_f, deriv_f, deriv_I);

    // Update Js 
    for (i = 0; i < n; i++) {
        x3J[i] += dt * dInu_dlogx[i];
        Jout[i] = fmaxl(x3J[i] / powl(xs[i], 3), 1e-30);
    }
    */

    // Compute q dot, the spectrum of energy deposition
    // \dot{q} =  kT (I_\nu - d/dx (x I_\nu))   <- (eq. 44)
    //         = -kT d(I_\nu)/d(log x)
    long double dInu_dlogx[n];
    compute_rhs(dInu_dlogx, xs, x3J, n, p);
    for (i = 0; i < n; i++) {
        qdot[i] = - kT * dInu_dlogx[i];
    }

    // Compute Q dot, the rate of energy deposition
    // \dot{Q} = (kT)^4 / (2 \pi^2 \hbar^3 c^3) \int dx I_\nu
    long double coeff = compute_coeff(kT, rho_N);
    long double coeffQ = powl(kT, 4) / (2 * pi*pi * powl(hbar * c, 3));
    long double edges[n+1], bin_w = logl(xs[1]) - logl(xs[0]);
    for (i = 0; i < n; i++) 
        edges[i] = xs[i] * expl(-0.5 * bin_w);
    edges[n] = xs[n-1] * expl(0.5 * bin_w);
    for (i = 0; i < n; i++)
        Qdot[i] = 0.5 * coeffQ * coeff * (edges[i+1] - edges[i]) * (I_nu[i+1] + I_nu[i]);
}

int main(int argc, char const *argv[]) {
    int i; // for loops

    // Get number of bins from command-line
    int n_bin = atoi(argv[1]);

    /** PARAMETERS **/
    long double beta = 1 / 1.0; // MeV^-1  (1-4 MeV)
    long double rhoN = 1e12; // g cm^-3
    long double Y_e  = 0.2; // (0.2 - 0.3)
    long double nN = rhoN * 1e3 * powl(c,2) / (m * e_unit) * powl(l_unit,3);
    long double t = 0;
    long double dt = 1e-6; // 1e-6
    int n_step = 5;

    bool include_protons = true;
    bool include_neutrons = true;

    /** SET UP BINNING **/
    long double min_E = 1.0;   // MeV
    long double max_E = 100.0; // MeV
    long double min_x = min_E * beta;
    long double max_x = max_E * beta;
    long double bin_w = (logl(max_x) - logl(min_x)) / (n_bin - 1);

    long double es[n_bin], xs[n_bin], Js[n_bin], x3J[n_bin];
    for (i = 0; i < n_bin; i++) {
        es[i] = min_E * expl(i * bin_w);
        xs[i] = min_x * expl(i * bin_w);
        Js[i] = gaussian(xs[i] / beta);
        x3J[i] = Js[i] * powl(xs[i], 3);
    }

    /** GET READY TO WRITE **/
    char *filename;
    asprintf(&filename, "results/cubicdf_lineardI/n%d_dt1e-6.out", n_bin);
    FILE *file = fopen(filename, "w");

    /** KOMPANEETS CALCULATIONS **/
    int nucleon_type;
    if (include_protons && include_neutrons)
        nucleon_type = 2; // both
    else if (include_protons)
        nucleon_type = 0; // protons only
    else 
        nucleon_type = 1; // neutrons only

    long double coeff = compute_coeff(1/beta, rhoN);

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

        bool kill = false;
        bool neg = false;
        for (i = 0; i < n_bin; i++) {
            if (Js[i] < 1e-30) {
                Js[i] = 1e-30;
                neg = true;
            }
            if (isnan(Js[i])) {
                kill = true;
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

        if (WRITE_RESULTS && (step % 1000) == 0) { 
            long double depE_x[n_bin], depE_y[n_bin];
            long double depN_x[n_bin], depN_y[n_bin];
            long double J_x[n_bin], J_y[n_bin];
            for (int i = 0; i < n_bin; i++) {
                long double energy = xs[i] / beta;
                depE_x[i] = energy * expl(-0.5 * bin_w);
                depE_y[i] = depE[i];
                depN_x[i] = energy;
                depN_y[i] = deltax3J[i] / dt / powl(xs[i], 3);
                J_x[i] = energy;
                J_y[i] = Js[i];
            }
            write_out(file, step, t * t_unit, n_bin, depN_x, depN_y, depE_x, depE_y, J_x, J_y);
        }
    }

    fclose(file);
    return 0;
}
