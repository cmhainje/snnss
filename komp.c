#include <math.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#include "num.h"

// CONSTANTS
long double hbar = 1.0545718e-34;
long double c = 299792458;
long double e = 1.60217662e-19;
long double G2_init = 1.55e-33 * 0.01*0.01*0.01; // m^3/MeV^2/s
long double m_n = 939; // MeV
long double Vp = -0.5 * (1 - 4 * 0.23122);
long double Ap = -1.2723 / 2;
long double Vn = -0.5;
long double An = 1.2723 / 2;
long double pi = M_PI;

long double e_unit = 1e6 * 1.60217662e-19;                               // 1 MeV       = 1e6 * e
long double t_unit = 1.0545718e-34 / (1e6 * 1.60217662e-19);             // 6.582e-22 s = hbar / e_unit
long double l_unit = 1.0545718e-34 / (1e6 * 1.60217662e-19) * 299792458; // 1.973e-13 m = t_unit * c

long double G2 = 1.327817e-22; // converted to natural units = G2_init * t_unit / l_unit**3
long double m = 939; // MeV
long double prot = (-0.5 * (1 - 4 * 0.23122)) * (-0.5 * (1 - 4 * 0.23122)) + 5 * (-1.2723 / 2) * (-1.2723 / 2);
long double neut = (-0.5) * (-0.5) + 5 * (1.2723 / 2) * (1.2723 / 2);

bool WRITE_RESULTS = true;

long double gaussian(long double x)
{
    // return 0.5 * expl(-0.5 * powl((x - 20.0) / 5.0, 2));
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
 * Physical parameters:
 * @param kT        nucleon temperature [MeV]
 * @param rho_N     nucleon mass density [g/cm^3]
 * @param Y_e       electron fraction
 * 
 * Input
 * @param energies  energy bins (zone *centers*) [MeV]
 * @param Js        distribution function
 * @param n         number of bins (length of energies and Js arrays)
 * @param dt        time step size [s]
 * 
 * Output
 * @param Jout      length: n
 * @param I_nu      length: n+1
 * @param qdot      length: n
 * @param Qdot
 */
void for_burrows(
    long double kT, long double rho_N, long double Y_e,
    long double energies[], long double Js[], int n, long double dt,
    long double Jout[], long double I_nu[], long double qdot[], long double Qdot[]
) {
    int i;
    long double beta = 1 / kT;
    long double nN = rho_N * powl(c,2) / (m * e_unit) * powl(l_unit,3);
    long double coeff = 2 * G2 * nN / (3 * pi * powl(beta,3) * m);

    long double xs[n], x3J[n], logx[n], logx3J[n];
    for (i = 0; i < n; i++) {
        xs[i] = energies[i] * beta;
        logx[i] = logl(xs[i]);
        x3J[i] = xs[i]*xs[i]*xs[i] * Js[i];
        logx3J[i] = logl(x3J[i]);
    }

    // Compute x^3 J and d(x^3 J)/d(log x) on inner bin edges
    long double itp_x3J[n-1], dx3J_dlogx[n-1];
    cubic_itp(itp_x3J, logx, logx3J, n-1);
    cubic_diff(dx3J_dlogx, logx, logx3J, n-1);
    for (i = 0; i < n - 1; i++) {
        itp_x3J[i] = expl(itp_x3J[i]);
        dx3J_dlogx[i] = dx3J_dlogx[i] * itp_x3J[i];
    }

    // Compute I_nu on bin edges
    I_nu[0] = 0;
    I_nu[n] = 0;
    for (i = 0; i < n - 1; i++) {
        long double x, lambda_p, lambda_n, coeff_p, coeff_n, phi;
        x = sqrtl(xs[i] * xs[i+1]);
        lambda_p = ((4*x-14)*powl(Vp,2) + (28*x-86)*powl(Ap,2)) / (prot * beta * m);
        lambda_n = ((4*x-14)*powl(Vn,2) + (28*x-86)*powl(An,2)) / (neut * beta * m);
        coeff_p = prot * Y_e * (1 - lambda_p); 
        coeff_n = neut * (1 - Y_e) * (1 - lambda_n);
        
        I_nu[i+1] = (coeff_p + coeff_n) * (
            powl(x, 2) * dx3J_dlogx[i] - 3 * powl(x, 2) * itp_x3J[i] 
            + powl(x, 3) * itp_x3J[i] - powl(itp_x3J[i], 2)
        );
    }

    // Compute d(I_nu)/d(log x) on bin centers
    long double log_edges[n+1];
    long double bin_w = logx[1] - logx[0];
    for (i = 0; i < n; i++)
        log_edges[i] = logx[i] - 0.5 * bin_w;
    log_edges[n] = logx[n-1] + 0.5 * bin_w;

    long double dInu_dlogx[n];
    cubic_diff(dInu_dlogx, log_edges, I_nu, n);

    // Update Js 
    for (i = 0; i < n; i++) {
        x3J[i] += dt * coeff * dInu_dlogx[i];
        Jout[i] = x3J[i] / powl(xs[i], 3);
    }

    // Compute q dot, the spectrum of energy deposition
    // \dot{q} =  kT (I_\nu - d/dx (x I_\nu))  <- eq. 44
    //         =  kT (I_\nu - (I_\nu + x d(I_\nu)/dx))
    //         = -kT x d(I_\nu)/d(log x)
    //         = -kT d(I_\nu)/d(log x)
    for (i = 0; i < n; i++) {
        qdot[i] = - kT * coeff * dInu_dlogx[i];
    }

    // Compute Q dot, the rate of energy deposition
    // \dot{Q} = (kT)^4 / (2 \pi^2 \hbar^3 c^3) \int dx I_\nu
    // hmm... this will require some thought
    // for now, I'm going to compute something completely different lol

}


/**
 * @param kT        temperature [MeV]
 * @param rho_N     mass density [g cm^-3]
 * @param Y_e       electron fraction
 * @param es        zone centers [MeV]
 * @param Js        mutable
 * @param Jout      (want: don't mutate Js, use this instead)
 * @param dt        step size [s]
 * 
 * @param Qdot      want
 * 
 * @param I_nu
 * @param qdot
 * 
 * @param t         don't use
 * @param depE      make optional
 * @param deltax3J  make optional
 */
void compute_step(long double depE[], long double deltax3J[],
    long double xs[], long double Js[], int n_bin, long double t, long double dt,
    long double beta, long double rhoN, long double Y_e)
{
    int i;

    // Make physically relevant parameters
    long double nN = rhoN * powl(c,2) / (m * e_unit) * powl(l_unit,3);

    // Compute interpolated values of x3J, d(x3J)/d(log x) on bin edges
    long double logx[n_bin], x3J[n_bin], logx3J[n_bin];
    for (i = 0; i < n_bin; i++) {
        logx[i] = logl(xs[i]);
        x3J[i] = Js[i] * powl(xs[i], 3);
        logx3J[i] = logl(x3J[i]);
    }
    long double itp_x3J[n_bin-1], d_x3J[n_bin-1];
    cubic_itp(itp_x3J, logx, logx3J, n_bin-1);
    cubic_diff(d_x3J, logx, logx3J, n_bin-1);
    for (i = 0; i < n_bin - 1; i++) {
        itp_x3J[i] = expl(itp_x3J[i]);
        d_x3J[i] = d_x3J[i] * itp_x3J[i];
    }

    // Compute I_nu on bin edges
    long double Inu[n_bin+1];
    Inu[0] = 0;
    Inu[n_bin] = 0;
    for (i = 0; i < n_bin - 1; i++) {
        long double x, lambda_p, lambda_n, coeff_p, coeff_n, phi;
        x = sqrtl(xs[i] * xs[i+1]);
        lambda_p = ((4*x-14)*powl(Vp,2) + (28*x-86)*powl(Ap,2)) / (prot * beta * m);
        lambda_n = ((4*x-14)*powl(Vn,2) + (28*x-86)*powl(An,2)) / (neut * beta * m);
        coeff_p = prot * Y_e * (1 - lambda_p); 
        coeff_n = neut * (1 - Y_e) * (1 - lambda_n);
        
        Inu[i+1] = (coeff_p + coeff_n) * (
            powl(x, 2) * d_x3J[i] - 3 * powl(x, 2) * itp_x3J[i] 
            + powl(x, 3) * itp_x3J[i] - powl(itp_x3J[i], 2)
        );
    }

    // Compute d(I_nu)/d(log x) on bin centers
    long double log_edges[n_bin+1];
    long double bin_w = logx[1] - logx[0];
    for (i = 0; i < n_bin; i++)
        log_edges[i] = logx[i] - 0.5 * bin_w;
    log_edges[n_bin] = logx[n_bin-1] + 0.5 * bin_w;
    long double dInu[n_bin];
    linear_diff(dInu, log_edges, Inu, n_bin);

    // Compute deposited energy and update to Js
    long double coeff = 2 * G2 * nN / (3 * pi * powl(beta,3) * m);
    for (i = 0; i < n_bin; i++) {
        depE[i] = -coeff / beta * Inu[i] / powl(xs[i] * expl(-bin_w * 0.5), 2);
        deltax3J[i] = dt * coeff * dInu[i];
        x3J[i] += deltax3J[i];
        Js[i] = x3J[i] / powl(xs[i], 3);
    }
}

int main(int argc, char const *argv[])
{
    int i; // for loops

    // Get number of bins from command-line
    int n_bin = atoi(argv[1]);

    /** PARAMETERS **/
    long double beta = 1.0; // MeV^-1  (1-4 MeV)
    long double rhoN = 1e13; // kg m^-3 (10^10 g/cm^3)
    long double Y_e  = 0.2; // (0.2 - 0.3)
    long double nN = rhoN * powl(c,2) / (m * e_unit) * powl(l_unit,3);
    long double t = 0;
    long double dt = 1e-6 / t_unit; // 1e-6
    int n_step = 50000;

    /** SET UP BINNING **/
    long double min_E = 1.0;   // MeV
    long double max_E = 100.0; // MeV
    long double min_x = min_E * beta;
    long double max_x = max_E * beta;
    long double bin_w = (logl(max_x) - logl(min_x)) / (n_bin - 1);

    long double xs[n_bin], Js[n_bin], x3J[n_bin];
    for (i = 0; i < n_bin; i++) {
        xs[i] = min_x * expl(i * bin_w);
        Js[i] = gaussian(xs[i] / beta);
        x3J[i] = Js[i] * powl(xs[i], 3);
    }

    /** GET READY TO WRITE **/
    char *filename;
    asprintf(&filename, "results/cubicdf_lineardI/n%d_dt1e-6.out", n_bin);
    FILE *file = fopen(filename, "w");

    /** KOMPANEETS CALCULATIONS **/
    int step;
    for (step = 0; step < n_step + 1; step++) {

        long double depE[n_bin+1], deltax3J[n_bin];
        compute_step(depE, deltax3J, xs, Js, n_bin, t, dt, beta, rhoN, Y_e);
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
