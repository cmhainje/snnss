#ifndef CH_KOMP
#define CH_KOMP

#include <math.h>

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

/**
 * Computes the physical coefficient (referred to as `alpha` in my paper).
 *
 * @param kT    nucleon temperature [MeV]
 * @param rho_N nucleon mass density [g cm^-3]
 */
long double compute_coeff(long double kT, long double rho_N);

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
 * @param Qdot      currently just zeros [length: n]
 */
void compute_step(
    long double kT, long double rho_N, long double Y_e, int n_type,
    long double energies[], long double Js[], int n, long double dt,
    long double Jout[], long double I_nu[], long double qdot[], long double Qdot[]
);

/**
 * Dev version of `compute_step()`. Provides a method for directly
 * specifying the interpolation and differentiation methods.
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
 * @param Qdot      currently just zeros [length: n]
 * 
 * Numerical methods
 * @param interp_f  interpolation function to use on f
 * @param deriv_f   differentiation function to use on f
 * @param deriv_I   differentiation function to use on I_nu
 */
void compute_step_dev(
    long double kT, long double rho_N, long double Y_e, int n_type,
    long double energies[], long double Js[], int n, long double dt,
    long double Jout[], long double I_nu[], long double qdot[], long double Qdot[],
    void (*interp_f)(long double[], long double[], long double[], int),
    void (*deriv_f)(long double[], long double[], long double[], int),
    void (*deriv_I)(long double[], long double[], long double[], int),
    int stepper, bool force_pos
);

#endif
