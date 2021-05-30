/**
 * main.h
 * Connor Hainje (cmhainje@gmail.com)
 * Provides all of the Kompaneets functionality in a single C90-compatible file
 * This header allows one to include the relevant external functions
 */

#ifndef CH_KOMP
#define CH_KOMP

#include <math.h>
#include <stdio.h>


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
 * Input (arrays which are used by the routine)
 * @param energies  energy bins (zone *centers*) [MeV]
 * @param Js        distribution function
 * @param n         number of bins (length of energies and Js arrays)
 * @param dt        time step size [s]
 * 
 * Output (arrays which are filled by the routine)
 * @param Jout      updated J values [length: n]
 * @param I_nu      I_nu on the bin edges [length: n+1]
 * @param qdot      energy deposition spectrum [length: n]
 * @param Qdot      energy deposition rate [length: n]
 */
void compute_step(
    long double kT, long double rho_N, long double Y_e, int n_type,
    long double energies[], long double Js[], int n, long double dt,
    long double Jout[], long double I_nu[], long double qdot[], long double Qdot[]
);

#endif
