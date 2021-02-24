# Solver for inelastic neutrino-nucleon scattering in supernovae
This repository implements a solver for inelastic neutrino-nucleon scattering in
supernova explosions, based on the Kompaneets formalism developed by [Wang and
Burrows](https://arxiv.org/abs/2006.12240).

## Use
There are two primary ways to use the methods provided herein. One can use the
solver itself by compiling `komp.c` as follows
```
gcc komp.c num.c -o komp
```
The resulting program takes a single integer command-line argument which is the
number of sampling points to use to sample the energy distribution of neutrinos.
It then starts with a sample distribution of a Gaussian (mean 10 MeV, width 3
MeV) and uses the hard-coded values of nucleon temperature (`kT)`, mass density
(`rhoN`), etc. and evolves this distribution in time using the solver. The
results are written to a folder called `results/`, and can be analyzed using the
`py/analysis.ipynb` notebook.

The other method of use is to compile the numerical methods for external use.
This can be done with 
```
make ext
```
This command will make an outward-facing library called `lib.so` with the
interpolation, differentiation, and ODE-solving methods. These methods can then
be used in other applications, such as Python or Julia notebooks. As an example,
see the `py/eval.ipynb` notebook, which evaluates the different numerical
methods.