# Solver for inelastic neutrino-nucleon scattering in supernovae
This repository implements a solver for inelastic neutrino-nucleon scattering in
supernova explosions, based on the Kompaneets formalism developed by [Wang and
Burrows](https://arxiv.org/abs/2006.12240).

## Use
There are two primary ways to use the methods provided herein. The primary way
that one can use the solver is to use the `main()` method in `komp.c` to test
the step function. To do this, one can run 
```
make
```
This creates an executable called `komp` than takes a single integer
command-line argument, which is the number of energy zones to use. By default,
it starts with a sample distribution that is a Gaussian with mean 10 MeV and
width 3 MeV. Physical parameters, such as the nucleon temperature (`kT`) and the
mass density (`rhoN`) are hard-coded values. The distribution is evolved in time
for a fixed number of steps, and snapshots are written to a folder called
`results/` which can be analyzed using the `py/analysis.ipynb` notebook.

The other primary method is to compile the methods for external use. This can be
done with
```
make ext
```
This command will make an outward-facing library called `lib.so` with the
the step, interpolation, differentiation, and ODE-solving methods. These
methods can then be used in other applications, such as Python or Julia
notebooks. As an example, see the `py/eval.ipynb` notebook, which evaluates
the different numerical methods, or the `py/test.ipynb` which runs the full
solver.