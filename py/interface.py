"""
interface.py
Connor Hainje

Provides a Python interface for the methods.
Requires that `make ext` has been run in the parent directory.
"""

import ctypes 
import numpy as np

lib = ctypes.CDLL("../lib.so")

# Some useful constants
E_UNIT = ctypes.c_longdouble.in_dll(lib, "e_unit").value
T_UNIT = ctypes.c_longdouble.in_dll(lib, "t_unit").value
L_UNIT = ctypes.c_longdouble.in_dll(lib, "l_unit").value

# Interpolation and differentiation
atypes = [
    np.ctypeslib.ndpointer(dtype=ctypes.c_longdouble, ndim=1, flags=('C','W')),
    np.ctypeslib.ndpointer(dtype=ctypes.c_longdouble, ndim=1, flags=('C','W')),
    np.ctypeslib.ndpointer(dtype=ctypes.c_longdouble, ndim=1, flags=('C','W')),
    ctypes.c_int
]
lib.linear_itp.argtypes = atypes
lib.cubic_itp.argtypes = atypes
lib.spline_itp.argtypes = atypes
lib.linear_diff.argtypes = atypes
lib.cubic_diff.argtypes = atypes
lib.spline_diff.argtypes = atypes

def use(method, xs, ys):
    """Helper method to use interpolation and differentiation methods

    Args:
        method (cfunction): method from the shared lib
        xs (ndarray): array of sampling points
        ys (ndarray): array of sampled distribution values

    Returns:
        ndarray: numpy array with output values
    """
    n = len(xs) - 1
    out = np.empty(n, dtype=ctypes.c_longdouble)
    method(out, xs.astype(ctypes.c_longdouble), ys.astype(ctypes.c_longdouble), n)
    return out 

def linear_itp(xs, ys):
    """Interpolates to midpoint of each interval using linear interpolants

    Args:
        xs (ndarray): array of sampling points
        ys (ndarray): array of sampled distribution values

    Returns:
        ndarray: array of interpolated values
    """
    return use(lib.linear_itp, xs, ys)

def cubic_itp(xs, ys):
    """Interpolates to midpoint of each interval using cubic interpolants

    Args:
        xs (ndarray): array of sampling points
        ys (ndarray): array of sampled distribution values

    Returns:
        ndarray: array of interpolated values
    """
    return use(lib.cubic_itp, xs, ys)

def spline_itp(xs, ys):
    """Interpolates to midpoint of each interval using cubic splines

    Args:
        xs (ndarray): array of sampling points
        ys (ndarray): array of sampled distribution values

    Returns:
        ndarray: array of interpolated values
    """
    return use(lib.spline_itp, xs, ys)

def linear_diff(xs, ys):
    """Computes derivative at midpoint of each interval using linear interpolants

    Args:
        xs (ndarray): array of sampling points
        ys (ndarray): array of sampled distribution values

    Returns:
        ndarray: array of derivative values
    """
    return use(lib.linear_diff, xs, ys)

def cubic_diff(xs, ys):
    """Computes derivative at midpoint of each interval using cubic interpolants

    Args:
        xs (ndarray): array of sampling points
        ys (ndarray): array of sampled distribution values

    Returns:
        ndarray: array of derivative values
    """
    return use(lib.cubic_diff, xs, ys)

def spline_diff(xs, ys):
    """Computes derivative at midpoint of each interval using cubic splines

    Args:
        xs (ndarray): array of sampling points
        ys (ndarray): array of sampled distribution values

    Returns:
        ndarray: array of derivative values
    """
    return use(lib.spline_diff, xs, ys)

# Formalism methods
lib.compute_coeff.argtypes = [ctypes.c_longdouble, ctypes.c_longdouble]
lib.compute_coeff.restype = ctypes.c_longdouble
def compute_coeff(kT, rho_N):
    """Computes the physical coefficient, alpha in my paper

    Args:
        kT (float): nucleon temperature in MeV
        rho_N (float): nucleon mass density in g/cm^3

    Returns:
        float: physical coefficient of I_nu
    """
    return lib.compute_coeff(kT, rho_N)

atypes = [
    ctypes.c_longdouble,
    ctypes.c_longdouble,
    ctypes.c_longdouble,
    ctypes.c_int,
    np.ctypeslib.ndpointer(dtype=ctypes.c_longdouble, ndim=1, flags=('C','W')),
    np.ctypeslib.ndpointer(dtype=ctypes.c_longdouble, ndim=1, flags=('C','W')),
    ctypes.c_int,
    ctypes.c_longdouble,
    np.ctypeslib.ndpointer(dtype=ctypes.c_longdouble, ndim=1, flags=('C','W')),
    np.ctypeslib.ndpointer(dtype=ctypes.c_longdouble, ndim=1, flags=('C','W')),
    np.ctypeslib.ndpointer(dtype=ctypes.c_longdouble, ndim=1, flags=('C','W')),
    np.ctypeslib.ndpointer(dtype=ctypes.c_longdouble, ndim=1, flags=('C','W')),
]
lib.compute_step.argtypes = atypes

def compute_step(kT, rho_N, Y_e, n_type, energies, Js, dt, n_type=2):
    """Computes a step update of step size dt

    Args:
        kT (float): nucleon temperature in MeV
        rho_N (float): nucleon mass density in g/cm^3
        Y_e (float): electron fraction (dimensionless)
        n_type (int): nucleon type (0: protons, 1: neutrons, else: both)
        energies (ndarray): centers of energy zones in MeV
        Js (ndarray): neutrino distribution sampled at zone centers
        dt (float): desired stepsize in seconds

    Returns:
        ndarray: updated J distribution
        ndarray: I_nu on zone boundaries (eq. 42)
        ndarray: q dot on zone centers (eq. 44)
        ndarray: Q dot on zone centers (eq. 45)
    """
    n = len(energies)
    Jout = np.empty(n, dtype=ctypes.c_longdouble)
    I_nu = np.empty(n+1, dtype=ctypes.c_longdouble)
    qdot = np.empty(n, dtype=ctypes.c_longdouble)
    Qdot = np.empty(n, dtype=ctypes.c_longdouble)

    lib.compute_step(
        kT, rho_N, Y_e, n_type,
        energies.astype(ctypes.c_longdouble),
        Js.astype(ctypes.c_longdouble),
        n, dt,
        Jout, I_nu, qdot, Qdot
    )
    return Jout, I_nu, qdot, Qdot

lib.compute_step_dev.argtypes = atypes + [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
]

def compute_step_dev(kT, rho_N, Y_e, n_type, energies, Js, dt, itp_f, drv_f, drv_I):
    """Computes a step update of step size dt

    Args:
        kT (float): nucleon temperature in MeV
        rho_N (float): nucleon mass density in g/cm^3
        Y_e (float): electron fraction (dimensionless)
        n_type (int): nucleon type (0: protons, 1: neutrons, else: both)
        energies (ndarray): centers of energy zones in MeV
        Js (ndarray): neutrino distribution sampled at zone centers
        dt (float): desired stepsize in seconds
        itp_f (string): desired interpolation method to use on f (J). Allowed
        values are "linear", "cubic", or "spline".
        drv_f (string): desired differentiation method to use on f (J). Allowed
        values are "linear", "cubic", or "spline".
        drv_I (string): desired differentiation method to use on I_nu. Allowed
        values are "linear", "cubic", or "spline".

    Returns:
        ndarray: updated J distribution
        ndarray: I_nu on zone boundaries (eq. 42)
        ndarray: q dot on zone centers (eq. 44)
        ndarray: Q dot on zone centers (eq. 45)
    """
    itp_methods = {'linear': lib.linear_itp, 'cubic': lib.cubic_itp, 'spline': lib.spline_itp}
    drv_methods = {'linear': lib.linear_diff, 'cubic': lib.cubic_diff, 'spline': lib.spline_diff}

    n = len(energies)
    Jout = np.empty(n, dtype=ctypes.c_longdouble)
    I_nu = np.empty(n+1, dtype=ctypes.c_longdouble)
    qdot = np.empty(n, dtype=ctypes.c_longdouble)
    Qdot = np.empty(n, dtype=ctypes.c_longdouble)

    lib.compute_step_dev(
        kT, rho_N, Y_e, n_type,
        energies.astype(ctypes.c_longdouble),
        Js.astype(ctypes.c_longdouble),
        n, dt,
        Jout, I_nu, qdot, Qdot,
        itp_methods[itp_f],
        drv_methods[drv_f],
        drv_methods[drv_I]
    )
    return Jout, I_nu, qdot, Qdot
