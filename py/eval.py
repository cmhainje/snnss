# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.10.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Evaluation notebook
# This notebook is built to test my implementations of various numerical methods in C.

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from interface import *

# %%
def gauss(x, mu, sigma):
    """Returns the value of a gaussian with mean mu, width sigma at x"""
    return 0.5 * np.exp(-0.5 * (x - mu)**2 / sigma**2)

def dgauss(x, mu, sigma):
    """Returns the derivative of a gaussian with mean mu, width sigma at x"""
    return -(x - mu) / sigma**2 * gauss(x, mu, sigma)

def compute_error(xs, ys, ref_f):
    """Computes the least absolute error (LAE) between the sampled values and a
    reference distribution.

    Args:
        xs (ndarray): sampling points
        ys (ndarray): sampled values
        ref (function): reference distribution function of one variable, x

    Returns:
        float: the least absolute error
    """
    return np.sum(np.abs(ys - ref_f(xs)))

# %% [markdown]
# # Interpolators

# %%
# Make samples
xs = np.geomspace(1, 100, 12)
ys = gauss(xs, 10, 3)

# Plot
plt.figure(figsize=(12,8))
xref = np.linspace(1, 100, 500)
plt.plot(xref, gauss(xref, 10, 3), 'k--')
edges = np.sqrt(xs[:-1] * xs[1:])

for itp in [linear_itp, cubic_itp, spline_itp]:
    plt.plot(edges, itp(xs, ys), 'o')

plt.legend(['Reference', 'Linear', 'Cubic Lagrange', 'Cubic spline'])
plt.show()

# %%
# Make samples
xs = np.geomspace(1, 100, 12)
ys = gauss(xs, 10, 3)

# Plot
plt.figure(figsize=(12,8))
xref = np.linspace(1, 100, 500)
plt.plot(xref, gauss(xref, 10, 3), 'k--')
edges = np.sqrt(xs[:-1] * xs[1:])

for itp in [linear_itp, cubic_itp, spline_itp]:
    plt.plot(edges, np.exp(itp(xs, np.log(ys))), 'o')

plt.legend(['Reference', 'Linear', 'Cubic Lagrange', 'Cubic spline'])
plt.show()

# %%
nbins = [8, 16]

fig, ax = plt.subplots(len(nbins),1, figsize=(8,5), tight_layout=True)
for a in ax:
    xref = np.linspace(1, 100, 500)
    a.plot(xref, gauss(xref, 10, 3), 'k--')

for a, n in zip(ax, nbins):
    a.set_title(f'{n} intervals')

for i, n in enumerate(nbins):
    xs = np.geomspace(1, 100, n+1)
    ys = gauss(xs, 10, 3)
    edges = np.sqrt(xs[:-1] * xs[1:])

    for itp in [linear_itp, cubic_itp, spline_itp]:
        ax[i].plot(edges, itp(xs, ys), 'o')

for a in ax:
    a.set_ylabel("$f(x)$")
    a.legend(['Reference', 'Linear', 'Cubic Lagrange', 'Cubic spline'])
ax[-1].set_xlabel('$x$')
# plt.savefig('plots/itp_methods.pdf', bbox_inches='tight')
plt.show()


fig, ax = plt.subplots(len(nbins),1, figsize=(8,5), tight_layout=True)
for a in ax:
    xref = np.linspace(1, 100, 500)
    a.plot(xref, gauss(xref, 10, 3), 'k--')

for a, n in zip(ax, nbins):
    a.set_title(f'{n} intervals')

for i, n in enumerate(nbins):
    xs = np.geomspace(1, 100, n+1)
    ys = gauss(xs, 10, 3)
    edges = np.sqrt(xs[:-1] * xs[1:])

    for itp in [linear_itp, cubic_itp, spline_itp]:
        ax[i].plot(edges, np.exp(itp(xs, np.log(ys))), 'o')

for a in ax:
    a.set_ylabel("$f(x)$")
    a.legend(['Reference', 'Linear', 'Cubic Lagrange', 'Cubic spline'])
ax[-1].set_xlabel('$x$')
# plt.savefig('plots/itp_methods_logexp.pdf', bbox_inches='tight')
plt.show()


# %%
# Error versus number of bins
def get_errs(itp, nbins, pos=False, logx=False):
    func = lambda x: gauss(x, 10, 3)
    errs = []
    for n in nbins:
        xs = np.geomspace(1, 100, n)
        ys = func(xs)

        if pos and logx: lin = np.exp(itp(np.log(xs), np.log(ys)))
        elif pos:        lin = np.exp(itp(xs, np.log(ys)))
        elif logx:       lin = itp(np.log(xs), ys)
        else:            lin = itp(xs, ys)

        x_itp = np.sqrt(xs[:-1] * xs[1:])
        errs.append(compute_error(x_itp, lin, func))
    return errs

nbins = np.unique(np.geomspace(12, 200, 100).astype(int))
fmt = '-'

fig, ax = plt.subplots(3, 1, figsize=(6,10), tight_layout=True)

for i, l in zip([linear_itp, cubic_itp, spline_itp],
                ['Linear', 'Cubic Lagrange', 'Cubic spline']):
    e = get_errs(i, nbins, pos=False, logx=False)
    slope = linregress(np.log(nbins), np.log(e).astype('float64')).slope
    ax[0].plot(np.log(nbins), np.log(e), fmt, label=f"{l} (slope: {slope:.2f})")

    e = get_errs(i, nbins, pos=False, logx=True)
    slope = linregress(np.log(nbins), np.log(e).astype('float64')).slope
    ax[1].plot(np.log(nbins), np.log(e), fmt, label=f"{l} (slope: {slope:.2f})")

    e = get_errs(i, nbins, pos=True, logx=True)
    slope = linregress(np.log(nbins), np.log(e).astype('float64')).slope
    ax[2].plot(np.log(nbins), np.log(e), fmt, label=f"{l} (slope: {slope:.2f})")

ax[0].set_title('Standard interpolation')
ax[1].set_title('Interpolation with log(x)')
ax[2].set_title('Forced positive interpolation with log(x)')
for a in ax:
    a.set_xlabel('log(# bins)')
    a.set_ylabel('log(LAE)')
    a.set_ylim((-14,-1))
    a.legend()
# plt.savefig('plots/itp_error.pdf', bbox_inches='tight')
plt.show()

# %%
# Time versus number of bins
from datetime import datetime, timedelta

def get_times(itp, nbins):
    times = []
    for n in nbins:
        xs = np.geomspace(1, 100, n)
        ys = gauss(xs, 10, 3)

        trials = []
        for t in range(1000):
            start = datetime.now()
            lin = itp(xs, ys)
            stop = datetime.now()
            trials.append((stop - start) / timedelta(microseconds=1))
        times.append(np.average(trials))
    return times

nbins = np.unique(np.geomspace(12, 200, 100).astype(int))
plt.figure(figsize=(8,6))
for i in [linear_itp, cubic_itp, spline_itp]:
    times = get_times(i, nbins)
    plt.plot(nbins, times, '-')
    # plt.plot(np.log(nbins), np.log(times), '.')
plt.legend(['Linear', 'Cubic Lagrange', 'Cubic spline'])
plt.xlabel('Number of sampling points')
plt.ylabel('Time [ms]')
# plt.savefig('plots/itp_time.pdf', bbox_inches='tight')
plt.show()

# %% [markdown]
# # Differentiators

# %%
# Make samples
xs = np.geomspace(1, 100, 25)
ys = gauss(xs, 10, 3)

# Plot
plt.figure(figsize=(12,4))
xref = np.linspace(1, 100, 500)
plt.plot(xref, dgauss(xref, 10, 3), 'k--')
edges = np.sqrt(xs[:-1] * xs[1:])

for dif in [linear_diff, cubic_diff, spline_diff]:
    plt.plot(edges, dif(xs, ys), 'o')

plt.legend(['Reference', 'Linear', 'Cubic Lagrange', 'Cubic spline'])
plt.show()

# %%
nbins = [8, 16]

fig, ax = plt.subplots(len(nbins),1, figsize=(8,5), tight_layout=True)
for a in ax:
    xref = np.linspace(1, 100, 500)
    a.plot(xref, xref * dgauss(xref, 10, 3), 'k--')

for a, n in zip(ax, nbins):
    a.set_title(f'{n} intervals')

for i, n in enumerate(nbins):
    xs = np.geomspace(1, 100, n+1)
    ys = gauss(xs, 10, 3)
    edges = np.sqrt(xs[:-1] * xs[1:])

    for dif in [linear_diff, cubic_diff, spline_diff]:
        lin = dif(np.log(xs), ys)
        ax[i].plot(edges, lin, 'o')

for a in ax:
    a.legend(['Reference', 'Linear', 'Cubic Lagrange', 'Cubic spline'])
    a.set_ylabel("$df\,/\,d(\log x)$")
ax[-1].set_xlabel('$x$')
# plt.savefig('plots/deriv_methods.pdf', bbox_inches='tight')
plt.show()


# %%
# Error versus number of bins
def get_errs(dif, nbins, logx=False):
    errs = []

    dfunc = lambda x: dgauss(x, 10, 3)

    for n in nbins:
        xs = np.geomspace(1, 100, n+1)
        ys = gauss(xs, 10, 3)

        x_itp = np.sqrt(xs[:-1] * xs[1:])

        if logx:
            lin = dif(np.log(xs), ys)
            dgauss_dlogx = lambda x: x * dfunc(x)
            errs.append(compute_error(x_itp, lin, dgauss_dlogx))
        else:
            lin = dif(xs, ys)
            errs.append(compute_error(x_itp, lin, dfunc))

    return errs

nbins = np.unique(np.geomspace(12, 200, 100).astype(int))

fig, ax = plt.subplots(2, 1, figsize=(7,12))
for d, l in zip([linear_diff, cubic_diff, spline_diff],
                ['Linear', 'Cubic Lagrange', 'Cubic spline']):
    e = get_errs(d, nbins)
    slope = linregress(np.log(nbins), np.log(e).astype('float64')).slope
    ax[0].plot(np.log(nbins), np.log(e), '-', label=f"{l} (slope: {slope:.2f})")
    
    e = get_errs(d, nbins, logx=True)
    slope = linregress(np.log(nbins), np.log(e).astype('float64')).slope
    ax[1].plot(np.log(nbins), np.log(e), '-', label=f"{l} (slope: {slope:.2f})")

ax[0].set_title('Error when computing $df/dx$')
ax[1].set_title('Error when computing $df/d(\log x)$')
for a in ax:
    a.set_xlabel('log(# bins)')
    a.set_ylabel('log(LAE)')
    a.legend()
# plt.savefig('plots/deriv_error.pdf', bbox_inches='tight')
plt.show()

# %%
# Time versus number of bins
from datetime import datetime, timedelta

def get_times(dif, nbins):
    times = []
    for n in nbins:
        xs = np.geomspace(1, 100, n+1)
        ys = gauss(xs, 10, 3)

        trials = []
        for t in range(1000):
            start = datetime.now()
            lin = dif(xs, ys)
            stop = datetime.now()
            trials.append((stop - start) / timedelta(microseconds=1))
        times.append(np.average(trials))
    return times

nbins = np.unique(np.geomspace(12, 200, 100).astype(int))
plt.figure(figsize=(8,6))
for i, l in zip([linear_diff, cubic_diff, spline_diff],
                ['Linear', 'Cubic Lagrange', 'Cubic spline']):
    times = get_times(i, nbins)
    plt.plot(nbins, times, '-', label=l)
    # slope = linregress(np.log(nbins), np.log(times).astype('float64')).slope
    # plt.plot(np.log(nbins), np.log(times), '-', label=f"{l} (slope: {slope:.2f})")
plt.legend()
plt.title('Timing data for differentiation methods')
plt.xlabel('Number of sampling points')
plt.ylabel('Time [ms]')
# plt.savefig('plots/deriv_time.pdf', bbox_inches='tight')
plt.show()

# %%
