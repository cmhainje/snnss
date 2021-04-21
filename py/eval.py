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
xs = np.geomspace(1, 100, 12, endpoint=True)
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
xs = np.geomspace(1, 100, 12, endpoint=True)
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
# Make samples
xs = np.geomspace(1, 100, 12, endpoint=True)
ys = gauss(xs, 10, 3)

# Plot
plt.figure(figsize=(12,8))
xref = np.linspace(1, 100, 500)
plt.plot(xref, gauss(xref, 10, 3), 'k--')
edges = np.sqrt(xs[:-1] * xs[1:])

for itp in [linear_itp, cubic_itp, spline_itp]:
    plt.plot(edges, np.expm1(itp(xs, np.log1p(ys))), 'o')

plt.legend(['Reference', 'Linear', 'Cubic Lagrange', 'Cubic spline'])
plt.show()

# %%
nbins = [12, 25]

f = lambda x: gauss(x, 8, 5)

fig, ax = plt.subplots(len(nbins),1, figsize=(6,6), tight_layout=True)
for a in ax:
    xref = np.linspace(1, 100, 500)
    a.plot(xref, f(xref), 'k--')

for a, n in zip(ax, nbins):
    a.set_title(f'{n} intervals')

for i, n in enumerate(nbins):
    xs = np.geomspace(1, 100, n+1, endpoint=True)
    ys = f(xs)
    edges = np.sqrt(xs[:-1] * xs[1:])

    for itp in [linear_itp, cubic_itp, spline_itp]:
        ax[i].plot(edges, itp(np.log(xs), ys), 'o')

for a in ax:
    a.set_xscale('log')
    a.set_ylabel("$f(x)$", fontsize=12)
    a.legend(['Reference', 'Linear', 'Cubic Lagrange', 'Cubic spline'])
ax[-1].set_xlabel('$x$', fontsize=12)
plt.savefig('plots/itp_methods.pdf', bbox_inches='tight')
plt.show()


fig, ax = plt.subplots(len(nbins),1, figsize=(6,6), tight_layout=True)
for a in ax:
    xref = np.linspace(1, 100, 500)
    a.plot(xref, f(xref), 'k--')

for a, n in zip(ax, nbins):
    a.set_title(f'{n} intervals')

for i, n in enumerate(nbins):
    xs = np.geomspace(1, 100, n+1, endpoint=True)
    ys = f(xs)
    edges = np.sqrt(xs[:-1] * xs[1:])

    for itp in [linear_itp, cubic_itp, spline_itp]:
        ax[i].plot(edges, np.exp(itp(np.log(xs), np.log(ys))), 'o')

for a in ax:
    a.set_xscale('log')
    a.set_ylabel("$f(x)$", fontsize=12)
    a.legend(['Reference', 'Linear', 'Cubic Lagrange', 'Cubic spline'])
ax[-1].set_xlabel('$x$', fontsize=12)
plt.savefig('plots/itp_methods_logexp.pdf', bbox_inches='tight')
plt.show()

# %%
nbins = [8, 16]
func = lambda x: np.sin(x)

fig, ax = plt.subplots(len(nbins),1, figsize=(8,5), tight_layout=True)
for a in ax:
    xref = np.linspace(0, 10, 500)
    a.plot(xref, func(xref), 'k--')

for a, n in zip(ax, nbins):
    a.set_title(f'{n} intervals')

for i, n in enumerate(nbins):
    xs = np.linspace(0, 10, n+1, endpoint=True)
    ys = func(xs)
    edges = 0.5 * (xs[:-1] + xs[1:])

    for itp in [linear_itp, cubic_itp, spline_itp]:
        ax[i].plot(edges, itp(xs, ys), 'o')

for a in ax:
    a.set_ylabel("$\sin(x)$")
    a.legend(['Reference', 'Linear', 'Cubic Lagrange', 'Cubic spline'])
ax[-1].set_xlabel('$x$')
plt.savefig('plots/itp_methods_sine.pdf', bbox_inches='tight')
plt.show()


# %%
# Error versus number of bins
def get_errs(itp, nbins, pos=False, logx=False, log1p=False):
    errs = []
    for n in nbins:
        xs = np.geomspace(1, 100, n, endpoint=True)
        x_itp = np.sqrt(xs[:-1] * xs[1:])
        local_errs = []
        for mu in np.linspace(5, 25, 30, endpoint=True):
            func = lambda x: gauss(x, mu, 3)
            ys = func(xs)
            
            if logx:
                if pos:     lin = np.exp(itp(np.log(xs), np.log(ys)))
                elif log1p: lin = np.expm1(itp(np.log(xs), np.log1p(ys)))
                else:       lin = itp(np.log(xs), ys)
            else:
                if pos:     lin = np.exp(itp(xs, np.log(ys)))
                elif log1p: lin = np.expm1(itp(xs, np.log1p(ys)))
                else:       lin = itp(xs, ys)
                    
            local_errs.append(compute_error(x_itp, lin, func))
        errs.append(np.average(local_errs))
    return errs

nbins = np.unique(np.geomspace(12, 200, 100, endpoint=True).astype(int))

plt.figure(figsize=(6,7), tight_layout=True)

for i, l, c in zip([linear_itp, cubic_itp, spline_itp],
                   ['Linear', 'Cubic Lagrange', 'Cubic spline'],
                   ['C0', 'C1', 'C2']):
    e = get_errs(i, nbins, pos=False, logx=True)
    logn = np.log10(nbins)
    loge = np.log10(e)
    slope = linregress(logn, loge.astype('float64')).slope
    plt.plot(nbins, e, '-', color=c, label=f"{l}\n(log-log slope: {slope:.2f})")

for i, l, c in zip([linear_itp, cubic_itp, spline_itp],
                   ['Linear', 'Cubic Lagrange', 'Cubic spline'],
                   ['C0', 'C1', 'C2']):
    e = get_errs(i, nbins, pos=True, logx=True)
    logn = np.log10(nbins)
    loge = np.log10(e)
    slope = linregress(logn, loge.astype('float64')).slope
    plt.plot(nbins, e, '--', color=c, label=f"{l} (+)\n(log-log slope: {slope:.2f})")
    
plt.xlabel('Number of bins', fontsize=14)
plt.ylabel('LAE', fontsize=14)
plt.xscale('log')
plt.yscale('log')
plt.ylim((-6,0))
plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.01), frameon=False, fontsize=12, ncol=2)
# plt.savefig('plots/itp_error.pdf', bbox_inches='tight')
plt.show()

# %%
# Time versus number of bins
from datetime import datetime, timedelta

def get_times(itp, nbins):
    times = []
    for n in nbins:
        xs = np.geomspace(1, 100, n, endpoint=True)
        ys = gauss(xs, 10, 3)

        trials = []
        for t in range(1000):
            start = datetime.now()
            lin = itp(xs, ys)
            stop = datetime.now()
            trials.append((stop - start) / timedelta(microseconds=1))
        times.append(np.median(trials))
    return times

nbins = np.unique(np.geomspace(10, 100, 100).astype(int))

plt.figure(figsize=(6,7), tight_layout=True)
for i, l in zip([linear_itp, cubic_itp, spline_itp], ['Linear', 'Cubic Lagrange', 'Cubic spline']):
    times = get_times(i, nbins)
    lr = linregress(nbins[2:], times[2:])
    plt.plot(nbins[2:], times[2:], '-', label=f"{l}\nFit: {lr.slope:.2f}$n$ + {lr.intercept:.2f} (R$^2$ = {lr.rvalue**2:.2f})")
plt.legend(fontsize=11, frameon=False, loc='lower center', bbox_to_anchor=(0.5,1.01))
plt.xlabel('Number of sampling points', fontsize=14)
plt.ylabel('Time [ms]', fontsize=14)
plt.savefig('plots/itp_time.pdf', bbox_inches='tight')
plt.show()

# %% [markdown]
# # Differentiators

# %%
# Make samples
xs = np.geomspace(1, 100, 25, endpoint=True)
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
nbins = [12, 25]
f = lambda x: gauss(x, 8, 5)
df = lambda x: dgauss(x, 8, 5) * x

fig, ax = plt.subplots(len(nbins),1, figsize=(6,6), tight_layout=True)
for a in ax:
    xref = np.linspace(1, 100, 500)
    a.plot(xref, df(xref), 'k--')

for a, n in zip(ax, nbins):
    a.set_title(f'{n} intervals')

for i, n in enumerate(nbins):
    xs = np.geomspace(1, 100, n+1, endpoint=True)
    ys = f(xs)
    edges = np.sqrt(xs[:-1] * xs[1:])

    for dif in [linear_diff, cubic_diff, spline_diff]:
        lin = dif(np.log(xs), ys)
        ax[i].plot(edges, lin, 'o')

for a in ax:
    a.set_xscale('log')
    a.legend(['Reference', 'Linear', 'Cubic Lagrange', 'Cubic spline'])
    a.set_ylabel("$df\,/\,d(\log x)$", fontsize=12)
ax[-1].set_xlabel('$x$', fontsize=12)
plt.savefig('plots/deriv_methods.pdf', bbox_inches='tight')
plt.show()

# %%
nbins = [8, 16]

func = lambda x: np.sin(x)
dfunc = lambda x: np.cos(x)


fig, ax = plt.subplots(len(nbins),1, figsize=(8,5), tight_layout=True)
for a in ax:
    xref = np.linspace(0, 10, 500)
    a.plot(xref, dfunc(xref), 'k--')

for a, n in zip(ax, nbins):
    a.set_title(f'{n} intervals')

for i, n in enumerate(nbins):
    xs = np.linspace(0, 10, n+1, endpoint=True)
    ys = func(xs)
    edges = 0.5 * (xs[:-1] + xs[1:])

    for dif in [linear_diff, cubic_diff, spline_diff]:
        ax[i].plot(edges, dif(xs, ys), 'o')

for a in ax:
    a.set_ylabel(r"$\frac{d\sin(x)}{dx}$")
    a.legend(['Reference', 'Linear', 'Cubic Lagrange', 'Cubic spline'])
ax[-1].set_xlabel('$x$')
plt.savefig('plots/drv_methods_sine.pdf', bbox_inches='tight')
plt.show()


# %%
# Error versus number of bins
def get_errs(dif, nbins, logx=False):
    
    errs = []
    for n in nbins:
        xs = np.geomspace(1, 100, n, endpoint=True)
        x_itp = np.sqrt(xs[:-1] * xs[1:])
        local_errs = []
        for mu in np.linspace(5, 25, 50, endpoint=True):
            func = lambda x: gauss(x, mu, 5)
            dfunc = lambda x: dgauss(x, mu, 5)
            ys = func(xs)
            if logx:
                lin = dif(np.log(xs), ys)
                dgauss_dlogx = lambda x: x * dfunc(x)
                local_errs.append(compute_error(x_itp, lin, dgauss_dlogx))
            else:
                lin = dif(xs, ys)
                local_errs.append(compute_error(x_itp, lin, dfunc))
        errs.append(np.average(local_errs))
    return errs

nbins = np.unique(np.geomspace(12, 200, 100, endpoint=True).astype(int))

# fig, ax = plt.subplots(2, 1, figsize=(7,12))
# for d, l in zip([linear_diff, cubic_diff, spline_diff],
#                 ['Linear', 'Cubic Lagrange', 'Cubic spline']):
#     e = get_errs(d, nbins)
#     slope = linregress(np.log10(nbins), np.log10(e).astype('float64')).slope
#     ax[0].plot(np.log10(nbins), np.log10(e), '-', label=f"{l} (slope: {slope:.2f})")
    
#     e = get_errs(d, nbins, logx=True)
#     slope = linregress(np.log10(nbins), np.log10(e).astype('float64')).slope
#     ax[1].plot(np.log10(nbins), np.log10(e), '-', label=f"{l} (slope: {slope:.2f})")

# ax[0].set_title('Error when computing $df/dx$')
# ax[1].set_title('Error when computing $df/d(\log x)$')
# for a in ax:
#     a.set_xlabel('log$_{10}$(# bins)', fontsize=12)
#     a.set_ylabel('log$_{10}$(LAE)', fontsize=12)
#     a.legend()
# plt.savefig('plots/deriv_error.pdf', bbox_inches='tight')
# plt.show()

plt.figure(figsize=(6,7), tight_layout=True)
for d, l in zip([linear_diff, cubic_diff, spline_diff],
                ['Linear', 'Cubic Lagrange', 'Cubic spline']):
    e = get_errs(d, nbins, logx=True)
    slope = linregress(np.log10(nbins), np.log10(e).astype('float64')).slope
    plt.plot(nbins, e, '-', label=f"{l} (slope: {slope:.2f})")
    
plt.xlabel('Number of bins', fontsize=14)
plt.ylabel('LAE', fontsize=14)
plt.xscale('log')
plt.yscale('log')
plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.01), frameon=False, fontsize=12)
# plt.savefig('plots/deriv_error.pdf', bbox_inches='tight')
plt.show()

# %%
# Time versus number of bins
from datetime import datetime, timedelta

def get_times(dif, nbins):
    times = []
    for n in nbins:
        xs = np.geomspace(1, 100, n+1, endpoint=True)
        ys = gauss(xs, 10, 3)

        trials = []
        for t in range(1000):
            start = datetime.now()
            lin = dif(xs, ys)
            stop = datetime.now()
            trials.append((stop - start) / timedelta(microseconds=1))
        times.append(np.median(trials))
    return times

nbins = np.unique(np.geomspace(10, 100, 100, endpoint=True).astype(int))
plt.figure(figsize=(6,7), tight_layout=True)
for i, l in zip([linear_diff, cubic_diff, spline_diff],
                ['Linear', 'Cubic Lagrange', 'Cubic spline']):
    times = get_times(i, nbins)
    lr = linregress(nbins[2:], times[2:])
    plt.plot(nbins[2:], times[2:], '-', label=f"{l}\nFit: {lr.slope:.2f}$n$ + {lr.intercept:.2f} (R$^2$ = {lr.rvalue**2:.2f})")
plt.legend(fontsize=11, frameon=False, loc='lower center', bbox_to_anchor=(0.5,1.01))
plt.xlabel('Number of sampling points', fontsize=14)
plt.ylabel('Time [ms]', fontsize=14)
plt.savefig('plots/deriv_time.pdf', bbox_inches='tight')
plt.show()
