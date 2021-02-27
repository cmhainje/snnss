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

# %%
import ctypes
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl

from scipy.stats import linregress
from tqdm import tqdm

lib = ctypes.CDLL("../lib.so")

# %%
atypes = [
    np.ctypeslib.ndpointer(dtype=ctypes.c_longdouble, ndim=1, flags=('C','W')),
    np.ctypeslib.ndpointer(dtype=ctypes.c_longdouble, ndim=1, flags=('C','W')),
    np.ctypeslib.ndpointer(dtype=ctypes.c_longdouble, ndim=1, flags=('C','W')),
    ctypes.c_int
]

lib.linear_itp.argtypes     = atypes
lib.cubic_itp.argtypes      = atypes
lib.spline_itp.argtypes     = atypes
lib.linear_diff.argtypes    = atypes
lib.cubic_diff.argtypes     = atypes
lib.spline_diff.argtypes    = atypes

def use(method, xs, ys):
    n = len(xs) - 1
    out = np.empty(n, dtype=ctypes.c_longdouble)
    method(out, xs.astype(ctypes.c_longdouble), ys.astype(ctypes.c_longdouble), n)
    return out


# %%
atypes = [
    ctypes.c_longdouble,
    ctypes.c_longdouble,
    ctypes.c_longdouble,
    np.ctypeslib.ndpointer(dtype=ctypes.c_longdouble, ndim=1, flags=('C','W')),
    np.ctypeslib.ndpointer(dtype=ctypes.c_longdouble, ndim=1, flags=('C','W')),
    ctypes.c_int,
    ctypes.c_longdouble,
    np.ctypeslib.ndpointer(dtype=ctypes.c_longdouble, ndim=1, flags=('C','W')),
    np.ctypeslib.ndpointer(dtype=ctypes.c_longdouble, ndim=1, flags=('C','W')),
    np.ctypeslib.ndpointer(dtype=ctypes.c_longdouble, ndim=1, flags=('C','W')),
    np.ctypeslib.ndpointer(dtype=ctypes.c_longdouble, ndim=1, flags=('C','W'))
]

lib.advance_step.argtypes = atypes 

def step(kT, rho_N, Y_e, energies, Js, dt):
    """
    Physical parameters
    @param kT        nucleon temperature [MeV]
    @param rho_N     nucleon mass density [g/cm^3]
    @param Y_e       electron fraction
    
    Input
    @param energies  energy bins (zone *centers*) [MeV]
    @param Js        distribution function
    @param n         number of energy zones
    @param dt        time step size [s]
    
    Output
    @param Jout      updated J values [length: n]
    @param I_nu      I_nu on the bin edges [length: n+1]
    @param qdot      energy deposition spectrum [length: n]
    @param Qdot      currently just zeros [length: n]
    """
    n = len(energies)

    Jout = np.empty(n, dtype=ctypes.c_longdouble)
    I_nu = np.empty(n+1, dtype=ctypes.c_longdouble)
    qdot = np.empty(n, dtype=ctypes.c_longdouble)
    Qdot = np.empty(n, dtype=ctypes.c_longdouble)

    lib.advance_step(
        kT, rho_N, Y_e,
        energies.astype(ctypes.c_longdouble),
        Js.astype(ctypes.c_longdouble),
        n, dt,
        Jout, I_nu, qdot, Qdot
    )
    return Jout, I_nu, qdot, Qdot


# %%
class Data():
    def __init__(self, kT, rho_N, Y_e, e_min, e_max, n_bins, dt, des_error=None):
        self.kT = kT
        self.rho_N = rho_N
        self.Y_e = Y_e
        self.es = np.geomspace(e_min, e_max, num=n_bins)

        self.n_bins = n_bins
        self.dt = dt
        self.des_error = des_error

        self.times = []
        self.Jlist = []
        self.Ilist = []
        self.deltaJlist = []

    def plot(self, ax0, ax1, ax2, i, fmt='-', color=None):
        xs = self.es / self.kT
        binw = np.log(xs[1]) - np.log(xs[0])
        edges = np.exp((np.arange(self.n_bins+1) - 0.5) * binw)
        label = self.times[i]
        ax0.plot(xs, self.deltaJlist[i] * xs**2, fmt, color=color, label=f'{label:.3f} s')
        ax1.plot(edges, self.Ilist[i], fmt, color=color, label=f'{label:.3f} s')
        ax2.plot(xs, self.Jlist[i], fmt, color=color, label=f'{label:.3f} s')

    def make_plot(self):
        fig, ax = plt.subplots(3, 1, figsize=(12,12), tight_layout=True)
        colors = plt.cm.get_cmap('viridis')(np.linspace(0, 1, len(self.Jlist)))
        for i in range(len(self.Jlist)):
            self.plot(ax[0], ax[1], ax[2], i, color=colors[i])

        for a in ax: a.legend(title='Time evolved')
        ax[0].set_ylabel(r'$x^2 \,\frac{df}{dt}$', fontsize=14)
        ax[1].set_ylabel(r'$I_\nu$', fontsize=14)
        ax[2].set_ylabel(r'$J$', fontsize=14)
        ax[2].set_xlabel(r'$x$', fontsize=14)

        plt.suptitle(f'$kT$ = {self.kT} MeV, $\\rho_N$ = {self.rho_N:.1e} g/cm$^3$, $Y_e$ = {self.Y_e}')
        return fig

    def integrate_nsteps(self, n_steps, epoch_size=1000):
        Js = init(self.es)
        t = 0
        for s in tqdm(range(n_steps+1)):
            Jout, Inu, _, _ = step(self.kT, self.rho_N, self.Y_e, self.es, Js, self.dt)
            if (s % epoch_size) == 0:
                self.Jlist.append(Jout)
                self.Ilist.append(Inu)
                self.deltaJlist.append(Jout - Js)
                self.times.append(t)
            t = self.dt * s
            Js = Jout

    def integrate_time(self, time, epoch_size=1000, des_change=None):
        Js = init(self.es)
        dt = self.dt
        t = 0
        s = 0

        dtlist = []

        while t <= time:
            Jout, Inu, _, _ = step(self.kT, self.rho_N, self.Y_e, self.es, Js, dt)

            if (s % epoch_size) == 0:
                self.Jlist.append(Jout)
                self.Ilist.append(Inu)
                self.deltaJlist.append(Jout - Js)
                self.times.append(t)
                print(f"{len(self.Jlist)} epochs; {t:.3e} s evolved; current step size: {dt:.3e}")
            s += 1
            t += dt

            if des_change is not None:
                max_change = np.amax(np.abs(Jout - Js) / Js)
                dt = min(1e-6, dt * des_change / max_change)

            Js = Jout
            dtlist.append(dt)
        return dtlist
    
    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)


def init(eps):
    mean = 10
    sigma = 3
    return 0.5 * np.exp(-0.5 * (eps - mean)**2 / sigma**2)


# %% tags=[]
test = Data(
    kT=1.0, 
    rho_N=1e10, 
    Y_e=0.2, 
    e_min=1, 
    e_max=100, 
    n_bins=50, 
    dt=1e-6
)
test.integrate_nsteps(10000, epoch_size=2000)
# dtlist = test.integrate_time(1, epoch_size=1000, des_change=0.2)

# plt.plot(dtlist, '.')
# plt.yscale('log')

# %%
test.make_plot()
plt.show()

# %%
n_bins = 100
n_steps = 100000
dt = 1e-6 # s

kT = 1.0      # MeV
rho_N = 1e10  # g/cm^3
Y_e = 0.2

write_every = 10000

def init(eps):
    mean = 10
    sigma = 3
    return 0.5 * np.exp(-0.5 * (eps - mean)**2 / sigma**2)

es = np.geomspace(1, 100, num=n_bins)
Js = init(es)

Jlist = []
deltaJlist = []
Ilist = []
times = []

t = 0
for s in tqdm(range(n_steps+1)):
    Jout, Inu, _, _ = step(kT, rho_N, Y_e, es, Js, dt)
    if (s % write_every) == 0:
        Jlist.append(Jout)
        Ilist.append(Inu)
        deltaJlist.append(Jout - Js)
        times.append(t)
    t = dt * s
    Js = Jout

# %%
fig, ax = plt.subplots(3, 1, figsize=(12,12), tight_layout=True)
colors = plt.cm.get_cmap('viridis')(np.linspace(0, 1, len(Jlist)))

xs = es / kT
binw = np.log(xs[1]) - np.log(xs[0])
edges = np.exp((np.arange(n_bins+1) - 0.5) * binw)

for i in range(len(Jlist)):
    ax[0].plot(xs, deltaJlist[i] * xs**2, color=colors[i], label=f'{i * write_every * dt:.3f} s')
    ax[1].plot(edges, Ilist[i], color=colors[i], label=f'{i * write_every * dt:.3f} s')
    ax[2].plot(xs, Jlist[i], color=colors[i], label=f'{i * write_every * dt:.3f} s')

for a in ax: a.legend(title='Time evolved')

ax[0].set_ylabel(r'$x^2 \,\frac{df}{dt}$', fontsize=14)
ax[1].set_ylabel(r'$I_\nu$', fontsize=14)
ax[2].set_ylabel(r'$J$', fontsize=14)
ax[2].set_xlabel('$x$', fontsize=14)

plt.suptitle(f'$kT$ = {kT} MeV, $\\rho_N$ = {rho_N:.1e} g/cm^3, $Y_e$ = {Y_e}')
plt.show()
