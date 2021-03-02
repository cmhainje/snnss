"""
integrator.py
Connor Hainje

Provides a class for running and saving integrations.
"""

from interface import *

import numpy as np
import matplotlib.pyplot as plt
import pickle

from tqdm import tqdm

class Data():
    def __init__(self, kT, rho_N, Y_e, n_bins, e_min=1, e_max=100, dt=1e-6,
                 itp_f="cubic", drv_f="cubic", drv_I="linear"):
        self.kT = kT
        self.rho_N = rho_N
        self.Y_e = Y_e
        self.es = np.geomspace(e_min, e_max, num=n_bins)

        self.n_bins = n_bins
        self.dt = dt

        self.times = []
        self.Jlist = []
        self.Ilist = []
        self.deltaJlist = []
        self.itp_f = itp_f
        self.drv_f = drv_f
        self.drv_I = drv_I

    def step(self, Js, dt):
        return step(self.kT, self.rho_N, self.Y_e, self.es, Js, dt, 
                    self.itp_f, self.drv_f, self.drv_I)

    def plot(self, ax0, ax1, ax2, i, fmt='-', color=None):
        xs = self.es / self.kT
        binw = np.log(xs[1]) - np.log(xs[0])
        edges = np.exp((np.arange(self.n_bins+1) - 0.5) * binw)
        label = self.times[i]
        ax0.plot(xs, 1e19 * self.deltaJlist[i] * xs**2, fmt, color=color, label=f'{label:.3f} s')
        ax1.plot(edges, 1e18 * self.Ilist[i], fmt, color=color, label=f'{label:.3f} s')
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

    def integrate_nsteps(self, n_steps, epoch_size=1000, quiet=False,
                         init=lambda x: 0.5 * np.exp(-0.5 * (x-10)**2/3**2)):
        Js = init(self.es)
        alpha = lib.compute_coeff(self.kT, self.rho_N)
        t = 0
        for s in tqdm(range(n_steps+1), disable=quiet):
            Jout, Inu, _, _ = self.step(Js, self.dt)
            if (s % epoch_size) == 0:
                self.Jlist.append(Jout)
                self.Ilist.append(alpha * Inu)
                self.deltaJlist.append((Jout - Js) / self.dt * T_UNIT)
                self.times.append(t)
            t = self.dt * s
            Js = Jout

            # prevent negatives
            Js[Js < 1e-30] = 1e-30

    def integrate_time(self, time, des_change, epoch_size=1000,
                       init=lambda x: 0.5 * np.exp(-0.5 * (x-10)**2/3**2)):
        Js = init(self.es)
        dt = self.dt
        t = 0
        s = 0
        alpha = compute_coeff(self.kT, self.rho_N)

        dtlist = []

        while t <= time:
            Jout, Inu, _, _ = self.step(Js, dt)

            # Run again with smaller stepsize if there are any negatives
            if np.sum(Jout < 0) > 0:
                max_change = np.amax(np.abs(Jout - Js) / Js)
                dt = np.clip(dt * des_change / max_change, 1e-150, 1e-5)
                Jout, Inu, _, _ = self.step(Js, dt)

            if np.sum(np.isnan(Jout)) != 0:
                print('Killed due to NaNs')
                break

            if (s % epoch_size) == 0:
                self.Jlist.append(Jout)
                self.Ilist.append(alpha * Inu)
                self.deltaJlist.append((Jout - Js) / dt * t_unit)
                self.times.append(t)
                print(f"{len(self.Jlist)} epochs; {t:.3e} s evolved; current step size: {dt:.3e}")
            s += 1
            t += dt
            dtlist.append(dt)

            max_change = np.amax(np.abs(Jout - Js) / Js)
            dt = min(1e-5, dt * des_change / max_change)

            Js = Jout
        return dtlist

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

if __name__ == "__main__":
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
    test.__module__ = "test"
    test.save('test.pickle')
