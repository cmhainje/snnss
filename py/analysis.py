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
import numpy as np 
import matplotlib.pyplot as plt


# %%
class Data():
    def __init__(self, fname):
        f = open(fname, 'r')
        lines = f.readlines()
        f.close()

        self.steps, self.times = [], []
        self.N_x, self.N_y = [], []
        self.E_x, self.E_y = [], []
        self.J_x, self.J_y = [], []
        
        for i in range(0, len(lines), 11):
            l = lines[i:i+11]
            
            self.steps.append(int(l[0].strip().split(' ')[0]))
            self.times.append(float(l[0].strip().split(' ')[1]))

            self.N_x.append(np.array([float(x) for x in l[1].strip().split()]))
            self.N_y.append(np.array([float(x) for x in l[2].strip().split()]))

            self.E_x.append(np.array([float(x) for x in l[4].strip().split()]))
            self.E_y.append(np.array([float(x) for x in l[5].strip().split()]))

            self.J_x.append(np.array([float(x) for x in l[7].strip().split()]))
            self.J_y.append(np.array([float(x) for x in l[8].strip().split()]))

        self.steps = np.array(self.steps)
        self.times = np.array(self.times)


# %%
fig, ax = plt.subplots(3, 1, figsize=(15,15))
i = 28

fmt = '.'
n = 100
d = Data(f'../results/cubicdf_lineardI/n{n}_dt1e-6.out')
ax[0].plot(d.N_x[i], 1e19 * d.N_y[i] * d.N_x[i]**2, fmt)
ax[1].plot(d.E_x[i], -1e18 * d.E_y[i] * d.E_x[i]**2, fmt)
ax[2].plot(d.J_x[i], d.J_y[i], fmt)

fmt = 'o'
n = 12
d = Data(f'../results/cubicdf_lineardI/n{n}_dt1e-6.out')
ax[0].plot(d.N_x[i], 1e19 * d.N_y[i] * d.N_x[i]**2, fmt)
ax[1].plot(d.E_x[i], -1e18 * d.E_y[i] * d.E_x[i]**2, fmt)
ax[2].plot(d.J_x[i], d.J_y[i], fmt)

# for a in ax: a.set_xscale('log')
ax[0].set_ylabel(r'$\varepsilon^2 \frac{df}{dt}$', fontsize=14)
ax[1].set_ylabel(r'$I_\nu$', fontsize=14)
ax[2].set_ylabel(r'$f$', fontsize=14)
plt.show()

# %%
fig, ax = plt.subplots(3, 1, figsize=(15,15))
i = 30
nlist = [12]

fmt = '-'
for n in nlist:
    d = Data(f'../results/cubicdf_lineardI/n{n}_dt1e-6.out')
    ax[0].plot(d.N_x[i], 1e19 * d.N_y[i] * d.N_x[i]**2, fmt)
    ax[1].plot(d.E_x[i], -1e18 * d.E_y[i] * d.E_x[i]**2, fmt)
    ax[2].plot(d.J_x[i], d.J_y[i], fmt)

for n in nlist:
    d = Data(f'../results/cubicdf_cubicdI/n{n}_dt1e-6.out')
    ax[0].plot(d.N_x[i], 1e19 * d.N_y[i] * d.N_x[i]**2, fmt)
    ax[1].plot(d.E_x[i], -1e18 * d.E_y[i] * d.E_x[i]**2, fmt)
    ax[2].plot(d.J_x[i], d.J_y[i], fmt)

for n in nlist:
    d = Data(f'../results/splinedf_lineardI/n{n}_dt1e-6.out')
    ax[0].plot(d.N_x[i], 1e19 * d.N_y[i] * d.N_x[i]**2, fmt)
    ax[1].plot(d.E_x[i], -1e18 * d.E_y[i] * d.E_x[i]**2, fmt)
    ax[2].plot(d.J_x[i], d.J_y[i], fmt)

for n in nlist:
    d = Data(f'../results/splinedf_cubicdI/n{n}_dt1e-6.out')
    ax[0].plot(d.N_x[i], 1e19 * d.N_y[i] * d.N_x[i]**2, fmt)
    ax[1].plot(d.E_x[i], -1e18 * d.E_y[i] * d.E_x[i]**2, fmt)
    ax[2].plot(d.J_x[i], d.J_y[i], fmt)

for n in nlist:
    d = Data(f'../results/splineitp_splinedf_lineardI/n{n}_dt1e-6.out')
    ax[0].plot(d.N_x[i], 1e19 * d.N_y[i] * d.N_x[i]**2, fmt)
    ax[1].plot(d.E_x[i], -1e18 * d.E_y[i] * d.E_x[i]**2, fmt)
    ax[2].plot(d.J_x[i], d.J_y[i], fmt)

for a in ax: a.legend(['cubic/linear','cubic/cubic','spline/linear','spline/cubic','spline/linear/spline'])
for a in ax: a.set_xscale('log')
ax[0].set_ylabel(r'$\varepsilon^2 \frac{df}{dt}$', fontsize=14)
ax[1].set_ylabel(r'$I_\nu$', fontsize=14)
ax[2].set_ylabel(r'$f$', fontsize=14)
plt.show()

# %%
fig, ax = plt.subplots(3, 1, figsize=(15,15))
i = 0
nlist = [50, 25, 12]

fmt = 'o'
for n in nlist:
    d = Data(f'../results/splineitp_splinedf_lineardI/n{n}_dt1e-6.out')
    ax[0].plot(d.N_x[i], 1e19 * d.N_y[i] * d.N_x[i]**2, fmt)
    ax[1].plot(d.E_x[i], -1e18 * d.E_y[i] * d.E_x[i]**2, fmt)
    ax[2].plot(d.J_x[i], d.J_y[i], fmt)

fmt = '--'
for n in nlist:
    d = Data(f'../results/splinedf_lineardI/n{n}_dt1e-6.out')
    ax[0].plot(d.N_x[i], 1e19 * d.N_y[i] * d.N_x[i]**2, fmt)
    ax[1].plot(d.E_x[i], -1e18 * d.E_y[i] * d.E_x[i]**2, fmt)
    ax[2].plot(d.J_x[i], d.J_y[i], fmt)

for a in ax: a.legend(nlist)
# for a in ax: a.set_xscale('log')
ax[0].set_ylabel(r'$\varepsilon^2 \frac{df}{dt}$', fontsize=14)
ax[1].set_ylabel(r'$I_\nu$', fontsize=14)
ax[2].set_ylabel(r'$f$', fontsize=14)
plt.show()

# %%
fig, ax = plt.subplots(3, 1, figsize=(15,15))

ilist = list(range(0, 30, 4))
colors = plt.cm.get_cmap('viridis')(np.linspace(0, 1, max(ilist)+1))

fmt = '-'
colors = plt.cm.get_cmap('viridis')(np.linspace(0, 1, max(ilist)+1))
d = Data(f'../results/cubicdf_lineardI/n100_dt1e-6.out')
for i in ilist:
    ax[0].plot(d.N_x[i], 1e19 * d.N_y[i] * d.N_x[i]**2, fmt, color=colors[i])
    ax[1].plot(d.E_x[i], -1e18 * d.E_y[i] * d.E_x[i]**2, fmt, color=colors[i])
    ax[2].plot(d.J_x[i], d.J_y[i], fmt, color=colors[i])

fmt = 'o'
colors = plt.cm.get_cmap('viridis')(np.linspace(0, 1, max(ilist)+1))
d = Data(f'../results/cubicdf_lineardI/n12_dt1e-6.out')
for i in ilist:
    ax[0].plot(d.N_x[i], 1e19 * d.N_y[i] * d.N_x[i]**2, fmt, color=colors[i])
    ax[1].plot(d.E_x[i], -1e18 * d.E_y[i] * d.E_x[i]**2, fmt, color=colors[i])
    ax[2].plot(d.J_x[i], d.J_y[i], fmt, color=colors[i])

# d = Data('../results/test/n100_dt1e-6.out')
# fmt = '--'
# for i in ilist:
#     ax[0].plot(d.N_x[i], 1e19 * d.N_y[i] * d.N_x[i]**2, fmt, color=(0,0,0,0.3))
#     ax[1].plot(d.E_x[i], -1e18 * d.E_y[i] * d.E_x[i]**2, fmt, color=(0,0,0,0.3))
#     ax[2].plot(d.J_x[i], d.J_y[i], fmt, color=(0,0,0,0.3))

# d = Data('../results/test/n12_dt1e-6.out')
# for i in ilist:
#     ax[0].plot(d.N_x[i], 1e19 * d.N_y[i] * d.N_x[i]**2, '--', color=(0,0,0,0.3))
#     ax[1].plot(d.E_x[i], -1e18 * d.E_y[i] * d.E_x[i]**2, '--', color=(0,0,0,0.3))
#     ax[2].plot(d.J_x[i], d.J_y[i], '--', color=(0,0,0,0.3))

for a in ax: a.legend(d.steps[ilist])
# for a in ax: a.set_xscale('log')
ax[0].set_ylabel(r'$\varepsilon^2 \frac{df}{dt}$', fontsize=14)
ax[1].set_ylabel(r'$I_\nu$', fontsize=14)
ax[2].set_ylabel(r'$f$', fontsize=14)
plt.show()

# %%
