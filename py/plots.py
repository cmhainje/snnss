# -*- coding: utf-8 -*-
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
# # Plots for the Evaluation section of the write-up
#
# This notebook contains the code used to make all of the plots featured in the Evaluation section (Section VI) of my paper.

# %%
from integrator import *

from scipy.interpolate import interp1d
from scipy.integrate import trapz, simps
from scipy.stats import linregress


# %%
# Some utility functions
def make_plot(d, ilist, title=""):
    rainbow = plt.get_cmap('rainbow')(np.linspace(0, 1, num=len(ilist)))
    fig, ax = plt.subplots(1, 1, figsize=(6,5), tight_layout=True)
    add_plot(ax, d, ilist, title=title, legend=True)
    return fig
    
def add_plot(ax, d, ilist, title="", legend=True, legendMillisec=True, legendFontsize=10, labels=True):
    rainbow = plt.get_cmap('rainbow')(np.linspace(0, 1, num=len(ilist)))
    for i, c in zip(ilist, rainbow):
        if legendMillisec:
            label = f'{d.times[i] * 1e3:.0f} ms'
        else:
            label = f'{d.times[i]:.3f} s'
        ax.plot(d.es, d.Jlist[i], color=c, label=label)
    ax.set_xscale('log')
    if labels:
        ax.set_xlabel('Energy, $\\varepsilon$ [MeV]', fontsize=14)
        ax.set_ylabel('Neutrino distribution function, $f$', fontsize=14)
    if legend: 
        ax.legend(frameon=False, fontsize=legendFontsize)
    ax.set_title(title)
    
def compute_error(xs, ys, ref_f):
    return np.sum(np.abs(ys - ref_f(xs)))


# %% [markdown] heading_collapsed=true
# # Fixed versus adaptive step-size

# %% hidden=true
### FIXED ###
fixed = Data(2.0, 1e11, 0.2, 50, dt=1e-6)
fixed.integrate_nsteps(30_000)

# %% hidden=true
ilist = list(range(0, len(fixed.Jlist), 3)) # take every third epoch
make_plot(fixed, ilist)
# plt.savefig('plots/0p03_fixed.pdf', bbox_inches='tight')
plt.show()

# %% hidden=true
### ADAPTIVE ###
adapt = Data(2.0, 1e11, 0.2, 50)
dtlist = adapt.integrate_time(0.03, 0.01, epoch_size=1000, max_dt_change=1e-9)

# %% hidden=true
ilist = [0, 9, 10, 11, 12, 13, 14, 15]
make_plot(adapt, ilist)
# plt.savefig('plots/0p03_adapt.pdf', bbox_inches='tight')
plt.show()

plt.figure(figsize=(6,5), tight_layout=True)
plt.plot(dtlist)
plt.yscale('log')
plt.xlabel('Step number', fontsize=14)
plt.ylabel('Step size [s]', fontsize=14)
# plt.savefig('plots/0p03_adapt_stepsizes.pdf', bbox_inches='tight')
plt.show()

# %% [markdown] heading_collapsed=true
# # Comparison to Thompson et al.

# %% [markdown] hidden=true
# We compare specifically to Figure 2 of [Thompson et al. 2000](https://arxiv.org/pdf/astro-ph/0003054.pdf).
# This distribution was created and integrated forward in time prior to the creation of the notebook.
# A similar run could be constructed by using 
# ```python
# f = lambda x: 0.8 * np.exp(-0.5 * (x - 40)**2/10**2)
# d = Data(kT=14.51, rho_N=1.02e13, Y_e=0.202, n_bins=100, n_type=1)
# _ = d.integrate_time(1e-3, 0.00005, epoch_size=2500, init=f)
# d.save('pickles/thomp_comp_1.pickle')
# ```
# Note the `n_type=1` argument, which specifies that we only consider neutrino-neutron interactions.

# %% hidden=true
with open("pickles/thomp_comp_1.pickle", "rb") as f:
    d = pickle.load(f)

# %% hidden=true
thomp_times = [0.0, 0.33, 1.0, 3.3, 10.0, 33.0, 1000.0]

def get_snaps(times):
    """Find the snapshot indices whose times 
    most closely match the given times"""
    snap_times = np.array(d.times) * 1e6
    snap_indices = []
    
    for t in times:
        snap_indices.append(np.argmin(np.abs(snap_times - t)))
        
    return snap_indices
    
print(get_snaps(thomp_times))

# %% hidden=true
plt.figure(figsize=(6,6), tight_layout=True)
rainbow = plt.get_cmap('rainbow')(np.linspace(0, 1, len(d.Jlist)))
ilist = get_snaps(thomp_times)

rainbow = plt.get_cmap('rainbow')(np.linspace(0, 1, len(ilist)))
for i, c in zip(ilist, rainbow):
    plt.plot(d.es, d.Jlist[i], color=c)
plt.ylim(0,1)
plt.legend([f"{d.times[i] * 1e6:.2f} µs" for i in ilist], frameon=False)
plt.xlabel('Energy, $\\varepsilon$ [MeV]', fontsize=14)
plt.ylabel('Neutrino distribution function, $f$', fontsize=14)
# plt.savefig('plots/thompson_compare.pdf', bbox_inches='tight')
plt.show()

# %% [markdown]
# # Binning scheme

# %%
### ENERGY FROM 1 TO 100 MeV ###
nbins = [12, 25, 50, 100]
runs = [Data(2.0, 1e11, 0.2, n, dt=1e-6) for n in nbins]
for r in runs:
    r.integrate_nsteps(30_000)

# %%
fig, axs = plt.subplots(2, 2, figsize=(7,7), sharex=True, sharey=True) #, tight_layout=True)
axs = axs.ravel()

for r, ax in zip(runs, axs):
    ilist = list(range(0, len(r.Jlist), 3))
    add_plot(ax, r, ilist, title=f"{r.n_bins} bins", legend=False, labels=False)
    
axs[0].set_ylabel('$f$', fontsize=14)
axs[2].set_ylabel('$f$', fontsize=14)
axs[2].set_xlabel('$\\varepsilon$ [MeV]', fontsize=14)
axs[3].set_xlabel('$\\varepsilon$ [MeV]', fontsize=14)
axs[3].legend(frameon=False, loc='center left', bbox_to_anchor=(1.05, 1.05))

# plt.savefig('plots/solver_nbin_evolution.pdf', bbox_inches='tight')
plt.show()

# %%
fig, ax = plt.subplots(1, 1, figsize=(7,6), tight_layout=True)

ilist = list(range(0, len(runs[0].Jlist), 3))
rainbow = plt.get_cmap('rainbow')(np.linspace(0, 1, num=len(ilist)))

for i, c in zip(ilist, rainbow):
    plt.plot(runs[-1].es, runs[-1].Jlist[i], '-', color=c, label=f"{runs[-1].times[i] * 1e3:.0f} ms")
for i, c in zip(ilist, rainbow):
    plt.plot(runs[0].es, runs[0].Jlist[i], 'o', color=c)

plt.xscale('log')
plt.xlabel('$\\varepsilon$ [MeV]', fontsize=14)
plt.ylabel('$f$', fontsize=14)
plt.legend(frameon=False)
# plt.savefig('plots/evolution_12_vs_100.pdf', bbox_inches='tight')
plt.show()

# %%
### ERROR ANALYSIS ###
plt.figure(figsize=(7,6), tight_layout=True)

ilist = list(range(0, len(runs[-1].Jlist), 3))
rainbow = plt.get_cmap('rainbow')(np.linspace(0, 1, num=len(ilist)-1))

for i, c in zip(ilist[1:], rainbow):
    ref = interp1d(runs[-1].es, runs[-1].Jlist[i])
    ns, errs = [], []
    for r in runs[:-1]:
        lae = compute_error(r.es, r.Jlist[i], ref)
        ns.append(r.n_bins)
        errs.append(lae)
        
    plt.plot(ns, errs, '-', color=c, label=f"{runs[-1].times[i]*1e3:.0f} ms")
        
plt.legend(frameon=False)
plt.xlabel('Number of bins', fontsize=14)
plt.ylabel('LAE', fontsize=14)
plt.xscale('log')
plt.yscale('log')
plt.savefig('plots/error_evolution.pdf', bbox_inches='tight')
plt.show()

# %%
### 1 TO 100 MeV -- ERROR ANALYSIS ###
nbins = [12, 17, 25, 37, 50, 70, 100]
error_runs = [Data(2.0, 1e11, 0.2, n, dt=1e-6) for n in nbins]
for r in error_runs:
    r.integrate_nsteps(30_000, epoch_size=2000)

# %%
### ERROR ANALYSIS ###
plt.figure(figsize=(7,6), tight_layout=True)

ilist = list(range(0, len(error_runs[-1].Jlist), 1))
rainbow = plt.get_cmap('rainbow')(np.linspace(0, 1, num=len(ilist)-1))

for i, c in zip(ilist[1:], rainbow):
    ref = interp1d(error_runs[-1].es, error_runs[-1].Jlist[i])
    ns, errs = [], []
    for r in error_runs[:-1]:
        lae = compute_error(r.es, r.Jlist[i], ref)
        ns.append(r.n_bins)
        errs.append(lae)
        
    plt.plot(np.log10(ns), np.log10(errs), '-', color=c, label=f"{error_runs[-1].times[i]*1e3:.0f} ms")
        
plt.legend(frameon=False)
plt.xlabel('$\log_{10}$ (number of bins)', fontsize=14)
plt.ylabel('$\log_{10}$ (LAE)', fontsize=14)
# plt.savefig('plots/error_evolution_dense.pdf', bbox_inches='tight')
plt.show()

# %%
### ENERGY FROM 1 TO 300 MeV ###
wide_runs = [Data(2.0, 1e11, 0.2, 12, dt=1e-6, e_max=300), Data(2.0, 1e11, 0.2, 100, dt=1e-6, e_max=300)]
for r in wide_runs:
    r.integrate_nsteps(30_000)

# %%
fig, ax = plt.subplots(1, 1, figsize=(7,6), tight_layout=True)

ilist = list(range(0, len(wide_runs[0].Jlist), 3))
rainbow = plt.get_cmap('rainbow')(np.linspace(0, 1, num=len(ilist)))

for i, c in zip(ilist, rainbow):
    plt.plot(wide_runs[-1].es, wide_runs[-1].Jlist[i], '-', color=c, label=f"{wide_runs[-1].times[i] * 1e3:.0f} ms")
for i, c in zip(ilist, rainbow):
    plt.plot(wide_runs[0].es, wide_runs[0].Jlist[i], 'o', color=c)

plt.xscale('log')
plt.xlabel('$\\varepsilon$ [MeV]', fontsize=14)
plt.ylabel('$f$', fontsize=14)
plt.legend(frameon=False)
# plt.savefig('plots/evolution_1_to_300.pdf', bbox_inches='tight')
plt.show()


# %% [markdown] heading_collapsed=true
# # Physical parameters

# %% [markdown] hidden=true
# The data being analyzed in this section was run in separate runs previously and stored as pickles in `pickles/`.

# %% hidden=true
def load(kT, rho_N, Y_e, n_bin):
    filename = f"pickles/n{n_bin}_kT{kT:.0f}_rhoN{rho_N:.0e}_Ye{Y_e:.1f}"
    with open(f"{filename}.pickle", "rb") as f:
        d = pickle.load(f)
    dtlist = np.load(f"{filename}_dtlist.npy")
    return d, dtlist


# %% hidden=true
def plot_vary(which):
    
    if which == 'kT':
        kT_list = [1, 2, 3, 4]
        rho_N_list = [1e11 for kT in kT_list]
        Y_e_list = [0.2 for kT in kT_list]
        
    elif which == 'rho_N':
        rho_N_list = [1e9, 1e10, 1e11, 1e12]
        kT_list = [2 for rho_N in rho_N_list]
        Y_e_list = [0.2 for rho_N in rho_N_list]
        
    elif which == 'Y_e':
        Y_e_list = [0.1, 0.2, 0.3, 0.4]
        kT_list = [2 for Y_e in Y_e_list]
        rho_N_list = [1e11 for Y_e in Y_e_list]
        
    fig, axs = plt.subplots(2, 2, figsize=(8,8), sharex=True, sharey=True)
    axs = axs.ravel()
    
    for kT, rho_N, Y_e, ax in zip(kT_list, rho_N_list, Y_e_list, axs):
        d, dtlist = load(kT, rho_N, Y_e, 50)
        
        ilist = np.unique(np.geomspace(1, len(d.Jlist), 12).astype(int)) - 1
        
        if which == 'kT':
            title=f"$kT = ${kT:.0f} MeV"
        elif which == 'rho_N':
            title=f"$\\rho_N = ${rho_N:.0e} g/cm$^3$"
        elif which == 'Y_e':
            title=f"$Y_e = ${Y_e:.1f}"
        
        add_plot(ax, d, ilist, title=title, legend=True, legendMillisec=False, legendFontsize=8, labels=False)

    axs[0].set_ylabel('$f$', fontsize=14)
    axs[2].set_ylabel('$f$', fontsize=14)
    axs[2].set_xlabel('$\\varepsilon$ [MeV]', fontsize=14)
    axs[3].set_xlabel('$\\varepsilon$ [MeV]', fontsize=14)
    
    return fig

def plot_last(which):
    
    if which == 'kT':
        kT_list = [1, 2, 3, 4]
        rho_N_list = [1e11 for kT in kT_list]
        Y_e_list = [0.2 for kT in kT_list]
        
    elif which == 'rho_N':
        rho_N_list = [1e9, 1e10, 1e11] #, 1e12]
        kT_list = [2 for rho_N in rho_N_list]
        Y_e_list = [0.2 for rho_N in rho_N_list]
        
    elif which == 'Y_e':
        Y_e_list = [0.1, 0.2, 0.3, 0.4]
        kT_list = [2 for Y_e in Y_e_list]
        rho_N_list = [1e11 for Y_e in Y_e_list]
        
    fig = plt.figure(figsize=(7,6), tight_layout=True)
    
    for kT, rho_N, Y_e in zip(kT_list, rho_N_list, Y_e_list):
        d, _ = load(kT, rho_N, Y_e, 50)
        
        if which == 'kT':
            title=f"$kT = ${kT:.0f} MeV"
        elif which == 'rho_N':
            title=f"$\\rho_N = ${rho_N:.0e} g/cm$^3$"
        elif which == 'Y_e':
            title=f"$Y_e = ${Y_e:.1f}"
        title += f" (after {d.times[-1]:.1f} s)"
        
        plt.plot(d.es, d.Jlist[-1], label=title)
        
    plt.xscale('log')
    plt.xlabel('$\\varepsilon$ [MeV]', fontsize=14)
    plt.ylabel('$f$', fontsize=14)
    
    plt.legend(frameon=False)
    
    return fig


# %% hidden=true
plot_vary('Y_e')
# plt.savefig('plots/vary_Y_e.pdf', bbox_inches='tight')
plt.show()

plot_last('Y_e')
plt.show()

# %% hidden=true
plot_vary('rho_N')
# plt.savefig('plots/vary_rho_N.pdf', bbox_inches='tight')
plt.show()

plot_last('rho_N')
plt.show()

# %% hidden=true
plot_vary('kT')
# plt.savefig('plots/vary_kT.pdf', bbox_inches='tight')
plt.show()

plot_last('kT')
# plt.savefig('plots/vary_kT_laststep.pdf', bbox_inches='tight')
plt.show()


# %% [markdown] heading_collapsed=true
# # Neutrino conservation

# %% hidden=true
def compute_num_neutrinos(d, i):
    """Neutrino number is given by the integral of x^2 f"""
    xs = d.es / d.kT
    fs = d.Jlist[i]
    ys = xs**2 * fs
    
    # simple integration
    bin_width = np.log(xs[1]) - np.log(xs[0])
    bin_edges = xs[0] * np.exp((np.arange(len(d.es) + 1) - 0.5) * bin_width)
    bin_sizes = bin_edges[1:] - bin_edges[:-1]
    integral = np.sum(ys * bin_sizes)
    
    # scipy integration methods
    t = trapz(ys, xs)
    s = simps(ys, xs)
    
    return integral, t, s


# %% hidden=true
fig, axs = plt.subplots(2, 2, figsize=(8,8), tight_layout=True, sharex=True, sharey=True)
axs = axs.ravel()

for kT, ax in zip([1, 2, 3, 4], axs):
    d, _ = load(kT, 1e11, 0.2, 50)
    numbers = [compute_num_neutrinos(d, i) for i in range(len(d.Jlist))]
    
    for i, ls in zip([0, 2], ['-', '--']):
        ns = np.array([n[i] for n in numbers])
        ax.plot(d.times, (ns - ns[0]) / ns[0], ls=ls)
    ax.set_title(f'$kT = ${kT:.0f} MeV')

fig.text(0.0, 0.55, 'Fractional deviation in neutrino number', 
         ha='center', va='center', rotation='vertical', fontsize=14)

fig.text(0.55, 0.0, 'Time [s]', 
         ha='center', va='center', fontsize=14)

fig.legend(['Simple', 'Simpson'], frameon=False, loc='lower center', 
           bbox_to_anchor=(1.08,0.45), fontsize=12)

# plt.savefig('plots/neutrino-conservation.pdf', bbox_inches='tight')

plt.show()

# %% hidden=true
fig, axs = plt.subplots(2, 2, figsize=(8,8), tight_layout=True, sharex=True, sharey=True)
axs = axs.ravel()

for rho_N, ax in zip([1e9,1e10,1e11], axs):
    d, _ = load(2.0, rho_N, 0.2, 50)
    numbers = [compute_num_neutrinos(d, i) for i in range(len(d.Jlist))]
    
    for i, ls in zip([0, 2], ['-', '--']):
        ns = np.array([n[i] for n in numbers])
        ax.plot(d.times, (ns - ns[0]) / ns[0], ls=ls)
    ax.set_title(f'$\\rho_N = ${rho_N:.0e} g/cm$^3$')

fig.text(0.0, 0.55, 'Fractional deviation in neutrino number', 
         ha='center', va='center', rotation='vertical', fontsize=14)

fig.text(0.55, 0.0, 'Time [s]', 
         ha='center', va='center', fontsize=14)

fig.legend(['Simple', 'Simpson'], frameon=False, loc='lower center', 
           bbox_to_anchor=(1.08,0.45), fontsize=12)

# plt.savefig('plots/neutrino-cons-rhoN.pdf', bbox_inches='tight')

plt.show()

# %% [markdown]
# # Performance

# %%
nbins = np.geomspace(12, 100, 10, endpoint=True).astype(int)
nbins = [10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,100]
print(nbins)

# %%
kT, rho_N, Y_e, dt = 2.0, 1e11, 0.2, 1e-6

lin_lin = [Data(kT, rho_N, Y_e, n, dt=dt, itp_f='linear', drv_f='linear', drv_I='linear') for n in nbins]
lin_cub = [Data(kT, rho_N, Y_e, n, dt=dt, itp_f='linear', drv_f='linear', drv_I='cubic') for n in nbins]
lin_spl = [Data(kT, rho_N, Y_e, n, dt=dt, itp_f='linear', drv_f='linear', drv_I='spline') for n in nbins]

cub_lin = [Data(kT, rho_N, Y_e, n, dt=dt, itp_f='cubic', drv_f='cubic', drv_I='linear') for n in nbins]
cub_cub = [Data(kT, rho_N, Y_e, n, dt=dt, itp_f='cubic', drv_f='cubic', drv_I='cubic') for n in nbins]
cub_spl = [Data(kT, rho_N, Y_e, n, dt=dt, itp_f='cubic', drv_f='cubic', drv_I='spline') for n in nbins]

# spl_lin = [Data(kT, rho_N, Y_e, n, dt=dt, itp_f='spline', drv_f='spline', drv_I='linear') for n in nbins]
# spl_cub = [Data(kT, rho_N, Y_e, n, dt=dt, itp_f='spline', drv_f='spline', drv_I='cubic') for n in nbins]
# spl_spl = [Data(kT, rho_N, Y_e, n, dt=dt, itp_f='spline', drv_f='spline', drv_I='spline') for n in nbins]

# %%
tqdm._instances.clear()

# %%
n_steps, epoch_size = 1_000, 1_000

init_f = lambda x: 0.5 * np.exp(-0.5 * (x - 10)**2 / 3**2)

for r in tqdm(lin_lin): r.integrate_nsteps(n_steps, epoch_size=epoch_size, init=init_f, quiet=True)
for r in tqdm(lin_cub): r.integrate_nsteps(n_steps, epoch_size=epoch_size, init=init_f, quiet=True)
for r in tqdm(lin_spl): r.integrate_nsteps(n_steps, epoch_size=epoch_size, init=init_f, quiet=True)

for r in tqdm(cub_lin): r.integrate_nsteps(n_steps, epoch_size=epoch_size, init=init_f, quiet=True)
for r in tqdm(cub_cub): r.integrate_nsteps(n_steps, epoch_size=epoch_size, init=init_f, quiet=True)
for r in tqdm(cub_spl): r.integrate_nsteps(n_steps, epoch_size=epoch_size, init=init_f, quiet=True)

# for r in spl_lin: r.integrate_nsteps(n_steps, epoch_size=epoch_size)
# for r in spl_cub: r.integrate_nsteps(n_steps, epoch_size=epoch_size)
# for r in spl_spl: r.integrate_nsteps(n_steps, epoch_size=epoch_size)

# %% [markdown]
# ## Error analysis

# %%
def make_error_plot(ax, runs, title="", fmt='o'):
    ilist = [0,-1]
    # rainbow = plt.get_cmap('rainbow')(np.linspace(0, 1, num=len(ilist)-1))

    for i in ilist[1:]:
        ref = interp1d(cub_lin[-1].es, cub_lin[-1].Jlist[i])
        ns, errs = [], []
        for r in runs[:-1]:
            lae = compute_error(r.es, r.Jlist[i], ref)
            ns.append(r.n_bins)
            errs.append(lae)
        ax.plot(ns, errs, fmt, label=f"{runs[-1].times[i]*1e3:.1f} ms")
    ax.set_title(title)


# %%
fig, axs = plt.subplots(2, 3, figsize=(12,8), sharex=True, sharey=False, tight_layout=True)

make_error_plot(axs[0,0], lin_lin, title="$f$: linear, $I_\\nu$: linear")
make_error_plot(axs[0,1], lin_cub, title="$f$: linear, $I_\\nu$: cubic")
make_error_plot(axs[0,2], lin_spl, title="$f$: linear, $I_\\nu$: spline")

make_error_plot(axs[1,0], cub_lin, title="$f$: cubic, $I_\\nu$: linear")
make_error_plot(axs[1,1], cub_cub, title="$f$: cubic, $I_\\nu$: cubic")
make_error_plot(axs[1,2], cub_spl, title="$f$: cubic, $I_\\nu$: spline")

fig.text(0.35, -0.02, "$\log$(number of bins)", fontsize=14)
fig.text(-0.02, 0.45, "$\log$(LAE) w.r.t. cubic/linear/100", fontsize=14, rotation="vertical")

plt.show()

fig, axs = plt.subplots(2, 1, figsize=(7,10), tight_layout=True, sharex=True, sharey=True)

make_error_plot(axs[0], lin_lin, fmt='-')
make_error_plot(axs[0], lin_cub, fmt='-')
make_error_plot(axs[0], lin_spl, fmt='-')

make_error_plot(axs[1], cub_lin, fmt='-')
make_error_plot(axs[1], cub_cub, fmt='-')
make_error_plot(axs[1], cub_spl, fmt='-')

axs[0].set_title('Linear interp/diff on $f(x)$', fontsize=14)
axs[1].set_title('Cubic Lagrange interp/diff on $f(x)$', fontsize=14)
# axs[0].set_ylabel('$\log$(LAE)', fontsize=14)

axs[1].set_xlabel("Number of bins",fontsize=14)

axs[0].set_xscale('log')
axs[0].set_yscale('log')
axs[1].set_xscale('log')
axs[1].set_yscale('log')

fig.text(-0.02,0.5,"LAE",fontsize=14,rotation="vertical")

fig.legend(['Linear','Cubic Lagrange','Cubic spline'], title="Diff method on $I_\\nu(x)$", frameon=False, fontsize=12, title_fontsize=12, loc='center left', bbox_to_anchor=(1.0,0.5))
# plt.savefig('plots/error_methods.pdf', bbox_inches='tight')
plt.show()

# %% [markdown] heading_collapsed=true
# ## Timing analysis

# %% hidden=true
from datetime import datetime, timedelta

def get_times(nbin_list, nsteps=1000, f_method='cubic', I_method='linear'):
    kT, rho_N, Y_e, dt = 2.0, 1e11, 0.2, 1e-6

    times = []
    for n in tqdm(nbin_list):
        d = Data(kT, rho_N, Y_e, n, dt=dt, itp_f=f_method, drv_f=f_method, drv_I=I_method)
        start = datetime.now()
        d.integrate_nsteps(nsteps, epoch_size=nsteps + 10, quiet=True)
        stop = datetime.now()
        times.append((stop - start) / timedelta(microseconds=1000) / nsteps)

    return np.array(times)


# %% hidden=true
nbins = np.geomspace(12, 100, 15, endpoint=True).astype(int)
print(nbins)

lin_lin = get_times(nbins, f_method='linear', I_method='linear')
lin_cub = get_times(nbins, f_method='linear', I_method='cubic')
lin_spl = get_times(nbins, f_method='linear', I_method='spline')

cub_lin = get_times(nbins, f_method='cubic', I_method='linear')
cub_cub = get_times(nbins, f_method='cubic', I_method='cubic')
cub_spl = get_times(nbins, f_method='cubic', I_method='spline')

# %% hidden=true
from scipy.stats import linregress

fig, axs = plt.subplots(2, 1, figsize=(6,10), sharex=True, sharey=True)

for times, label in zip([lin_lin, lin_cub, lin_spl], ['Linear','Cubic Lagrange','Cubic spline']):
    lr = linregress(nbins, times)
    axs[0].plot(nbins, times, 'o', label=f"{label}\nFit: {lr.slope:.4f} n + {lr.intercept:.4f} (R$^2$ = {lr.rvalue**2:.2f})")

for times, label in zip([cub_lin, cub_cub, cub_spl], ['Linear','Cubic Lagrange','Cubic spline']):
    lr = linregress(nbins, times)
    axs[1].plot(nbins, times, 'o', label=f"{label}\nFit: {lr.slope:.4f} n + {lr.intercept:.4f} (R$^2$ = {lr.rvalue**2:.2f})")

ax[0].set_title('Linear interp/diff on $f(x)$')

axs[0].set_title('Linear interp/diff on $f(x)$', fontsize=14)
axs[1].set_title('Cubic Lagrange interp/diff on $f(x)$', fontsize=14)

axs[1].set_xlabel("Number of bins",fontsize=14)
fig.text(0.02,0.40,"Time per time step [ms]",fontsize=14,rotation="vertical")

axs[0].legend(frameon=False, title="Diff method on $I_\\nu(x)$", title_fontsize=12, loc='center left', bbox_to_anchor=(1.01,0.5))
axs[1].legend(frameon=False, title="Diff method on $I_\\nu(x)$", title_fontsize=12, loc='center left', bbox_to_anchor=(1.01,0.5))

plt.savefig('plots/solver-runtime.pdf', bbox_inches='tight')
plt.show()

# %% [markdown]
# # ODE steppers

# %% [markdown]
# We want to test our three ODE steppers: `euler`, `rk2`, and `rk4`.
# For fixed time steps, we want to analyze the resulting plots, error performance, and runtime.
# For adaptive time steps, we want to compare the resulting distributions and time step sizes.

# %%
eul = Data(2.0, 1e11, 0.2, 50, step='euler')
rk2 = Data(2.0, 1e11, 0.2, 50, step='rk2')
rk4 = Data(2.0, 1e11, 0.2, 50, step='rk4')

for r in [eul, rk2, rk4]:
    r.integrate_nsteps(30_000, epoch_size=3_000)
    make_plot(r, range(len(r.Jlist)))
    plt.show()

# %%
fig, axs = plt.subplots(3, 1, figsize=(6, 12), sharex=True)

add_plot(axs[0], eul, range(len(eul.Jlist)), legend=False, labels=False)
add_plot(axs[1], rk2, range(len(rk2.Jlist)), legend=False, labels=False)
add_plot(axs[2], rk4, range(len(rk4.Jlist)), legend=False, labels=False)

axs[0].set_title("Euler", fontsize=14)
axs[1].set_title("RK2", fontsize=14)
axs[2].set_title("RK4", fontsize=14)

axs[2].set_xlabel('$\\varepsilon$ [MeV]', fontsize=14)
fig.text(0.0, 0.37, 'Neutrino distribution function, $f$', fontsize=14, rotation='vertical')

axs[1].legend(frameon=False, loc='center left', bbox_to_anchor=(1.01,0.5), fontsize=12)
# plt.savefig('plots/ode_methods.pdf', bbox_inches='tight')
plt.show()

# %%

# %%
nbins = [12, 15, 18, 21, 25, 31, 37, 50, 100]

euls = [Data(2.0, 1e11, 0.2, n, step='euler') for n in nbins]
rk2s = [Data(2.0, 1e11, 0.2, n, step='rk2') for n in nbins]
rk4s = [Data(2.0, 1e11, 0.2, n, step='rk4') for n in nbins]

# %%
times = []

for l in [euls, rk2s, rk4s]:
    l_times = []
    for r in tqdm(l):
        start = datetime.now()
        r.integrate_nsteps(1000, epoch_size=200, quiet=True)
        stop = datetime.now()
        l_times.append( (stop-start) / timedelta(microseconds=1000) / 1000 )
    times.append(l_times)


# %%
def make_error_plot(ax, runs, title="", fmt='o'):
    ilist = [0,-1]
    # rainbow = plt.get_cmap('rainbow')(np.linspace(0, 1, num=len(ilist)-1))

    for i in ilist[1:]:
        ref = interp1d(euls[-1].es, euls[-1].Jlist[i])
        ns, errs = [], []
        for r in runs[:-1]:
            lae = compute_error(r.es, r.Jlist[i], ref)
            ns.append(r.n_bins)
            errs.append(lae)
        ax.plot(ns, errs, fmt, label=f"{runs[-1].times[i]*1e3:.1f} ms")
    ax.set_title(title)


# %%
fig, ax = plt.subplots(1, 1, figsize=(7,6))

make_error_plot(ax, euls, fmt='-')
make_error_plot(ax, rk2s, fmt='-')
make_error_plot(ax, rk4s, fmt='-')

plt.xlabel('Number of bins', fontsize=14)
plt.ylabel('LAE', fontsize=14)
plt.legend(['Euler', 'RK2', 'RK4'], frameon=False, fontsize=12)

plt.xscale('log')
plt.yscale('log')

# plt.savefig('plots/ode_errors.pdf', bbox_inches='tight')

plt.show()

# %%
fig, ax = plt.subplots(1, 1, figsize=(6,6))

for ts, l in zip(times, ['Euler', 'RK2', 'RK4']):
    lr = linregress(nbins, ts)
    plt.plot(nbins, ts, 'o', label=f"{l}\nFit: {lr.slope:.4f} n + {lr.intercept:.4f} (R$^2$ = {lr.rvalue**2:.2f})")

plt.legend(frameon=False, title='Time-stepping method', title_fontsize=12, loc='center left', bbox_to_anchor=(1.01, 0.5))

plt.xlabel('Number of bins', fontsize=14)
plt.ylabel('Time per time-step [ms]', fontsize=14)
plt.savefig('plots/ode_runtime.pdf', bbox_inches='tight')
plt.show()

# %%
eul = Data(2.0, 1e11, 0.2, 50, step='euler')
rk2 = Data(2.0, 1e11, 0.2, 50, step='rk2')
rk4 = Data(2.0, 1e11, 0.2, 50, step='rk4')

dts = []

for r in [eul, rk2, rk4]:
    dtlist = r.integrate_time(0.01, 0.005, epoch_size=1500, max_dt_change=1e-9)
    dts.append(dtlist)

# %%
plt.figure(figsize=(6,6), tight_layout=True)
for dt in dts:
    plt.plot(dt)
plt.yscale('log')
plt.xlabel('Step number', fontsize=14)
plt.ylabel('Step size [s]', fontsize=14)
plt.legend(['Euler', 'RK2', 'RK4'], frameon=False, fontsize=12, loc='lower right')
plt.savefig('plots/ode_timestep.pdf', bbox_inches='tight')
plt.show()

fig, axs = plt.subplots(3, 1, figsize=(6, 10), tight_layout=True, sharex=True)
eul_t = np.array(dts[0])[:18220]
rk2_t = np.array(dts[1])
rk4_t = np.array(dts[2])

axs[0].plot((rk2_t - eul_t) / eul_t)
axs[0].set_ylabel('$(dt_{RK2} - dt_{E})/dt_{E}$', fontsize=14)

axs[1].plot((rk4_t - eul_t) / eul_t)
axs[1].set_ylabel('$(dt_{RK4} - dt_{E})/dt_{E}$', fontsize=14)

axs[2].plot((rk4_t - rk2_t) / rk2_t)
axs[2].set_ylabel('$(dt_{RK4} - dt_{RK2})/dt_{RK2}$', fontsize=14)
axs[2].set_xlabel('Step number', fontsize=14)

plt.savefig('plots/ode_timestep_compare.pdf', bbox_inches='tight')
plt.show()

# %%
