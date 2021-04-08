from integrator import *
from argparse import ArgumentParser

ap = ArgumentParser()
ap.add_argument('--kT',    type=float)
ap.add_argument('--rho_N', type=float)
ap.add_argument('--Y_e',   type=float)
ap.add_argument('--n_bin', type=int)
ap.add_argument('--time',  type=float)
ap.add_argument('--epoch_size', type=int)
ap.add_argument('--des_change', type=float)
ap.add_argument('--max_change', type=float)
args = ap.parse_args()

print(args)

d = Data(args.kT, args.rho_N, args.Y_e, args.n_bin)
dtlist = d.integrate_time(args.time, args.des_change, 
                          epoch_size=args.epoch_size, 
                          max_dt_change=args.max_change)

filename = f'pickles/n{args.n_bin}_kT{args.kT:.0f}_rhoN{args.rho_N:.0e}_Ye{args.Y_e:.1f}'
d.save(f'{filename}.pickle')
np.save(f'{filename}_dtlist.npy', dtlist)
