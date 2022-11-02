import numpy as np
from scipy import spatial
import os.path
import argparse
import ROOT
import tqdm
import h5py
import pandas as pd


class SKGeom:
    def __init__(self, path, posname, tubename):
        self.geom = np.load(path)
        self.tree = spatial.KDTree(self.geom[posname])
        self.posname = posname
        self.tubename = tubename

    def get_id(self, pmt_pos):
        distances, indices = self.tree.query(pmt_pos)
        return self.geom[self.tubename][indices]

    def get_id_lin(self, pmt_pos):
        return self.geom[self.tubename][np.linalg.norm(self.geom[self.posname] - pmt_pos, axis=1).argmin()]

    def get_n_pmts(self):
        return len(self.geom[self.tubename])


def read(filename, geo):
    rootfile = ROOT.TFile(filename)
    tree = rootfile.Get("T")
    branches_name = ['T', 'Q', 'X', 'Y', 'Z']
    buf = {f'{name}': ROOT.std.vector('double')(1) for name in branches_name}
    for name in buf:
        tree.SetBranchAddress(name, buf[name])

    nevts = int(tree.GetEntries())
    e_idx = []
    PMT_Pos = []
    PMT_ID = []
    PMT_T = []
    PMT_Q = []
    for i in tqdm.tqdm(range(nevts)):
        tree.GetEntry(i)
        nhits = int(buf['T'].size())
        e_idx = np.append(e_idx, nhits)
        PMT_Pos.append(np.c_[np.asarray(buf['X']),
                             np.asarray(buf['Y']),
                             np.asarray(buf['Z'])])
        PMT_ID = np.append(PMT_ID, geo.get_id(PMT_Pos[-1]))
        PMT_T = np.append(PMT_T, np.asarray(buf['T']))
        PMT_Q = np.append(PMT_Q, np.asarray(buf['Q']))

    return e_idx, PMT_Pos, PMT_ID, PMT_T, PMT_Q


def create_h5(filename, e_idx, PMT_ID, PMT_T, PMT_Q):
    f = h5py.File(filename, "w")
    n_events = len(np.asarray(e_idx)[:-1])
    n_hits = np.asarray(e_idx).cumsum()[-1]

    f.create_dataset("e_idx", (n_events,), dtype='i',
                     data=np.asarray(e_idx).cumsum()[:-1])

    f.create_dataset("hit_pmt", shape=(n_hits,), dtype='i',
                     data=np.concatenate(PMT_ID))
    f.create_dataset("hit_time", shape=(n_hits,), dtype='f',
                     data=np.concatenate(PMT_T))
    f.create_dataset("hit_charge", shape=(n_hits,), dtype='f',
                     data=np.concatenate(PMT_Q))


def create_df(e_idx, PMT_Pos, PMT_ID, PMT_T, PMT_Q):
    df_idx = []
    for i, idx in enumerate(e_idx):
        df_idx = np.append(df_idx, [i] * int(idx))
    return pd.DataFrame(np.c_[df_idx, np.concatenate(PMT_Pos), PMT_ID, PMT_T, PMT_Q],
                        columns=['EID', 'X', 'Y', 'Z', 'ID', 'T', 'Q'])


class Particle:
    def __init__(self, pos, dir, e):
        self.vtx_pos = pos
        self.vtx_dir = dir
        self.E = e

    def __repr__(self):
        return f'{self.vtx_pos} {self.vtx_dir} {self.E}'

    def __str__(self):
        return f'{self.vtx_pos} {self.vtx_dir} {self.E}'


def read_kin(path):
    def chunker(seq, size):
        return (seq[pos:pos + size] for pos in range(0, len(seq), size))

    particles = []
    with open(path) as file:
        lines = file.readlines()
        for group in chunker(lines, 4):
            vtx_pos = [float(_) for _ in group[1].split()[2:5]]
            E = float(group[2].split()[3])
            vtx_dir = [float(_) for _ in group[2].split()[4:7]]
            particles.append(Particle(vtx_pos, vtx_dir, E))
    return particles


def get_full_df(hitpath, vecpath, geompath='geom/SK_geo.npz'):
    geom = SKGeom(geompath, 'position', 'tube_no')
    event_hits_index, hit_pos, hit_pmt, hit_time, hit_charge = \
        read(hitpath, geom)
    df = create_df(event_hits_index, hit_pos, hit_pmt, hit_time, hit_charge)
    df['EID'] = df['EID'].astype(int)
    df['ID']  = df['ID'].astype(int)

    particles = read_kin(vecpath)
    distances = []
    for i, (EID, evt) in enumerate(df.groupby('EID')):
        d = evt[['X', 'Y', 'Z']].to_numpy() - np.asarray(particles[i].vtx_pos)
        distances.append(d)

    df['DT'] = df.groupby('EID', group_keys=False).apply(
        lambda x: (x['T'] - x['T'].shift()).cumsum().fillna(0))
    df['VD'] = np.sqrt(np.square(np.concatenate(distances)).sum(axis=1))
    return df


def parse():
    parser = argparse.ArgumentParser(
        description='Convert SK ROOT STL vectors of PMTs hits, T, Q to h5'
    )
    parser.add_argument('file', type=str)
    parser.add_argument('-o', type=str, dest='out', default='')
    return parser.parse_args()


if __name__ == "__main__":
    package_directory = os.path.dirname(os.path.abspath(__file__))
    geom_file = os.path.join(package_directory, 'geom', 'SK_geo.npz')
    geom = SKGeom(geom_file, 'position', 'tube_no')

    ###
    args = parse()

    if not os.path.isfile(args.file):
        print(f'{args.file} does not exist')
        exit(1)

    ###
    event_hits_index, hit_pos, hit_pmt, hit_time, hit_charge = read(args.file, geom)

    ###
    out = args.out if args.out else os.path.splitext(args.file)[0] + '.hdf5'
    create_h5(out, event_hits_index, hit_pmt, hit_time, hit_charge)
