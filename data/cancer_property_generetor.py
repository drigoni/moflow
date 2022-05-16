import os
import sys
# for linux env.
sys.path.insert(0,'..')
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import time
import tqdm
import mflow.utils.environment as env
from rdkit import Chem
from rdkit.Chem import AllChem


def parse():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_name', type=str, default='cancer',
                        choices=['cancer'],
                        help='dataset to be use')
    args = parser.parse_args()
    return args


start_time = time.time()
args = parse()
data_name = args.data_name
print('args', vars(args))
data_dir = "."
os.makedirs(data_dir, exist_ok=True)

if data_name == 'cancer':
    print('Preprocessing cancer data')
    df_cancer = pd.read_csv('cancer.csv', index_col=None)
    cancer_dataset = df_cancer.to_numpy()
else:
    raise ValueError("[ERROR] Unexpected value data_name={}".format(data_name))

# transform to molecule structure
smiles = cancer_dataset[:, 0]
eff = cancer_dataset[:, 1]
mols = []
for tmp in smiles:
    mm = Chem.MolFromSmiles(tmp)
    mols.append(mm)
assert len(smiles) == len(eff) == len(mols)

# calculate other properties
qed = []
plogp = []
energy = []
wt = []
charge = []
n_atoms = []
tmp_count = 0
for idx in tqdm.tqdm(range(len(smiles))):
    smi = smiles[idx]
    mol = mols[idx]
    # print("Processing molecule: {} .".format(smi))
    if mol is not None:
        qed.append(env.qed(mol))
        plogp.append(env.penalized_logp(mol))
        tmp_energy = env.calculate_mol_energy(mol)
        energy.append(tmp_energy)
        if tmp_energy is None:
            tmp_count += 1
        wt.append(env.calculate_mol_wt(mol))
        charge.append(env.calculate_mol_charge(mol))
        n_atoms.append(mol.GetNumAtoms())
    else:
        qed.append(None)
        plogp.append(None)
        energy.append(None)
        wt.append(None)
        charge.append(None)
        n_atoms.append(None)

print("Energy non found for {}/{} molecules. ".format(tmp_count, len(energy)))  # Energy non found for 399/76989 molecules.

# make final csv
cancer_properties_dataframe = pd.DataFrame({
    "qed": qed,
    "plogp": plogp,
    "energy": energy,
    "wt": wt,
    "charge": charge,
    "eff": eff,
    "n_atoms": n_atoms,
    "smiles": smiles
})
cancer_properties_dataframe.to_csv("./cancer_property.csv", index=False)

# print final time requested
print('Total time:', time.strftime("%H:%M:%S", time.gmtime(time.time()-start_time)) )

# TODO: print
# for i in range(0, 210, 10):
#     tmp = cancer_properties_dataframe.loc[cancer_properties_dataframe['n_atoms'] > i]
#     tmp_list = []
#     for j in range(0, 5):
#         score = float(j)/10
#         tmp_list.append((tmp.loc[tmp['eff'] > score]).shape[0])
#     print("Number of atoms {}:  {} .".format(i, tmp_list))
#
# Number of atoms 0:  [6603, 6333, 1121, 221, 1] .
# Number of atoms 10:  [6525, 6256, 1105, 218, 1] .
# Number of atoms 20:  [4879, 4658, 829, 162, 1] .
# Number of atoms 30:  [1954, 1835, 344, 74, 1] .
# Number of atoms 40:  [647, 602, 129, 33, 1] .
# Number of atoms 50:  [261, 233, 58, 19, 1] .
# Number of atoms 60:  [141, 126, 36, 11, 0] .
# Number of atoms 70:  [76, 65, 15, 3, 0] .
# Number of atoms 80:  [50, 42, 11, 2, 0] .
# Number of atoms 90:  [32, 25, 7, 1, 0] .
# Number of atoms 100:  [18, 14, 3, 0, 0] .
# Number of atoms 110:  [10, 9, 2, 0, 0] .
# Number of atoms 120:  [7, 6, 1, 0, 0] .
# Number of atoms 130:  [6, 5, 1, 0, 0] .
# Number of atoms 140:  [5, 4, 1, 0, 0] .
# Number of atoms 150:  [3, 2, 0, 0, 0] .
# Number of atoms 160:  [3, 2, 0, 0, 0] .
# Number of atoms 170:  [3, 2, 0, 0, 0] .
# Number of atoms 180:  [3, 2, 0, 0, 0] .
# Number of atoms 190:  [2, 1, 0, 0, 0] .
# Number of atoms 200:  [1, 1, 0, 0, 0] .