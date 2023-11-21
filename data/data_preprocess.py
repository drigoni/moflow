import os
import sys
# for linux env.
sys.path.insert(0,'..')
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import time
from data.data_frame_parser import DataFrameParser
from data.data_loader import NumpyTupleDataset
from data.smile_to_graph import GGNNPreprocessor
import mflow.utils.environment as env
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw


def parse():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_name', type=str, default='qm9',
                        choices=['qm9', 'zinc250k', 'cancer'],
                        help='dataset to be downloaded')
    parser.add_argument('--data_type', type=str, default='relgcn',
                        choices=['gcn', 'relgcn'],)
    args = parser.parse_args()
    return args


def cancer_smiles_csv_to_properties(smiles, eff):
    n = len(smiles)
    f = open('cancer_property.csv', "w")
    f.write('qed,plogp,eff,energy,smile\n')
    results = []
    total = 0
    bad_qed = 0
    bad_plogp = 0
    bad_eff = 0
    bad_energy = 0
    invalid = 0
    for i, (smile, eff) in enumerate(zip(smiles, eff)):
        if i % 10000 == 0:
            print('In {}/{} line'.format(i, n))
        total += 1
        mol = Chem.MolFromSmiles(smile)
        if mol ==  None:
            print("Invalid smile: ", i, smile)
            invalid += 1
            qed = -1
            plogp = -999
            bad_eff = -1
            bad_energy = -999
            results.append((qed, plogp, bad_eff, bad_energy, smile))
            f.write('{},{},{},{},{}\n'.format(qed, plogp, bad_eff, bad_energy, smile))
            continue

        try:
            qed = env.qed(mol)
        except ValueError as e:
            bad_qed += 1
            qed = -1
            print(i + 1, Chem.MolToSmiles(mol, isomericSmiles=True), ' error in qed')

        try:
            plogp = env.penalized_logp(mol)
        except RuntimeError as e:
            bad_plogp += 1
            plogp = -999
            print(i + 1, Chem.MolToSmiles(mol, isomericSmiles=True), ' error in penalized_log')

        try:
            energy = env.calculate_mol_energy(mol)
        except ValueError as e:
            bad_energy += 1
            energy = 10000
            print(i + 1, Chem.MolToSmiles(mol, isomericSmiles=True), ' error in energy')

            
        results.append((qed, plogp, eff, energy, smile))
        f.write('{},{},{},{},{}\n'.format(qed, plogp, eff, energy, smile))
        f.flush()
    f.close()

    results.sort(key=lambda tup: tup[2], reverse=True)
    f = open('cancer_property_sorted_eff.csv', "w")  #
    f.write('qed,plogp,eff,energy,smile\n')
    for r in results:
        qed, plogp, eff, energy,smile = r
        f.write('{},{},{},{},{}\n'.format(qed, plogp, eff, energy, smile))
        f.flush()
    f.close()

    print('Dump done!')
    print('Total: {}\t Invalid: {}\t bad_plogp: {} \t bad_qed: {}\n'.format(total, invalid, bad_plogp, bad_qed))


start_time = time.time()
args = parse()
data_name = args.data_name
data_type = args.data_type
print('args', vars(args))

if data_name == 'qm9':
    max_atoms = 9
elif data_name == 'zinc250k':
    max_atoms = 38
elif data_name == 'cancer':
    # max_atoms = 180  # killed
    # max_atoms = 150  # Preprocess finished. FAIL 2411, SUCCESS 74578, TOTAL 76989
    # max_atoms = 100  # Preprocess finished. FAIL 2490, SUCCESS 74499, TOTAL 76989
    # max_atoms = 80  # Preprocess finished. FAIL 2694, SUCCESS 74295, TOTAL 76989
    max_atoms = 70  # Preprocess finished. FAIL 2894, SUCCESS 74095, TOTAL 76989
    # max_atoms = 50  # Preprocess finished. FAIL 4272, SUCCESS 72717, TOTAL 76989
else:
    raise ValueError("[ERROR] Unexpected value data_name={}".format(data_name))


if data_type == 'relgcn':
    # preprocessor = GGNNPreprocessor(out_size=max_atoms, kekulize=True, return_is_real_node=False)
    preprocessor = GGNNPreprocessor(out_size=max_atoms, kekulize=True)
else:
    raise ValueError("[ERROR] Unexpected value data_type={}".format(data_type))

data_dir = "."
os.makedirs(data_dir, exist_ok=True)

if data_name == 'qm9':
    print('Preprocessing qm9 data:')
    df_qm9 = pd.read_csv('qm9.csv', index_col=0)
    labels = ['A', 'B', 'C', 'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2',
              'zpve', 'U0', 'U', 'H', 'G', 'Cv']
    parser = DataFrameParser(preprocessor, labels=labels, smiles_col='SMILES1')
    result = parser.parse(df_qm9, return_smiles=True)
    dataset = result['dataset']
    smiles = result['smiles']
elif data_name == 'zinc250k':
    print('Preprocessing zinc250k data')
    # dataset = datasets.get_zinc250k(preprocessor)
    df_zinc250k = pd.read_csv('zinc250k.csv', index_col=0)
    # Caution: Not reasonable but used in used in chain_chemistry\datasets\zinc.py:
    # 'smiles' column contains '\n', need to remove it.
    # Here we do not remove \n, because it represents atom N with single bond
    labels = ['logP', 'qed', 'SAS']
    parser = DataFrameParser(preprocessor, labels=labels, smiles_col='smiles')
    result = parser.parse(df_zinc250k, return_smiles=True)
    dataset = result['dataset']
    smiles = result['smiles']
elif data_name == 'cancer':
    print('Preprocessing cancer data')
    df_cancer = pd.read_csv('cancer.csv', index_col=None)
    labels = ['eff']
    parser = DataFrameParser(preprocessor, labels=labels, smiles_col='smiles')
    result = parser.parse(df_cancer, return_smiles=True, return_is_successful=True)
    dataset = result['dataset']
    smiles = result['smiles']
    is_successful = result['is_successful']
    eff = df_cancer.loc[is_successful, 'eff']
    # len after processing with 70 atoms: FAIL 2894, SUCCESS 74095, TOTAL 76989
    # prepare file for property optimization
    cancer_smiles_csv_to_properties(smiles, eff)

else:
    raise ValueError("[ERROR] Unexpected value data_name={}".format(data_name))

NumpyTupleDataset.save(os.path.join(data_dir, '{}_{}_kekulized_ggnp.npz'.format(data_name, data_type)), dataset)
print('Total time:', time.strftime("%H:%M:%S", time.gmtime(time.time()-start_time)) )


