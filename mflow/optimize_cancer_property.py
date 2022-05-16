import argparse
import os
import sys
# for linux env.
sys.path.insert(0,'..')
from distutils.util import strtobool

import pickle
import torch

import numpy as np

from data.data_loader import NumpyTupleDataset
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw
import torch
import torch.nn as nn
import torch.nn.functional as F
from data import transform_cancer
from data.transform_cancer import cancer_atomic_num_list, transform_fn_cancer
# from mflow.generate import generate_mols_along_axis
from mflow.models.hyperparams import Hyperparameters
from mflow.models.utils import check_cancer_validity, construct_mol, adj_to_smiles
from mflow.utils.model_utils import load_model, get_latent_vec, smiles_to_adj
from mflow.utils.molecular_metrics import MolecularMetrics
from mflow.models.model import MoFlow, rescale_adj
from mflow.utils.timereport import TimeReport
import mflow.utils.environment as env
from sklearn.linear_model import LinearRegression
import time
import functools
print = functools.partial(print, flush=True)


class MoFlowProp(nn.Module):
    def __init__(self, model:MoFlow, hidden_size):
        super(MoFlowProp, self).__init__()
        self.model = model
        self.latent_size = model.b_size + model.a_size
        self.hidden_size = hidden_size

        vh = (self.latent_size,) + tuple(hidden_size) + (1,)
        modules = []
        for i in range(len(vh)-1):
            modules.append(nn.Linear(vh[i], vh[i+1]))
            if i < len(vh) - 2:
                # modules.append(nn.Tanh())
                modules.append(nn.ReLU())
        self.propNN = nn.Sequential(*modules)

    def encode(self, adj, x):
        with torch.no_grad():
            self.model.eval()
            adj_normalized = rescale_adj(adj).to(adj)
            z, sum_log_det_jacs = self.model(adj, x, adj_normalized)  # z = [h, adj_h]
            h = torch.cat([z[0].reshape(z[0].shape[0], -1), z[1].reshape(z[1].shape[0], -1)], dim=1)
        return h, sum_log_det_jacs

    def reverse(self, z):
        with torch.no_grad():
            self.model.eval()
            adj, x = self.model.reverse(z, true_adj=None)
        return adj, x

    def forward(self, adj, x):
        h, sum_log_det_jacs = self.encode(adj, x)
        output = self.propNN(h)  # do I need to add nll of the unsupervised part? or just keep few epoch? see the results
        return output, h,  sum_log_det_jacs


def fit_model(model, atomic_num_list, data, data_prop,  device, property_name='eff',
              max_epochs=10, learning_rate=1e-3, weight_decay=1e-5):
    start = time.time()
    print("Start at Time: {}".format(time.ctime()))
    model = model.to(device)
    model.train()
    # Loss and optimizer
    metrics = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    N = len(data.dataset)
    assert len(data_prop) == N
    iter_per_epoch = len(data)
    log_step = 20
    # batch_size = data.batch_size
    tr = TimeReport(total_iter = max_epochs * iter_per_epoch)
    if property_name == 'eff':
        col = 0
    elif property_name == 'energy':
        col = 1
    else:
        raise ValueError("Wrong property_name {}".format(property_name))

    for epoch in range(max_epochs):
        print("In epoch {}, Time: {}".format(epoch + 1, time.ctime()))
        for i, batch in enumerate(data):
            x = batch[0].to(device)   # (bs,9,5)
            adj = batch[1].to(device)   # (bs,4,9, 9)
            bs = x.shape[0]

            ps = i * bs
            pe = min((i+1)*bs, N)
            true_y = [[tt[col]] for tt in data_prop[ps:pe]]  #[[propf(mol)] for mol in true_mols]
            true_y = torch.tensor(true_y).float().cuda()
            # model and loss
            optimizer.zero_grad()

            y, z, sum_log_det_jacs = model(adj, x)

            loss = metrics(y, true_y)
            loss.backward()
            optimizer.step()
            tr.update()
            # Print log info
            if (i + 1) % log_step == 0:  # i % args.log_step == 0:
                print('Epoch [{}/{}], Iter [{}/{}], loss: {:.5f}, {:.2f} sec/iter, {:.2f} iters/sec: '.
                      format(epoch + 1, args.max_epochs, i + 1, iter_per_epoch,
                             loss.item(),
                             tr.get_avg_time_per_iter(), tr.get_avg_iter_per_sec()))
                tr.print_summary()
    tr.print_summary()
    tr.end()
    print("[fit_model Ends], Start at {}, End at {}, Total {}".
          format(time.ctime(start), time.ctime(), time.time()-start))
    return model

def _normalize(df, col_name):
    if col_name == 'energy':
        df[col_name] = df[col_name].replace('None', 1000)
        # df[col_name] = df[col_name].fillna(1000)
        df[col_name] = pd.to_numeric(df[col_name])
        df[col_name] = df[col_name].clip(lower=0, upper=1000)
        m = df[col_name].mean()
        mn = df[col_name].min()
        mx = df[col_name].max()
        df[col_name] = (df[col_name] - mn) / (mx-mn)
        return df
    elif col_name == 'eff':
        df[col_name] = df[col_name].replace('None', 0)
        # df[col_name] = df[col_name].fillna(0)
        df[col_name] = pd.to_numeric(df[col_name])
        m = df[col_name].mean()
        mn = df[col_name].min()
        mx = df[col_name].max()
        df[col_name] = (df[col_name] - mn) / (mx-mn)
        return df

def load_property_csv(normalize=True):
    filename = '../data/cancer_property.csv'
    df = pd.read_csv(filename)  # qed, plogp, eff, energy, smile
    df = df.drop("qed", axis=1)
    df = df.drop("plogp", axis=1)
    if normalize:
        df = _normalize(df, 'eff')
        df = _normalize(df, 'energy')

    tuples = [tuple(x) for x in df.values]
    print('Load {} done, length: {}'.format(filename, len(tuples)))
    return tuples

def optimize_mol(model:MoFlow, property_model:MoFlowProp, smiles, device, sim_cutoff, lr=2.0, num_iter=20,
             data_name='cancer', atomic_num_list=[6, 7, 8, 9, 0], property_name='eff', debug=True, random=False):
    if property_name == 'eff':
        # todo drigoni # [0, 1]
        propf = lambda: 1000
    elif property_name == 'energy':
        propf = env.calculate_mol_energy  # unbounded, normalized later???
    else:
        raise ValueError("Wrong property_name{}".format(property_name))
    model.eval()
    property_model.eval()
    with torch.no_grad():
        bond, atoms = smiles_to_adj(smiles, data_name)
        bond = bond.to(device)
        atoms = atoms.to(device)
        mol_vec, sum_log_det_jacs = property_model.encode(bond, atoms)
        if debug:
            adj_rev, x_rev = property_model.reverse(mol_vec)
            reverse_smiles = adj_to_smiles(adj_rev.cpu(), x_rev.cpu(), atomic_num_list)
            print(smiles, reverse_smiles)

            adj_normalized = rescale_adj(bond).to(device)
            z, sum_log_det_jacs = model(bond, atoms, adj_normalized)
            z0 = z[0].reshape(z[0].shape[0], -1)
            z1 = z[1].reshape(z[1].shape[0], -1)
            adj_rev, x_rev = model.reverse(torch.cat([z0, z1], dim=1))
            # val_res = check_validity(adj_rev, x_rev, atomic_num_list)
            reverse_smiles2 = adj_to_smiles(adj_rev.cpu(), x_rev.cpu(), atomic_num_list)
            train_smiles2 = adj_to_smiles(bond.cpu(), atoms.cpu(), atomic_num_list)
            print(train_smiles2, reverse_smiles2)

    mol = Chem.MolFromSmiles(smiles)
    fp1 = AllChem.GetMorganFingerprint(mol, 2)
    try:
        start_score = propf(mol)
    except:
        start_score = 1000
    start = (smiles, start_score, None) # , mol)

    cur_vec = mol_vec.clone().detach().requires_grad_(True).to(device)  # torch.tensor(mol_vec, requires_grad=True).to(mol_vec)
    start_vec = mol_vec.clone().detach().requires_grad_(True).to(device)

    visited = []
    visited_prop = []
    for step in range(num_iter):
        prop_val = property_model.propNN(cur_vec).squeeze()
        grad = torch.autograd.grad(prop_val, cur_vec)[0]
        if property_name == 'energy':
            grad = -grad
        # cur_vec = cur_vec.data + lr * grad.data
        if random:
            rad = torch.randn_like(cur_vec.data)
            cur_vec = start_vec.data + lr * rad / torch.sqrt(rad * rad)
        else:
            cur_vec = cur_vec.data + lr * grad.data / torch.sqrt(grad.data * grad.data)
        cur_vec = cur_vec.clone().detach().requires_grad_(True).to(device)  # torch.tensor(cur_vec, requires_grad=True).to(mol_vec)
        visited.append(cur_vec)
        prop_val = property_model.propNN(cur_vec).squeeze()
        visited_prop.append(prop_val)

    hidden_z = torch.cat(visited, dim=0).to(device)
    visited_prop_cat = torch.FloatTensor(visited_prop).to(device)
    adj, x = property_model.reverse(hidden_z)
    val_res = check_cancer_validity(adj, x, visited_prop_cat, atomic_num_list, debug=debug)
    valid_mols = val_res['valid_mols']
    valid_smiles = val_res['valid_smiles']
    valid_properties = val_res['valid_properties']
    results = []
    sm_set = set()
    sm_set.add(smiles)
    for m, s, s_prop in zip(valid_mols, valid_smiles, valid_properties):
        if s in sm_set:
            continue
        sm_set.add(s)
        try:
            p = propf(m)
        except:
            p = 1000
        fp2 = AllChem.GetMorganFingerprint(m, 2)
        sim = DataStructs.TanimotoSimilarity(fp1, fp2)
        if sim >= sim_cutoff:
            results.append((s, p, sim, smiles, s_prop))
    # smile, property, similarity, mol
    results.sort(key=lambda tup: tup[1], reverse=True)
    return results, start

def find_top_score_smiles(model, device, data_name, property_name, train_prop, topk, atomic_num_list, debug, model_dir):
    start_time = time.time()
    if property_name == 'eff':
        col = 0
    elif property_name == 'energy':
        col = 1
    print('Finding top {} score'.format(property_name))
    train_prop_sorted = sorted(train_prop, key=lambda tup: tup[col], reverse=True)  # qed, plogp, smile
    result_list = []
    for i, r in enumerate(train_prop_sorted):
        if i >= topk:
            break
        if i % 50 == 0:
            print('Optimization {}/{}, time: {:.2f} seconds'.format(i, topk, time.time() - start_time))
        qed, plogp, smile = r
        results, ori = optimize_mol(model, property_model, smile, device, sim_cutoff=0, lr=.005, num_iter=100,
                                      data_name=data_name, atomic_num_list=atomic_num_list,
                                      property_name=property_name, random=False, debug=debug)
        result_list.extend(results)  # results: [(smile2, property, sim, smile1, prop), ...]

    result_list.sort(key=lambda tup: tup[1], reverse=True)

    # check novelty
    train_smile = set()
    for i, r in enumerate(train_prop_sorted):
        qed, plogp, smile = r
        train_smile.add(smile)
        mol = Chem.MolFromSmiles(smile)
        smile2 = Chem.MolToSmiles(mol, isomericSmiles=True)
        train_smile.add(smile2)

    result_list_novel = []
    for i, r in enumerate(result_list):
        smile, score, sim, smile_original, prop = r
        if smile not in train_smile:
            result_list_novel.append(r)

    # dump results
    f = open(model_dir + '/' + data_name + '_' + property_name + '_discovered_sorted.csv', "w")
    for r in result_list_novel:
        smile, score, sim, smile_original, prop = r
        f.write('{},{},{},{},{}\n'.format(score, smile, sim, smile_original, prop))
        f.flush()
    f.close()
    print('Dump done!')


def constrain_optimization_smiles(model, device, data_name, property_name, train_prop, topk,
                                  atomic_num_list, debug, model_dir, sim_cutoff=0.0):
    start_time = time.time()
    if property_name == 'eff':
        col = 0
    elif property_name == 'energy':
        col = 1

    print('Constrained optimization of {} score'.format(property_name))
    train_prop_sorted = sorted(train_prop, key=lambda tup: tup[col]) #, reverse=True)  # qed, plogp, smile
    result_list = []
    nfail = 0
    for i, r in enumerate(train_prop_sorted):
        if i >= topk:
            break
        if i % 50 == 0:
            print('Optimization {}/{}, time: {:.2f} seconds'.format(i, topk, time.time() - start_time))
        qed, plogp, smile = r
        results, ori = optimize_mol(model, property_model, smile, device, sim_cutoff=sim_cutoff, lr=.005, num_iter=100,
                                    data_name=data_name, atomic_num_list=atomic_num_list,
                                    property_name=property_name, random=False, debug=debug)
        if len(results) > 0:
            smile2, property2, sim, _ = results[0]
            plogp_delta = property2 - plogp
            if plogp_delta >= 0:
                result_list.append((smile2, property2, sim, smile, qed, plogp, plogp_delta))
            else:
                nfail += 1
                print('Failure:{}:{}'.format(i, smile))
        else:
            nfail += 1
            print('Failure:{}:{}'.format(i, smile))

    df = pd.DataFrame(result_list,
                      columns=['smile_new', 'prop_new', 'sim', 'smile_old', 'qed_old', 'plogp_old', 'plogp_delta'])

    print(df.describe())
    df.to_csv(model_dir + '/' + data_name + '_' + property_name+'_constrain_optimization.csv', index=False)
    print('Dump done!')
    print('nfail:{} in total:{}'.format(nfail, topk))
    print('success rate: {}'.format((topk-nfail)*1.0/topk))


if __name__ == '__main__':
    start = time.time()
    print("Start at Time: {}".format(time.ctime()))

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default='./results', required=True)
    parser.add_argument("--data_dir", type=str, default='../data')
    parser.add_argument('--data_name', type=str, default='cancer', choices=['cancer'],
                        help='dataset name')
    parser.add_argument("--snapshot_path", "-snapshot", type=str, required=True)
    parser.add_argument("--hyperparams_path", type=str, default='moflow-params.json', required=True)
    parser.add_argument("--property_model_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument('-l', '--learning_rate', type=float, default=0.001, help='Base learning rate')
    parser.add_argument('-e', '--lr_decay', type=float, default=0.999995,
                        help='Learning rate decay, applied every step of the optimization')
    parser.add_argument('-w', '--weight_decay', type=float, default=1e-5,
                        help='L2 norm for the parameters')
    parser.add_argument('--hidden', type=str, default="",
                        help='Hidden dimension list for output regression')
    parser.add_argument('-x', '--max_epochs', type=int, default=5, help='How many epochs to run in total?')
    parser.add_argument('-g', '--gpu', type=int, default=0, help='GPU Id to use')

    parser.add_argument("--delta", type=float, default=0.01)
    parser.add_argument("--img_format", type=str, default='svg')
    parser.add_argument("--property_name", type=str, default='eff', choices=['energy', 'eff'])
    parser.add_argument('--additive_transformations', type=strtobool, default=False,
                        help='apply only additive coupling layers')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='temperature of the gaussian distributions')

    parser.add_argument('--topk', type=int, default=500, help='Top k smiles as seeds')
    parser.add_argument('--debug', type=strtobool, default='true', help='To run optimization with more information')

    parser.add_argument("--sim_cutoff", type=float, default=0.00)
    #
    parser.add_argument('--topscore', action='store_true', default=False, help='To find top score')
    parser.add_argument('--consopt', action='store_true', default=False, help='To do constrained optimization')

    args = parser.parse_args()

    # Device configuration
    device = -1
    if args.gpu >= 0:
        # device = args.gpu
        device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    property_name = args.property_name.lower()
    # chainer.config.train = False
    snapshot_path = os.path.join(args.model_dir, args.snapshot_path)
    hyperparams_path = os.path.join(args.model_dir, args.hyperparams_path)
    model_params = Hyperparameters(path=hyperparams_path)
    model = load_model(snapshot_path, model_params, debug=True)  # Load moflow model

    if args.hidden in ('', ','):
        hidden = []
    else:
        hidden = [int(d) for d in args.hidden.strip(',').split(',')]
    print('Hidden dim for output regression: ', hidden)
    property_model = MoFlowProp(model, hidden)
    # model.eval()  # Set model for evaluation

    atomic_num_list = cancer_atomic_num_list
    transform_fn = transform_cancer.transform_fn_cancer
    valid_idx = transform_cancer.get_val_ids()
    molecule_file = 'cancer_relgcn_kekulized_ggnp.npz'

    # dataset = NumpyTupleDataset(os.path.join(args.data_dir, molecule_file), transform=transform_fn)  # 133885
    dataset = NumpyTupleDataset.load(os.path.join(args.data_dir, molecule_file), transform=transform_fn)

    print('Load {} done, length: {}'.format(os.path.join(args.data_dir, molecule_file), len(dataset)))
    assert len(valid_idx) > 0
    train_idx = [t for t in range(len(dataset)) if t not in valid_idx]  # 224568 = 249455 - 24887
    n_train = len(train_idx)  # 120803 zinc: 224568
    train = torch.utils.data.Subset(dataset, train_idx)  # 120803
    test = torch.utils.data.Subset(dataset, valid_idx)  # 13082  not used for generation

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=args.batch_size)

    # print("loading hyperparamaters from {}".format(hyperparams_path))

    if args.property_model_path is None:
        print("Training regression model over molecular embedding:")
        prop_list = load_property_csv(normalize=True)
        train_prop = [prop_list[i] for i in train_idx]
        test_prop = [prop_list[i] for i in valid_idx]
        print('Prepare data done! Time {:.2f} seconds'.format(time.time() - start))
        property_model_path = os.path.join(args.model_dir, '{}_big_model.pt'.format(property_name))
        property_model = fit_model(property_model, atomic_num_list, train_dataloader, train_prop, device,
                                   property_name=property_name, max_epochs=args.max_epochs,
                                   learning_rate=args.learning_rate, weight_decay=args.weight_decay)
        print("saving {} regression model to: {}".format(property_name, property_model_path))
        torch.save(property_model, property_model_path)
        print('Train and save model done! Time {:.2f} seconds'.format(time.time() - start))
    else:
        print("Loading trained regression model for optimization")
        prop_list = load_property_csv(normalize=False)
        train_prop = [prop_list[i] for i in train_idx]
        test_prop = [prop_list[i] for i in valid_idx]
        print('Prepare data done! Time {:.2f} seconds'.format(time.time() - start))
        property_model_path = os.path.join(args.model_dir, args.property_model_path)
        print("loading {} regression model from: {}".format(property_name, property_model_path))
        device = torch.device('cpu')
        property_model = torch.load(property_model_path, map_location=device)
        print('Load model done! Time {:.2f} seconds'.format(time.time() - start))

        property_model.to(device)
        property_model.eval()

        model.to(device)
        model.eval()

        if args.topscore:
            print('Finding top score:')
            find_top_score_smiles(model, device, args.data_name, property_name, train_prop, args.topk, atomic_num_list, args.debug, args.model_dir)

        if args.consopt:
            print('Constrained optimization:')
            constrain_optimization_smiles(model, device, args.data_name, property_name, train_prop, args.topk,   # train_prop
                                      atomic_num_list, args.debug, args.model_dir, sim_cutoff=args.sim_cutoff)

        print('Total Time {:.2f} seconds'.format(time.time() - start))

