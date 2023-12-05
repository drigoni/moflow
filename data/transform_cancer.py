import json

import numpy as np

#                         H, Li,B, C, O, N, F, Na, Mg, Al, Si, P , S , Cl, K , Ca, Ti, V , Cr, Mn, Fe, Co, Ni, Cu, Zn, Ga, Ge, As, Se, Br, Zr, Nb, Mo, Ru, Rh, Pd, Ag, Cd, In, Sn, Sb, Te, I , Ba, La, Ce, Nd, Sm, Gd, Dy, Hf, Ta, W , Re, Os, Lr, Pt, Au, Hg, Tl, Pb, Bi, Fr, Ac, Th, Am,
cancer_atomic_num_list = [1, 3, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 19, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 40, 41, 42, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 56, 57, 58, 60, 62, 64, 66, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 87, 89, 90, 95, 0]  # 0 is for virtual node.


def one_hot_cancer(data, out_size=80):
    num_max_id = len(cancer_atomic_num_list)
    assert data.shape[0] == out_size
    b = np.zeros((out_size, num_max_id), dtype=np.float32)
    for i in range(out_size):
        ind = cancer_atomic_num_list.index(data[i])
        b[i, ind] = 1.
    return b


def transform_fn_cancer(data):
    node, adj, label = data
    # convert to one-hot vector
    # node = one_hot(node).astype(np.float32)
    node = one_hot_cancer(node).astype(np.float32)
    # single, double, triple and no-bond. Note that last channel axis is not connected instead of aromatic bond.
    adj = np.concatenate([adj[:3], 1 - np.sum(adj[:3], axis=0, keepdims=True)],
                         axis=0).astype(np.float32)
    return node, adj, label


def get_val_ids():
    file_path = '../data/valid_idx_cancer.json'
    print('loading train/valid split information from: {}'.format(file_path))
    with open(file_path) as json_data:
        data = json.load(json_data)
    val_ids = [idx-1 for idx in data]
    return val_ids
