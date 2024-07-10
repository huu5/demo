import os

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch import nn
from sklearn.metrics import f1_score, accuracy_score
import os.path as osp
from distance import AttrSim


class Base:

    def __init__(self, adj, features, device):
        self.adj = adj
        self.features = features.to(device)
        self.device = device
        self.cached_adj_norm = None

    def get_adj_norm(self):
        if self.cached_adj_norm is None:
            adj_norm = preprocess_adj(self.adj, self.device)
            self.cached_adj_norm= adj_norm
        return self.cached_adj_norm

    def make_loss(self, embeddings):
        return 0

    def transform_data(self):
        return self.get_adj_norm(), self.features

class PairwiseAttrSim(Base):

    def __init__(self, adj, features, nhid, device, idx_train, args, regression=True):
        args.idx_train = idx_train

        self.adj = adj

        self.args = args
        self.features = features.to(device)
        self.nfeat = features.shape[1]
        self.cached_adj_norm = None
        self.device = device

        # self.labeled = idx_train.cpu().numpy()
        self.labeled = np.array(idx_train)
        self.all = np.arange(adj.shape[0])
        self.unlabeled = np.array([n for n in self.all if n not in idx_train])

        self.regression = regression
        self.nclass = 1
        if regression:
            self.linear = nn.Linear(nhid, self.nclass).to(device)
        else:
            self.linear = nn.Linear(nhid, 2).to(device)

        self.pseudo_labels = None
        self.sims = None

    def transform_data(self):
        return self.get_adj_norm(), self.features

    def make_loss(self, embeddings):
        if self.regression:
            return self.regression_loss(embeddings)
        else:
            return self.classification_loss(embeddings)

    def regression_loss(self, embeddings):
        if self.pseudo_labels is None:
            agent = AttrSim(self.adj ,self.features, args=self.args)
            self.pseudo_labels = agent.get_label().to(self.device)
            node_pairs = agent.node_pairs
            self.node_pairs = node_pairs

        k = 10000
        node_pairs = self.node_pairs
        if len(self.node_pairs[0]) > k:
            sampled = np.random.choice(len(self.node_pairs[0]), k, replace=False)

            embeddings0 = embeddings[node_pairs[0][sampled]]
            embeddings1 = embeddings[node_pairs[1][sampled]]
            embeddings = self.linear(torch.abs(embeddings0 - embeddings1))
            loss = F.mse_loss(embeddings, self.pseudo_labels[sampled], reduction='mean')
        else:
            embeddings0 = embeddings[node_pairs[0]]
            embeddings1 = embeddings[node_pairs[1]]
            embeddings = self.linear(torch.abs(embeddings0 - embeddings1))
            loss = F.mse_loss(embeddings, self.pseudo_labels, reduction='mean')

        # print(loss)
        return loss

    def classification_loss(self, embeddings):
        if self.pseudo_labels is None:
            agent = AttrSim(self.adj ,self.features, self.labels, self.args)
            pseudo_labels = agent.get_class()
            self.pseudo_labels = torch.LongTensor(pseudo_labels).to(self.device)
            self.node_pairs = agent.node_pairs

        node_pairs = self.node_pairs
        embeddings0 = embeddings[node_pairs[0]]
        embeddings1 = embeddings[node_pairs[1]]
        embeddings = self.linear(torch.abs(embeddings0 - embeddings1))
        output = F.log_softmax(embeddings, dim=1)

        loss = F.nll_loss(output, self.pseudo_labels)
        print(loss)
        # from metric import accuracy
        # acc = accuracy(output, self.pseudo_labels)
        acc = accuracy_score(output, self.pseudo_labels)
        print(acc)
        return loss

    def sample(self, labels, ratio=0.1, k=2000):
        node_pairs = []
        for i in range(1, labels.max()+1):
            tmp = np.array(np.where(labels==i)).transpose()
            indices = np.random.choice(np.arange(len(tmp)), k, replace=False)
            node_pairs.append(tmp[indices])
        node_pairs = np.array(node_pairs).reshape(-1, 2).transpose()
        return node_pairs[0], node_pairs[1]


class MergedKNNGraph(Base):

    def __init__(self, adj, features, nhid, device, idx_train, args):
        self.adj = adj
        self.args = args

        self.features = features.to(device)
        self.nfeat = features.shape[1]
        self.cached_adj_norm = None
        self.device = device

        # self.labeled = idx_train.cpu().numpy()
        self.labeled = np.array(idx_train)
        self.all = np.arange(adj.shape[0])
        self.unlabeled = np.array([n for n in self.all if n not in idx_train])
        self.nclass = 1
        self.pseudo_labels = None

        # degree = self.adj.sum(0).A1

        if hasattr(args, 'k'):
            k = args.k
        elif args.k is not None:
            k = 20
        else:
            degree = self.adj.sum(0)
            k = int(torch.mean(degree))

        if not osp.exists('saved/'):
           os.mkdir('saved')
        if not os.path.exists(f'saved/{args.dataset}_sims_{k}.npz'):
            from sklearn.metrics.pairwise import cosine_similarity
            features = np.copy(features)
            features[features!=0] = 1
            sims = cosine_similarity(features)
            sims[(np.arange(len(sims)), np.arange(len(sims)))] = 0
            for i in range(len(sims)):
                indices_argsort = np.argsort(sims[i])
                sims[i, indices_argsort[: -k]] = 0

            self.A_feat = sp.csr_matrix(sims)
            sp.save_npz(f'saved/{args.dataset}_sims_{k}.npz', self.A_feat)
        else:
            print(f'loading saved/{args.dataset}_sims_{k}.npz')
            self.A_feat = sp.load_npz(f'saved/{args.dataset}_sims_{k}.npz')


    def transform_data(self, lambda_=None):
        if self.cached_adj_norm is None:
            if lambda_ is None:
                r_adj = self.adj + self.args.lambda_ * self.A_feat
            else:
                r_adj = self.adj + lambda_ * self.A_feat
            r_adj = preprocess_adj(r_adj, self.device)
            self.cached_adj_norm = r_adj
        return self.cached_adj_norm, self.features

    def make_loss(self, embeddings):
        return 0


def preprocess_adj_noloop(adj, device):
    # adj_normalizer = fetch_normalization(normalization)
    adj_normalizer = noaug_normalized_adjacency
    r_adj = adj_normalizer(adj)
    r_adj = sparse_mx_to_torch_sparse_tensor(r_adj).float()
    r_adj = r_adj.to(device)
    return r_adj


def preprocess_adj(adj, device):
    # adj_normalizer = fetch_normalization(normalization)
    adj_normalizer = aug_normalized_adjacency
    r_adj = adj_normalizer(adj)
    r_adj = sparse_mx_to_torch_sparse_tensor(r_adj).float()
    r_adj = r_adj.to(device)
    return r_adj


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)


def aug_normalized_adjacency(adj):
    adj = adj + sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()


def noaug_normalized_adjacency(adj):
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()