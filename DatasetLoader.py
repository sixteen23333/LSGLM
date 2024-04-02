'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017 Jiawei Zhang <jwzhanggy@gmail.com>
# License: TBD

from code.base_class.dataset import dataset
import torch
import numpy as np
import scipy.sparse as sp
from numpy.linalg import inv
import pickle


class DatasetLoader(dataset):
    c = 0.15
    k = 5
    data = None
    batch_size = None

    dataset_source_folder_path = None
    dataset_name = None

    load_all_tag = True#False
    compute_s = True #False

    def __init__(self, seed=None, dName=None, dDescription=None):
        super(DatasetLoader, self).__init__(dName, dDescription)

    def load_hop_wl_batch(self):
        print('Load WL Dictionary')
        f = open('home/Graph-Bert-master/result/WL/' + self.dataset_name, 'rb')
        wl_dict = pickle.load(f)
        f.close()

        print('Load Hop Distance Dictionary')
        f = open('home/Graph-Bert-master/result/Hop/hop_' + self.dataset_name + '_' + str(self.k), 'rb')
        hop_dict = pickle.load(f)
        f.close()

        print('Load Subgraph Batches')
        f = open('home/Graph-Bert-master/result/Batch/' + self.dataset_name + '_' + str(self.k), 'rb')
        batch_dict = pickle.load(f)
        f.close()

        return hop_dict, wl_dict, batch_dict

    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

    def adj_normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -0.5).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx).dot(r_mat_inv)
        return mx

    def accuracy(self, output, labels):
        preds = output.max(1)[1].type_as(labels)
        correct = preds.eq(labels).double()
        correct = correct.sum()
        return correct / len(labels)

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def encode_onehot(self, labels):
        classes = set(labels)
        classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                        enumerate(classes)}
        labels_onehot = np.array(list(map(classes_dict.get, labels)),
                                 dtype=np.int32)
        return labels_onehot

    def load(self):
        """Load citation network dataset (cora only for now)"""
        print('Loading {} dataset...'.format(self.dataset_name))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        idx_features_labels = np.genfromtxt("{}node_new_pac_head_each500_iot2022_tfc2016.txt".format(self.dataset_source_folder_path), dtype=np.dtype(str))
        # features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)

        #len(idx_features_labels[:, 1:-1][0]) = 16618
        #idx_features_labels[:, 1:-1] : array([['0', '0', '0', ..., '0', '0', '0'],['0', '0', '0', ..., '0', '0', '0'],['0', '0', '0', ..., '0', '0', '0']
        features = sp.csr_matrix(idx_features_labels[:, 1:-1],dtype=np.float32)

        one_hot_labels = self.encode_onehot(idx_features_labels[:, -1]) #idx_features_labels[:, -1] : array(['slowheaders', 'rudy', 'slowheaders', ...,'slowbody', 'slowbody','Hulk'], dtype='<U11')

        # build graph

        #idx : array([2727,  257, 1386, ..., 1151,  330, 2462],dtype=int32)
        idx = np.array(idx_features_labels[:, 0], dtype=np.int32)

        #idx_map : {2727: 0, 257: 1, 1386: 2, 2289: 3, 1773: 4, 2656: 5,843: 6, 1383: 7, 102: 8, 2016: 9,...
        idx_map = {j: i for i, j in enumerate(idx)}

        #index_id_map : {0: 2727, 1: 257, 2: 1386, 3: 2289, 4: 1773, 5: 2656,6: 843, 7: 1383, 8: 102,
        index_id_map = {i: j for i, j in enumerate(idx)}

        edges_unordered = np.genfromtxt("{}link_new_pac_head_each500_iot2022_tfc2016.txt".format(self.dataset_source_folder_path),
                                        dtype=np.int32)
        #import pudb;pu.db
        #edges : array([[2898, 1829],[2898, 2135],[2898, 2512],...,[ 855, 2241],
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                         dtype=np.int32).reshape(edges_unordered.shape)

        #edges[:, 0] : array([2898, 2898, 2898, ...,  855,  855,  855]
        #edges[:, 1] : array([1829, 2135, 2512, ..., 2241, 2549,  667],
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(one_hot_labels.shape[0], one_hot_labels.shape[0]),
                            dtype=np.float32)

        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        eigen_adj = None
        #import pudb;pu.db
        if self.compute_s:
            eigen_adj = self.c * inv((sp.eye(adj.shape[0]) - (1 - self.c) * self.adj_normalize(adj)).toarray())
            # import pudb;pu.db

        norm_adj = self.adj_normalize(adj + sp.eye(adj.shape[0]))
        # eigen_adj = torch.tensor(eigen_adj).to(device)
        #import pudb;pu.db
        if self.dataset_name == 'cora':
            idx_train = range(140)
            idx_test = range(200, 1200)
            idx_val = range(1200, 1500)

        elif self.dataset_name == 'other':
            idx_train = range(700)
            idx_val = range(700, 1000)
            idx_test = range(1000,5000)


        elif self.dataset_name == 'citeseer':
            idx_train = range(120)
            idx_test = range(200, 1200)
            idx_val = range(1200, 1500)
            #features = self.normalize(features)
        elif self.dataset_name == 'pubmed':
            idx_train = range(60)
            idx_test = range(6300, 7300)
            idx_val = range(6000, 6300)
        elif self.dataset_name == 'cora-small':
            idx_train = range(5)
            idx_val = range(5, 10)
            idx_test = range(5, 10)


        elif self.dataset_name == 'IOT2022':
            idx_train = range(490)
            idx_val = range(490, 700)
            idx_test = range(700, 1000)






        # elif self.dataset_name == 'dos2017':
        #     idx_train = range(120)
        #     idx_val = range(120, 784)
        #     idx_test = range(784, 1000)


        # elif self.dataset_name == 'dos2017':
        #     idx_train = range(2000)
        #     idx_val = range(2000, 2300)
        #     idx_test = range(2300, 3000)

        # elif self.dataset_name == 'dos2017':
        #     idx_train = range(4000)
        #     idx_val = range(4000, 4600)
        #     idx_test = range(4600, 6000)

        elif self.dataset_name == 'dos2017':
            idx_train = range(2666)
            idx_val = range(2666, 3066)
            idx_test = range(3066, 4000)


        # elif self.dataset_name == 'DDoS2019':#each500
        #     idx_train = range(2333)
        #     idx_val = range(2333, 2683)
        #     idx_test = range(2683, 3500)




        # elif self.dataset_name == 'DDoS2019': #each500,6
        #     idx_train = range(2000)
        #     idx_val = range(2000, 2300)
        #     idx_test = range(2300, 3000)


        elif self.dataset_name == 'DDoS2019':#each500，5
            idx_train = range(1575)
            idx_val = range(1575, 2250)
            idx_test = range(2250, 2500)

        # elif self.dataset_name == 'ids2012':
        #     idx_train = range(2000)
        #     idx_val = range(2000, 2300)
        #     idx_test = range(2300, 3000)

        # elif self.dataset_name == 'ids2012':
        #     idx_train = range(4000)
        #     idx_val = range(4000, 4600)
        #     idx_test = range(4600, 6000)

        # elif self.dataset_name == 'ids2012':
        #     idx_train = range(2174)
        #     idx_val = range(2174, 2500)
        #     idx_test = range(2500, 3575)

        elif self.dataset_name == 'ids2012':#each500,4
            idx_train = range(980)
            idx_val = range(980, 1400)
            idx_test = range(1400, 2000)

        # elif self.dataset_name == 'ids2012':#each500,3
        #     idx_train = range(912)
        #     idx_val = range(912, 1048)
        #     idx_test = range(1048, 1500)


        # elif self.dataset_name == 'ids2012':#each500
        #     idx_train = range(1520)
        #     idx_val = range(1520, 1748)
        #     idx_test = range(1748, 2500)

        elif self.dataset_name == 'ids2017':#each500
            idx_train = range(2666)
            idx_val = range(2000, 3066)
            idx_test = range(3066, 4000)

        # elif self.dataset_name == 'TFC2016':
        #     idx_train = range(2000)
        #     idx_val = range(2000, 2300)
        #     idx_test = range(2300, 3000)

        # elif self.dataset_name == 'TFC2016':
        #     idx_train = range(4000)
        #     idx_val = range(4000, 4600)
        #     idx_test = range(4600, 6000)



        # elif self.dataset_name == 'TFC2016':#all
        #     idx_train = range(20064)
        #     idx_val = range(20064, 23077)
        #     idx_test = range(23077, 33000)

        # elif self.dataset_name == 'TFC2016':#each1000
        #     idx_train = range(6688)
        #     idx_val = range(6688, 7692)
        #     idx_test = range(7692, 11000)

        # elif self.dataset_name == 'TFC2016':#each500
        #     idx_train = range(3344)
        #     idx_val = range(3344, 3846)
        #     idx_test = range(3846, 5500)

        elif self.dataset_name == 'TFC2016':#each500 8
            idx_train = range(2520)
            idx_val = range(2520, 3600)
            idx_test = range(3600, 4000)



        # elif self.dataset_name == 'TFC2016':#each500 9
        #     idx_train = range(2734)
        #     idx_val = range(2734, 3145)
        #     idx_test = range(3145, 4500)

        # elif self.dataset_name == 'TFC2016':
        #     idx_train = range(95)
        #     idx_val = range(95, 110)
        #     idx_test = range(110, 154)

        elif self.dataset_name == 'vpn2016':#each500
            idx_train = range(2000)
            idx_val = range(2000, 2300)
            idx_test = range(2300, 3000)

        # elif self.dataset_name == 'vpn2016':
        #     idx_train = range(4000)
        #     idx_val = range(4000, 4600)
        #     idx_test = range(4600, 6000)

        # elif self.dataset_name == 'vpn2016':
        #     idx_train = range(3000)
        #     idx_val = range(3000, 3450)
        #     idx_test = range(3450, 4290)

        elif self.dataset_name == 'ids1217':
            idx_train = range(4000)
            idx_val = range(4000, 4600)
            idx_test = range(4600, 6000)

        features = torch.FloatTensor(np.array(features.todense()))
        labels = torch.LongTensor(np.where(one_hot_labels)[1])
        adj = self.sparse_mx_to_torch_sparse_tensor(norm_adj)
        #import pudb;pu.db
        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

        if self.load_all_tag:
            hop_dict, wl_dict, batch_dict = self.load_hop_wl_batch()
            raw_feature_list = []
            role_ids_list = []
            position_ids_list = []
            hop_ids_list = []
            for node in idx:
                node_index = idx_map[node]
                neighbors_list = batch_dict[node]

                raw_feature = [features[node_index].tolist()]
                role_ids = [wl_dict[node]]
                position_ids = range(len(neighbors_list) + 1)
                hop_ids = [0]
                for neighbor, intimacy_score in neighbors_list:


                    neighbor_index = idx_map[neighbor]
                    raw_feature.append(features[neighbor_index].tolist())
                    role_ids.append(wl_dict[neighbor])
                    if neighbor in hop_dict[node]:
                        hop_ids.append(hop_dict[node][neighbor])
                    else:
                        hop_ids.append(99)
                raw_feature_list.append(raw_feature)
                role_ids_list.append(role_ids)
                position_ids_list.append(position_ids)
                hop_ids_list.append(hop_ids)

            raw_embeddings = torch.FloatTensor(raw_feature_list).to(device)
            wl_embedding = torch.LongTensor(role_ids_list).to(device)
            hop_embeddings = torch.LongTensor(hop_ids_list).to(device)
            int_embeddings = torch.LongTensor(position_ids_list).to(device)
            features = features.to(device)
            adj = adj.to(device)
            # index_id_map = index_id_map.to(device)

            index_id_map = {key: torch.tensor(index_id_map[key]).to(device) for key in index_id_map}

            edges_unordered = torch.tensor(edges_unordered).to(device)
            labels = labels.to(device)
            idx = torch.tensor(idx).to(device),
            idx_train = idx_train.to(device)
            idx_test = idx_test.to(device)
            idx_val = idx_val.to(device)

        else:
            raw_embeddings, wl_embedding, hop_embeddings, int_embeddings = None, None, None, None
        #import pudb;pu.db
        # import pudb;pu.db
        return {'X': features, 'A': adj, 'S': eigen_adj, 'index_id_map': index_id_map, 'edges': edges_unordered, 'raw_embeddings': raw_embeddings, 'wl_embedding': wl_embedding, 'hop_embeddings': hop_embeddings, 'int_embeddings': int_embeddings, 'y': labels, 'idx': idx, 'idx_train': idx_train, 'idx_test': idx_test, 'idx_val': idx_val}














# class DatasetLoader(dataset):
#     c = 0.15
#     k = 5
#     data = None
#     batch_size = None
#
#     dataset_source_folder_path = None
#     dataset_name = None
#
#     load_all_tag = True  # False
#     compute_s = True  # False
#
#     def __init__(self, seed=None, dName=None, dDescription=None):
#         super(DatasetLoader, self).__init__(dName, dDescription)
#
#     def load_hop_wl_batch(self):
#         print('Load WL Dictionary')
#         f = open('home/Graph-Bert-master/result/WL/' + self.dataset_name, 'rb')
#         wl_dict = pickle.load(f)
#         f.close()
#
#         print('Load Hop Distance Dictionary')
#         f = open('home/Graph-Bert-master/result/Hop/hop_' + self.dataset_name + '_' + str(self.k), 'rb')
#         hop_dict = pickle.load(f)
#         f.close()
#
#         print('Load Subgraph Batches')
#         f = open('home/Graph-Bert-master/result/Batch/' + self.dataset_name + '_' + str(self.k), 'rb')
#         batch_dict = pickle.load(f)
#         f.close()
#
#         return hop_dict, wl_dict, batch_dict
#
#     def normalize(self, mx):
#         """Row-normalize sparse matrix"""
#         rowsum = np.array(mx.sum(1))
#         r_inv = np.power(rowsum, -1).flatten()
#         r_inv[np.isinf(r_inv)] = 0.
#         r_mat_inv = sp.diags(r_inv)
#         mx = r_mat_inv.dot(mx)
#         return mx
#
#     def adj_normalize(self, mx):
#         """Row-normalize sparse matrix"""
#         rowsum = np.array(mx.sum(1))
#         r_inv = np.power(rowsum, -0.5).flatten()
#         r_inv[np.isinf(r_inv)] = 0.
#         r_mat_inv = sp.diags(r_inv)
#         mx = r_mat_inv.dot(mx).dot(r_mat_inv)
#         return mx
#
#     def accuracy(self, output, labels):
#         preds = output.max(1)[1].type_as(labels)
#         correct = preds.eq(labels).double()
#         correct = correct.sum()
#         return correct / len(labels)
#
#     def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
#         """Convert a scipy sparse matrix to a torch sparse tensor."""
#         sparse_mx = sparse_mx.tocoo().astype(np.float32)
#         indices = torch.from_numpy(
#             np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
#         values = torch.from_numpy(sparse_mx.data)
#         shape = torch.Size(sparse_mx.shape)
#         return torch.sparse.FloatTensor(indices, values, shape)
#
#     def encode_onehot(self, labels):
#         classes = set(labels)
#         classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
#                         enumerate(classes)}
#         labels_onehot = np.array(list(map(classes_dict.get, labels)),
#                                  dtype=np.int32)
#         return labels_onehot
#
#     def load(self):
#         """Load citation network dataset (cora only for now)"""
#         print('Loading {} dataset...'.format(self.dataset_name))
#
#         idx_features_labels = np.genfromtxt("{}node_new_each10.txt".format(self.dataset_source_folder_path),
#                                             dtype=np.dtype(str))
#         # features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
#
#         # len(idx_features_labels[:, 1:-1][0]) = 16618
#         # idx_features_labels[:, 1:-1] : array([['0', '0', '0', ..., '0', '0', '0'],['0', '0', '0', ..., '0', '0', '0'],['0', '0', '0', ..., '0', '0', '0']
#         features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
#
#         one_hot_labels = self.encode_onehot(idx_features_labels[:,
#                                             -1])  # idx_features_labels[:, -1] : array(['slowheaders', 'rudy', 'slowheaders', ...,'slowbody', 'slowbody','Hulk'], dtype='<U11')
#
#         # build graph
#
#         # idx : array([2727,  257, 1386, ..., 1151,  330, 2462],dtype=int32)
#         idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
#
#         # idx_map : {2727: 0, 257: 1, 1386: 2, 2289: 3, 1773: 4, 2656: 5,843: 6, 1383: 7, 102: 8, 2016: 9,...
#         idx_map = {j: i for i, j in enumerate(idx)}
#
#         # index_id_map : {0: 2727, 1: 257, 2: 1386, 3: 2289, 4: 1773, 5: 2656,6: 843, 7: 1383, 8: 102,
#         index_id_map = {i: j for i, j in enumerate(idx)}
#
#         edges_unordered = np.genfromtxt("{}link_new_each10.txt".format(self.dataset_source_folder_path),
#                                         dtype=np.int32)
#         # import pudb;pu.db
#         # edges : array([[2898, 1829],[2898, 2135],[2898, 2512],...,[ 855, 2241],
#         edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
#                          dtype=np.int32).reshape(edges_unordered.shape)
#
#         # edges[:, 0] : array([2898, 2898, 2898, ...,  855,  855,  855]
#         # edges[:, 1] : array([1829, 2135, 2512, ..., 2241, 2549,  667],
#         adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
#                             shape=(one_hot_labels.shape[0], one_hot_labels.shape[0]),
#                             dtype=np.float32)
#
#         adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
#         eigen_adj = None
#         if self.compute_s:
#             eigen_adj = self.c * inv((sp.eye(adj.shape[0]) - (1 - self.c) * self.adj_normalize(adj)).toarray())
#
#         norm_adj = self.adj_normalize(adj + sp.eye(adj.shape[0]))
#         # eigen_adj = torch.tensor(eigen_adj).to(device)
#         # import pudb;pu.db
#         if self.dataset_name == 'cora':
#             idx_train = range(140)
#             idx_test = range(200, 1200)
#             idx_val = range(1200, 1500)
#         elif self.dataset_name == 'citeseer':
#             idx_train = range(120)
#             idx_test = range(200, 1200)
#             idx_val = range(1200, 1500)
#             # features = self.normalize(features)
#         elif self.dataset_name == 'pubmed':
#             idx_train = range(60)
#             idx_test = range(6300, 7300)
#             idx_val = range(6000, 6300)
#         elif self.dataset_name == 'cora-small':
#             idx_train = range(5)
#             idx_val = range(5, 10)
#             idx_test = range(5, 10)
#
#         # elif self.dataset_name == 'dos2017':
#         #     idx_train = range(120)
#         #     idx_val = range(120, 784)
#         #     idx_test = range(784, 1000)
#
#         # elif self.dataset_name == 'dos2017':
#         #     idx_train = range(2000)
#         #     idx_val = range(2000, 2300)
#         #     idx_test = range(2300, 3000)
#
#         # elif self.dataset_name == 'dos2017':
#         #     idx_train = range(4000)
#         #     idx_val = range(4000, 4600)
#         #     idx_test = range(4600, 6000)
#
#         elif self.dataset_name == 'dos2017':
#             idx_train = range(6666)
#             idx_val = range(6666, 7666)
#             idx_test = range(7666, 10000)
#
#         # elif self.dataset_name == 'DDoS2019':
#         #     idx_train = range(2000)
#         #     idx_val = range(2000, 2300)
#         #     idx_test = range(2300, 3000)
#
#         # elif self.dataset_name == 'DDoS2019':
#         #     idx_train = range(4000)
#         #     idx_val = range(4000, 4600)
#         #     idx_test = range(4600, 6000)
#
#         elif self.dataset_name == 'DDoS2019':
#             idx_train = range(3043)
#             idx_val = range(3043, 3500)
#             idx_test = range(3500, 5005)
#
#
#         # elif self.dataset_name == 'ids2012':
#         #     idx_train = range(2000)
#         #     idx_val = range(2000, 2300)
#         #     idx_test = range(2300, 3000)
#
#         # elif self.dataset_name == 'ids2012':
#         #     idx_train = range(4000)
#         #     idx_val = range(4000, 4600)
#         #     idx_test = range(4600, 6000)
#
#         elif self.dataset_name == 'ids2012':
#             idx_train = range(2174)
#             idx_val = range(2174, 2500)
#             idx_test = range(2500, 3575)
#
#
#
#         elif self.dataset_name == 'ids2017':
#             idx_train = range(2000)
#             idx_val = range(2000, 2300)
#             idx_test = range(2300, 3000)
#
#         # elif self.dataset_name == 'TFC2016':
#         #     idx_train = range(2000)
#         #     idx_val = range(2000, 2300)
#         #     idx_test = range(2300, 3000)
#
#         # elif self.dataset_name == 'TFC2016':
#         #     idx_train = range(4000)
#         #     idx_val = range(4000, 4600)
#         #     idx_test = range(4600, 6000)
#
#         # elif self.dataset_name == 'TFC2016':
#         #     idx_train = range(4782)
#         #     idx_val = range(4782, 5500)
#         #     idx_test = range(5500, 7865)
#
#         elif self.dataset_name == 'TFC2016':
#             idx_train = range(95)
#             idx_val = range(95, 110)
#             idx_test = range(110, 154)
#
#         # elif self.dataset_name == 'vpn2016':
#         #     idx_train = range(2000)
#         #     idx_val = range(2000, 2300)
#         #     idx_test = range(2300, 3000)
#
#         # elif self.dataset_name == 'vpn2016':
#         #     idx_train = range(4000)
#         #     idx_val = range(4000, 4600)
#         #     idx_test = range(4600, 6000)
#
#         elif self.dataset_name == 'vpn2016':
#             idx_train = range(3000)
#             idx_val = range(3000, 3450)
#             idx_test = range(3450, 4290)
#
#         elif self.dataset_name == 'ids1217':
#             idx_train = range(4000)
#             idx_val = range(4000, 4600)
#             idx_test = range(4600, 6000)
#
#         features = torch.FloatTensor(np.array(features.todense()))
#         labels = torch.LongTensor(np.where(one_hot_labels)[1])
#         adj = self.sparse_mx_to_torch_sparse_tensor(norm_adj)
#         # import pudb;pu.db
#         idx_train = torch.LongTensor(idx_train)
#         idx_val = torch.LongTensor(idx_val)
#         idx_test = torch.LongTensor(idx_test)
#
#         if self.load_all_tag:
#             hop_dict, wl_dict, batch_dict = self.load_hop_wl_batch()
#             raw_feature_list = []
#             role_ids_list = []
#             position_ids_list = []
#             hop_ids_list = []
#             for node in idx:
#                 node_index = idx_map[node]
#                 neighbors_list = batch_dict[node]
#
#                 raw_feature = [features[node_index].tolist()]
#                 role_ids = [wl_dict[node]]
#                 position_ids = range(len(neighbors_list) + 1)
#                 hop_ids = [0]
#                 for neighbor, intimacy_score in neighbors_list:
#
#                     neighbor_index = idx_map[neighbor]
#                     raw_feature.append(features[neighbor_index].tolist())
#                     role_ids.append(wl_dict[neighbor])
#                     if neighbor in hop_dict[node]:
#                         hop_ids.append(hop_dict[node][neighbor])
#                     else:
#                         hop_ids.append(99)
#                 raw_feature_list.append(raw_feature)
#                 role_ids_list.append(role_ids)
#                 position_ids_list.append(position_ids)
#                 hop_ids_list.append(hop_ids)
#
#             raw_embeddings = torch.FloatTensor(raw_feature_list)
#             wl_embedding = torch.LongTensor(role_ids_list)
#             hop_embeddings = torch.LongTensor(hop_ids_list)
#             int_embeddings = torch.LongTensor(position_ids_list)
#
#
#         else:
#             raw_embeddings, wl_embedding, hop_embeddings, int_embeddings = None, None, None, None
#         # import pudb;pu.db
#
#         return {'X': features, 'A': adj, 'S': eigen_adj, 'index_id_map': index_id_map, 'edges': edges_unordered,
#                 'raw_embeddings': raw_embeddings, 'wl_embedding': wl_embedding, 'hop_embeddings': hop_embeddings,
#                 'int_embeddings': int_embeddings, 'y': labels, 'idx': idx, 'idx_train': idx_train, 'idx_test': idx_test,
#                 'idx_val': idx_val}
