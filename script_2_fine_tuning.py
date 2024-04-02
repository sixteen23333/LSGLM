from code.DatasetLoader import DatasetLoader
from code.MethodBertComp import GraphBertConfig
from code.MethodGraphBertNodeClassification import MethodGraphBertNodeClassification
from code.ResultSaving import ResultSaving
from code.Settings import Settings
import numpy as np
import torch
from code.MethodGraphBertGraphClustering import MethodGraphBertGraphClustering
import os
import psutil
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import scipy.sparse as sp
import sys
import codecs
from code.gcn import GCN
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

#CUDA_VISIBLE_DEVICES=0 python home/Graph-Bert-master/script_3_fine_tuning.py
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

#---- 'cora' , 'citeseer', 'pubmed' ----

dataset_name = 'other'

np.random.seed(1)
torch.manual_seed(1)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)  # 采用三元组(row, col, data)的形式存储稀疏邻接矩阵
    rowsum = np.array(adj.sum(1))  # 按行求和得到rowsum, 即每个节点的度
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()  # (行和rowsum)^(-1/2)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.  # isinf部分赋值为0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)  # 对角化; 将d_inv_sqrt 赋值到对角线元素上, 得到度矩阵^-1/2
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()  # (度矩阵^-1/2)*邻接矩阵*(度矩阵^-1/2)

def adj_normalize(mx):

    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx).dot(r_mat_inv)
    return mx


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))  # 给A加上一个单位矩阵
    return sparse_to_tuple(adj_normalized)

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                        enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                                 dtype=np.int32)
    return labels_onehot



def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

#---- cora-small is for debuging only ----
if dataset_name == 'cora-small':
    nclass = 7
    nfeature = 1433
    ngraph = 10

elif dataset_name == 'other':
    nclass = 13
    nfeature = 16349 #16716,32742
    ngraph = 1000  #1000



elif dataset_name == 'cora':
    nclass = 7
    nfeature = 1433
    ngraph = 2708
elif dataset_name == 'citeseer':
    nclass = 6
    nfeature = 3703
    ngraph = 3312
elif dataset_name == 'pubmed':
    nclass = 3
    nfeature = 500
    ngraph = 19717

elif dataset_name == 'IOT2022':
    nclass = 2
    nfeature = 256
    ngraph = 1000



elif dataset_name == 'dos2017':
    nclass = 9
    nfeature = 21465 #16618,31108，49506
    ngraph = 1000  #1000

elif dataset_name == 'DDoS2019':
    nclass = 5
    nfeature = 7789  #12052
    ngraph = 1500  #1000

elif dataset_name == 'ids2012':
    nclass = 4
    nfeature = 8153 #16716,32742
    ngraph = 1000  #1000

elif dataset_name == 'ids2017':
    nclass = 8
    nfeature = 22142
    ngraph = 1000  #1000

elif dataset_name == 'TFC2016':
    nclass = 8
    nfeature = 10755 #13970
    ngraph = 2000  #1000

elif dataset_name == 'vpn2016':
    nclass = 6
    nfeature = 13371 #14158
    ngraph = 1000  #1000

elif dataset_name == 'ids1217':
    nclass = 15
    nfeature = 14158
    ngraph = 1000  #1000

#---- Fine-Tuning Task 1: Graph Bert Node Classification (Cora, Citeseer, and Pubmed) ----
if 1:
    #---- hyper-parameters ---



    if dataset_name == 'pubmed':
        lr = 0.001
        k = 30
        max_epoch = 1000 # 500 ---- do an early stop when necessary ----

    elif dataset_name == 'other':
        k = 5  # 5,子图大小
        lr = 0.0001
        max_epoch = 1000  # 150




    elif dataset_name == 'cora':
        lr = 0.01
        k = 7
        max_epoch = 150 # 150 ---- do an early stop when necessary ----
    elif dataset_name == 'citeseer':
        k = 5
        lr = 0.001
        max_epoch = 2000 #2000 # it takes a long epochs to get good results, sometimes can be more than 2000

    elif dataset_name == 'IOT2022':
        k = 5 #5
        lr = 0.0001 #0.0001
        max_epoch = 200 #150



    elif dataset_name == 'dos2017':
        k = 5  #5
        lr = 0.0001 #0.0001
        max_epoch = 1000 #150

    elif dataset_name == 'DDoS2019':
        k = 5 #5
        lr = 0.0001
        max_epoch = 1000 #150

    elif dataset_name == 'ids2012':
        k = 5  #5,子图大小
        lr = 0.0001
        max_epoch = 1000 #150

    elif dataset_name == 'ids2017':
        k = 5  #5
        lr = 0.0001
        max_epoch = 1000 #150

    elif dataset_name == 'TFC2016':
        k = 5  #5
        lr = 0.0001
        max_epoch = 1000 #150

    elif dataset_name == 'vpn2016':
        k = 5  #5
        lr = 0.0001
        max_epoch = 1000 #150

    elif dataset_name == 'ids1217':
        k = 5  #5
        lr = 0.0001
        max_epoch = 1000 #150

    x_size = nfeature
    hidden_size = intermediate_size = 100 #32
    num_attention_heads = 2
    num_hidden_layers = 2
    y_size = nclass
    graph_size = ngraph
    residual_type = 'graph_raw'   #'graph_raw'  'raw'   'none'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # --------------------------

    print('************ Start ************')
    print('GrapBert, dataset: ' + dataset_name + ', residual: ' + residual_type + ', k: ' + str(k) + ', hidden dimension: ' + str(hidden_size) +', hidden layer: ' + str(num_hidden_layers) + ', attention head: ' + str(num_attention_heads))
    # ---- objection initialization setction ---------------
    data_obj = DatasetLoader()
    data_obj.dataset_source_folder_path = 'home/Graph-Bert-master/data/' + dataset_name + '/'
    data_obj.dataset_name = dataset_name
    data_obj.k = k
    data_obj.load_all_tag = True

    # import pudb;pu.db
    # setting_obj = Settings()
    # setting_obj.prepare(data_obj)
    # setting_obj.load_run_save_evaluate()

    # idx_features_labels = np.genfromtxt("{}node_new_pac_head_each500_test.txt".format(data_obj.dataset_source_folder_path), dtype=np.dtype(str))
    #
    # features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # one_hot_labels = encode_onehot(idx_features_labels[:, -1])
    # idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # idx_map = {j: i for i, j in enumerate(idx)}
    # index_id_map = {i: j for i, j in enumerate(idx)}
    # edges_unordered = np.genfromtxt("{}link_new_pac_head_each500_test.txt".format(data_obj.dataset_source_folder_path),
    #                                 dtype=np.int32)
    #
    # edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
    #                  dtype=np.int32).reshape(edges_unordered.shape)
    # adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
    #                     shape=(one_hot_labels.shape[0], one_hot_labels.shape[0]),
    #                     dtype=np.float32)
    #
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # norm_adj = adj_normalize(adj + sp.eye(adj.shape[0]))
    # adj = sparse_mx_to_torch_sparse_tensor(norm_adj)


    # adj = DatasetLoader(dataset)['A']
    # support = [preprocess_adj(adj)]
    # support = adj.to(device)
    # t_support = []
    # for i in range(len(support)):
    #     t_support.append(torch.Tensor(support[i]))  # 生成单精度浮点类型的张量
    # for i in range(len(support)):
    #     t_support = [t.to(device) for t in t_support if True]


    bert_config = GraphBertConfig(residual_type = residual_type, k=k, x_size=nfeature, y_size=y_size, hidden_size=hidden_size, intermediate_size=intermediate_size, num_attention_heads=num_attention_heads, num_hidden_layers=num_hidden_layers)
    method_obj = MethodGraphBertNodeClassification(bert_config).to(device)


    # import pudb;pu.db





    #---- set to false to run faster ----
    method_obj.spy_tag = True
    method_obj.max_epoch = max_epoch
    method_obj.lr = lr

    result_obj = ResultSaving()
    result_obj.result_destination_folder_path = 'home/Graph-Bert-master/result/GraphBert/'
    result_obj.result_destination_file_name = dataset_name + '_' + str(num_hidden_layers)

    setting_obj = Settings()

    evaluate_obj = None
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)

    setting_obj.load_run_save_evaluate()
    #import pudb;pu.db
    # ------------------------------------------------------


    method_obj.save_pretrained('home/Graph-Bert-master/result/PreTrained_GraphBert/' + dataset_name + '/node_classification_complete_model/')
    print('************ Finish ************')
#------------------------------------

print(u'当前进程的内存使用：%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024) )

#CUDA_VISIBLE_DEVICES=2 python home/Graph-Bert-master/script_3_fine_tuning.py
#python home/Graph-Bert-master/script_3_fine_tuning.py

#---- Fine-Tuning Task 2: Graph Bert Graph Clustering (Cora, Citeseer, and Pubmed) ----
if 0:
    #---- hyper-parameters ----
    if dataset_name == 'pubmed':
        lr = 0.001
        k = 30
        max_epoch = 500 # 500 ---- do an early stop when necessary ----
    elif dataset_name == 'cora':
        lr = 0.01
        k = 7
        max_epoch = 150 #150 # ---- do an early stop when necessary ----
    elif dataset_name == 'citeseer':
        k = 5
        lr = 0.001
        max_epoch = 300 #2000 # it takes a long epochs to converge, probably more than 2000

    elif dataset_name == 'dos2017':
        k = 5  #5
        lr = 0.0001
        max_epoch = 1000 #150

    elif dataset_name == 'DDoS2019':
        k = 5  #5
        lr = 0.0001
        max_epoch = 1500 #150

    elif dataset_name == 'ids2012':
        k = 5  #5
        lr = 0.0002
        max_epoch = 1000 #150

    elif dataset_name == 'ids2017':
        k = 5  #5
        lr = 0.0001
        max_epoch = 1000 #150

    elif dataset_name == 'TFC2016':
        k = 5  #5
        lr = 0.0001
        max_epoch = 1000 #150

    elif dataset_name == 'vpn2016':
        k = 5  #5
        lr = 0.0001
        max_epoch = 1000 #150

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x_size = nfeature
    hidden_size = intermediate_size = 32
    num_attention_heads = 2
    num_hidden_layers = 2
    y_size = nclass
    graph_size = ngraph
    residual_type = 'graph_raw'
    # --------------------------

    print('************ Start ************')
    print('GrapBert, dataset: ' + dataset_name + ', residual: ' + residual_type + ', k: ' + str(k) + ', hidden dimension: ' + str(hidden_size) +', hidden layer: ' + str(num_hidden_layers) + ', attention head: ' + str(num_attention_heads))
    # ---- objection initialization setction ---------------
    data_obj = DatasetLoader()
    data_obj.dataset_source_folder_path = 'home/Graph-Bert-master/data/' + dataset_name + '/'
    data_obj.dataset_name = dataset_name
    data_obj.k = k
    data_obj.load_all_tag = True

    bert_config = GraphBertConfig(residual_type = residual_type, k=k, x_size=nfeature, y_size=y_size, hidden_size=hidden_size, intermediate_size=intermediate_size, num_attention_heads=num_attention_heads, num_hidden_layers=num_hidden_layers)
    method_obj = MethodGraphBertGraphClustering(bert_config)
    #---- set to false to run faster ----
    method_obj.cluster_number = y_size
    method_obj.spy_tag = True
    method_obj.max_epoch = max_epoch
    method_obj.lr = lr

    result_obj = ResultSaving()
    result_obj.result_destination_folder_path = 'home/Graph-Bert-master/result/GraphBert/clustering_' + dataset_name
    result_obj.result_destination_file_name = '_' + ''

    setting_obj = Settings()

    evaluate_obj = None
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.load_run_save_evaluate()
    # ------------------------------------------------------

    print('************ Finish ************')
#------------------------------------
