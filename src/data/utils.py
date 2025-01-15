from copy import deepcopy
import torch
from torch_geometric.transforms import SVDFeatureReduction
from torch_geometric.datasets import Planetoid, WebKB, Amazon, WikipediaNetwork
from torch_geometric.data import Data
from torch_geometric.utils import degree, add_self_loops
from fastargs.decorators import param
import math

def x_padding(data, out_dim):
    
    assert data.x.size(-1) <= out_dim
    
    incremental_dimension = out_dim - data.x.size(-1)
    zero_features = torch.zeros((data.x.size(0), incremental_dimension), dtype=data.x.dtype, device=data.x.device)
    data.x = torch.cat([data.x, zero_features], dim=-1)

    return data


def x_svd(data, out_dim):
    
    assert data.x.size(-1) >= out_dim

    reduction = SVDFeatureReduction(out_dim)
    return reduction(data)



#   这个函数会返回一个 迭代器，包含多个 Pyg.Data（比如cora,citeseer等）
@param('general.cache_dir')
def iterate_datasets(data_names, cache_dir):
    #   预训练阶段 dataset是 ["cora","citeseer","cornell"]
    #   下游阶段  dataset 是 "phote"
    if isinstance(data_names, str):
        data_names = [data_names]
    #   dataset是["cora","citeseer","cornell"]
    for data_name in data_names:
        if data_name in ['cora', 'citeseer', 'pubmed']:
            data = Planetoid(root=cache_dir, name=data_name.capitalize())._data
        elif data_name in ['wisconsin', 'texas', 'cornell']:
            data = WebKB(root=cache_dir, name=data_name.capitalize())._data
        elif data_name in ['computers', 'photo']:
            data = Amazon(root=cache_dir, name=data_name.capitalize())._data
        elif data_name in ['chameleon', 'squirrel']:
            preProcDs = WikipediaNetwork(root=cache_dir, name=data_name.capitalize(), geom_gcn_preprocess=False)
            data = WikipediaNetwork(root=cache_dir, name=data_name.capitalize(), geom_gcn_preprocess=True)._data
            data.edge_index = preProcDs[0].edge_index
        else:
            raise ValueError(f'Unknown dataset: {data_name}')
        
        assert isinstance(data, (Data, dict)), f'Unknown data type: {type(data)}'
        #   yield 作用：例如generator_01 = iterate_datasets(data_names)。    会返回一个generator（一种特殊的迭代器iterator），其中包含多个成员，每个data就是一个成员。  
        #   每次调用yield就会产生一个data。  每次generator.__next__()方法就会  调用yield（产生）  并  返回一个成员。
        #   或者 for _ in generator：  这样每次循环访问的一个  _ ，效果就是调用yield  ，产生下一个成员，然后把这个成员返回。
        #   也可以用 list(generator)  把迭代器变成列表，这样可以用list[0]这样按下标访问。
        yield data if isinstance(data, Data) else Data(**data)

@param('general.cache_dir')
def iterate_dataset_feature_tokens(data_names, cache_dir):
    
    if isinstance(data_names, str):
        data_names = [data_names]
    
    for data_name in data_names:
        if data_name in ['cora', 'citeseer', 'pubmed']:
            data = Planetoid(root=cache_dir, name=data_name.capitalize())._data
        elif data_name in ['wisconsin', 'texas', 'cornell']:
            data = WebKB(root=cache_dir, name=data_name.capitalize())._data
        elif data_name in ['computers', 'photo']:
            data = Amazon(root=cache_dir, name=data_name.capitalize())._data
        elif data_name in ['chameleon', 'squirrel']:
            preProcDs = WikipediaNetwork(root=cache_dir, name=data_name.capitalize(), geom_gcn_preprocess=False)
            data = WikipediaNetwork(root=cache_dir, name=data_name.capitalize(), geom_gcn_preprocess=True)._data
            data.edge_index = preProcDs[0].edge_index
        else:
            raise ValueError(f'Unknown dataset: {data_name}')
        
        assert isinstance(data, (Data, dict)), f'Unknown data type: {type(data)}'

        yield data if isinstance(data, Data) else Data(**data)



# including projection operation, SVD。  这个函数是  把给定的Pyg.Data（一个graph），删除mask，并且节点特征维度统一到100
@param('data.node_feature_dim')
def preprocess(data, node_feature_dim):
    #   比如data是  photo数据集 ，节点特征[7650,745]，  edge_index:[2,238162]
    #   删除train_mask等，因为预训练不需要label，自然也就不需要用train_mask来划分训练节点等，预训练是自监督
    if hasattr(data, 'train_mask'):
        del data.train_mask
    if hasattr(data, 'val_mask'):
        del data.val_mask
    if hasattr(data, 'test_mask'):
        del data.test_mask

    if node_feature_dim <= 0:
        edge_index_with_loops = add_self_loops(data.edge_index, num_nodes=data.num_nodes)[0]
        data.x = degree(edge_index_with_loops[1]).reshape((-1,1))
    
    else:   #   把节点特征长度  统一到 node_feature_dim （默认值100）
        # import pdb
        # pdb.set_trace()        
        if data.x.size(-1) > node_feature_dim:
            data = x_svd(data, node_feature_dim)
        elif data.x.size(-1) < node_feature_dim:
            data = x_padding(data, node_feature_dim)
        else:
            pass
    
    return data

# For prompting
def loss_contrastive_learning(x1, x2):
    # T = 0.1
    T = 0.5
    batch_size, _ = x1.size()
    x1_abs = x1.norm(dim=1)
    x2_abs = x2.norm(dim=1)
    
    sim_matrix = torch.einsum('ik,jk->ij', x1+1e-7, x2+1e-7) / torch.einsum('i,j->ij', x1_abs+1e-7, x2_abs+1e-7)
    
    if(True in sim_matrix.isnan()):
        print('Emerging nan value')
    
    sim_matrix = torch.exp(sim_matrix / T)
    
    if(True in sim_matrix.isnan()):
        print('Emerging nan value')    
    
    pos_sim = sim_matrix[range(batch_size), range(batch_size)]

    if(True in pos_sim.isnan()):
        print('Emerging nan value')

    loss = pos_sim / ((sim_matrix.sum(dim=1) - pos_sim) + 1e-4)
    loss = - torch.log(loss).mean()
    if math.isnan(loss.item()):
        print("The value is NaN.")

    return loss

# used in pre_train.py
@param('general.reconstruct')
def gen_ran_output(data, simgrace, reconstruct):
    vice_model = deepcopy(simgrace)

    for (vice_name, vice_model_param), (name, param) in zip(vice_model.named_parameters(), simgrace.named_parameters()):
        if vice_name.split('.')[0] == 'projection_head':
            vice_model_param.data = param.data
        else:
            vice_model_param.data = param.data + 0.1 * torch.normal(0, torch.ones_like(
                param.data) * param.data.std())
    if(reconstruct==0.0):
    
        zj = vice_model.forward_cl(data)

        return zj
    
    else:
    
        zj, hj = vice_model.forward_cl(data)

        return zj, hj