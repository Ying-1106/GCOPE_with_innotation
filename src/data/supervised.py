from fastargs.decorators import param
import torch

#   下游节点，传入的data是  一个Pyg.Data对象，表示的是photo数据集
#   对于data中的每一个节点（比如photo有7650个节点），都生成一个以该节点为中心的诱导子图，返回值就是这些诱导子图
def induced_graphs(data, smallest_size=10, largest_size=30):

    from torch_geometric.utils import subgraph, k_hop_subgraph
    from torch_geometric.data import Data
    import numpy as np

    induced_graph_list = []
    total_node_num = data.x.size(0) #   data的节点总数

    for index in range(data.x.size(0)): # index表示每一个  节点ID
        current_label = data.y[index].item()    #   当前index节点 的 label

        current_hop = 2 #   跳数

        #   k_hop_subgraph是 Pyg提供的寻找k跳子图的函数。返回的subset是 子图中包含的节点ID（整图ID）
        subset, _, _, _ = k_hop_subgraph(node_idx=index, num_hops=current_hop,
                                            edge_index=data.edge_index, relabel_nodes=True)
        
        while len(subset) < smallest_size and current_hop < 5:
            current_hop += 1
            subset, _, _, _ = k_hop_subgraph(node_idx=index, num_hops=current_hop,
                                                edge_index=data.edge_index)
            
        if len(subset) < smallest_size:
            need_node_num = smallest_size - len(subset)
            pos_nodes = torch.argwhere(data.y == int(current_label)) 
            candidate_nodes = torch.from_numpy(np.setdiff1d(pos_nodes.numpy(), subset.numpy()))
            candidate_nodes = candidate_nodes[torch.randperm(candidate_nodes.shape[0])][0:need_node_num]
            subset = torch.cat([torch.flatten(subset), torch.flatten(candidate_nodes)])

        if len(subset) > largest_size:
            subset = subset[torch.randperm(subset.shape[0])][0:largest_size - 1]    #   从subset中随机选择 29个节点ID（整图节点ID）
            subset = torch.unique(torch.cat([torch.LongTensor([index]), torch.flatten(subset)]))    # 把这29个节点ID和  当前节点ID（index）拼到一起
        
    #   经过上面的操作后，subset就是  以index节点为中心的2跳子图  中的 所有节点ID（整图节点ID）

    #   这个subgraph是生成诱导子图的操作，给出subset（一些节点ID），返回一个诱导子图
    #   诱导子图中，节点是：subset这些节点，边是：这些节点之间的边
        sub_edge_index, _ = subgraph(subset, data.edge_index, relabel_nodes=True)   #   返回的sub_edge_index就是诱导子图中的边（边上两端点的ID是子图ID）
    #   x是诱导子图中的节点的特征
        x = data.x[subset]

        induced_graph = Data(x=x, edge_index=sub_edge_index, y=current_label)   #   这个induced_graph是就是生成的诱导子图，包含节点特征和边
        induced_graph_list.append(induced_graph)
        if(index%1000==0):
            print('生成的第{}/{}张子图数据'.format(index,total_node_num))
    
    print('生成了{}/{}张子图数据'.format(index,total_node_num))
    return induced_graph_list


@param('data.seed')
@param('general.cache_dir')
@param('general.few_shot')
def get_supervised_data(dataset, ratios, seed, cache_dir,few_shot):
    #   dataset  是 "photo"，表示下游阶段的数据集。ratios是[0.1,0.1,0.8]是训练、验证、测试的划分。cache_dir是存数据集的位置
    import os
    cache_dir = os.path.join(cache_dir, dataset)    #   返回值是 storage/.cache/photo
    os.makedirs(cache_dir, exist_ok=True)

    if(few_shot == 0):
        cache_path = os.path.join(cache_dir, ','.join([f'{r:f}' for r in ratios]) + f'_s{seed}' + '.pt')

        # import pdb
        # pdb.set_trace()

        if os.path.exists(cache_path):
            return torch.load(cache_path)

        from .utils import preprocess, iterate_datasets

        data = preprocess(next(iterate_datasets(dataset)))
            
        num_classes = torch.unique(data.y).size(0)
        target_graph_list = induced_graphs(data)

        from torch.utils.data import random_split
        train_set, val_set, test_set = random_split(target_graph_list, ratios, torch.Generator().manual_seed(seed))

    else:   #   storage/cache/photo 目录下， 创建一个  few_shot_seed.pt文件。 （.pt文件是用来保存 torch模型参数的文件）
        cache_path = os.path.join(cache_dir + f'/{few_shot}_shot' + f'_s{seed}' + '.pt')

        if os.path.exists(cache_path):
            return torch.load(cache_path)

        from .utils import preprocess, iterate_datasets

        #   iterate_datasets()返回一个迭代器，包含多个Pyg.Data（比如cora,citeseer等）。
        #   下游阶段只返回一个data，即Photo数据集。  Preprocess的作用是，对于给定的data，删除它的mask，并且把节点特征长度统一到100
        data = preprocess(next(iterate_datasets(dataset)))


        #   以photo数据集为例，data.y是一个长度为 7650的，表示每个节点的类别。 
        #   torch.unique(data.y) 返回的是[0,1,2,3,4,5,6,7] ，表示这个photo数据集是一个 8分类的节点分类问题    
        num_classes = torch.unique(data.y).size(0)



        #   train_dict_list是一个 字典，key是 0到7，每个关键字对应的值目前是一个空列表[]
        train_dict_list = {key.item():[] for key in torch.unique(data.y)}


        val_test_list = []
        target_graph_list = induced_graphs(data)    #   data(photo数据集)包含7650个节点，生成7650个诱导子图。（每个节点都生成一个以该节点为中心的诱导子图）


############################        目前学到了这里
        from torch.utils.data import random_split, Subset

        for index, graph in enumerate(target_graph_list):
            
            i_class = graph.y

            if( len(train_dict_list[i_class]) >= few_shot):
                val_test_list.append(graph)
            else:
                train_dict_list[i_class].append(index)
        
        all_indices = []
        for i_class, indice_list in train_dict_list.items():
            all_indices+=indice_list

        train_set = Subset(target_graph_list, all_indices)

        val_set, test_set = random_split(val_test_list, [0.1,0.9], torch.Generator().manual_seed(seed))
        

        
    # import pdb
    # pdb.set_trace()

    results = [
    {
        'train': train_set,
        'val': val_set,
        'test': test_set,
    }, 
        num_classes
    ]

    # save to cache
    torch.save(results, cache_path)

    return results