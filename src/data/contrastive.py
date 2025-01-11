import time
from fastargs.decorators import param
import torch
from torch_geometric.data import Batch
import numpy as np

@param('general.cache_dir')
@param('pretrain.cross_link')
@param('pretrain.cl_init_method')
@param('pretrain.cross_link_ablation')
@param('pretrain.dynamic_edge')
@param('pretrain.dynamic_prune')
@param('pretrain.split_method')
def get_clustered_data(dataset, cache_dir, cross_link, cl_init_method='learnable', cross_link_ablation=False, dynamic_edge='none',dynamic_prune=0.0,split_method='RandomWalk'):
    

    #################              实验代码，
    #   测试结论：iterate_datasets（dataset)函数，会返回一个可迭代的generator，这个generator有2种访问方式:
    #   第一种：用list()把它转换成列表来访问。第二种：在for循环中用in generator来依次访问每个成员
    #   实际情况：这个generator包含了多个data，每个 data  都是一个  PyG.Data。
    #   PyG中的cora数据集：2708个节点（特征长度1433），10556条边，train集 140个节点，val集500个节点，test集1000个节点。如果train_mask/val_mask/test_mask的划分是和节点数量对齐，那么这就是一个节点分类问题，y就是节点标签
    # ite_result = list(iterate_datasets(dataset))
    # data_list_test = []
    # for data in ite_result:

    #     #   preprocess(data)效果是：把data（一个PyG图数据集）中节点特征长度  变到规定的统一长度。
    #     #   然后把train_mask等三个mask删除掉，因为预训练阶段不需要label
    #     processed_data = preprocess(data)
    #     data_list_test.append(processed_data)

########################

    #   dataset是["cora","citeseer"]。  iterate_datasets生成一个generator(一种迭代器，粗略理解为一个列表[]就行)，包含多个数据集
    from .utils import preprocess, iterate_datasets
    #   iterate_datasets(dataset)会返回一个iterator迭代器（可以粗略认为是一个列表[]），其中包含多个数据集（每个数据集都是一个Pyg.Data )
    #   preprocess(data)效果是：  把一个Pyg.Data中的 train_mask等删除（预训练不需要label，自然也不用划分训练集），并且把节点特征长度统一到  100



    #   根据上面的实验代码得出结论：以下这个返回的data_list，包含 n 个数据集（data）
    #   每个data的节点特征长度统一到固定长度，并且删除train_mask等，保留了label，但是预训练用不到label
    data_list = [preprocess(data) for data in iterate_datasets(dataset)]
    #   最终返回的data_list，就是一个[]，包含多个Pyg.Data
    #   data_list[0]  是  Cora 数据集 , 节点 ： [2708,100]，边 ：[2,10556]， 还有 y（label）:节点的类别标签
    #   data_list[1]  是  Computers 数据集 , 节点 ： [13752,100]，边 ：[2,491722]
    #   data_list[1]  是  Citeseer数据集，节点特征：[3327,100] ,  边：[2,9104]
    #   data_list[2]  是  Cornell 数据集 , 节点 ： [183,100]，边 ：[2,298]
########################################################################################################
    from torch_geometric.data import Batch  #   
    data = Batch.from_data_list(data_list)  #   把 多个graph  合并  成一个大图（各个子图之间没有连接）。（具体：把多个pyg.Data  构造成  一个pyg.Batch对象P)
    #   构造后的data  是一个  pyg.batch(大图)，节点特征 x :[16460,100] , edge_index（边） 是：[2 , 502278] 
    #   此时的节点数等于两个小图节点个数之和，边数等于2个小图的边数量之和。（可以看出，此时这个大图还不包含协调器，也不存在和协调器连接的边）
    #   data.batch是一个 [0,0,......,1,1.....]  :  16460个元素，前2708个是0（表示cora的节点），后13752个是1（表示computers的节点）
    
    from copy import deepcopy
    data_for_similarity_computation = deepcopy(data)
    print(f'Isolated graphs have total {data.num_nodes} nodes, each dataset added {cross_link} graph coordinators')


    gco_model = None

    if(cross_link > 0):
        num_graphs = data.num_graphs
        graph_node_indices = []

        for graph_index in range(num_graphs):
            node_indices = (data.batch == graph_index).nonzero(as_tuple=False).view(-1)
            graph_node_indices.append(node_indices)
        #   graph_node_indices[0] 是 tensor :[0到2707]。  即第1个小图Cora的   大图节点ID
        #   graph_node_indices[1] 是 tensor :[2708到16459]  。即第2个小图Computers的   大图节点ID
            

        new_index_list = [i for i in range(num_graphs)]*cross_link  #   只有2个图的时候，new_index_list == [0,1]
        
        if(cl_init_method == 'mean'):
            new_node_features = []
            for node_graph_index in new_index_list:
                node_indices_corresponding_graph = (data.batch == node_graph_index).nonzero(as_tuple=False).view(-1)
                new_node_features.append(torch.mean(data.x[node_indices_corresponding_graph],dim=0).reshape((1,-1)))

            new_node_features = torch.cat(new_node_features, dim=0)
            data.x = torch.cat([data.x, new_node_features], dim=0)
        elif(cl_init_method == 'sum'):
            new_node_features = []
            for node_graph_index in new_index_list:
                node_indices_corresponding_graph = (data.batch == node_graph_index).nonzero(as_tuple=False).view(-1)
                new_node_features.append(torch.sum(data.x[node_indices_corresponding_graph],dim=0).reshape((1,-1)))

            new_node_features = torch.cat([new_node_features], dim=0)
            data.x = torch.cat([data.x, new_node_features], dim=0)
        elif(cl_init_method == 'simple'):
            new_node_features = torch.ones((len(new_index_list),data.num_node_features))
            data.x = torch.cat([data.x, new_node_features], dim=0) 
        elif(cl_init_method == 'learnable'):
            from model.graph_coordinator import GraphCoordinator     

            #   定义协调器节点的初始化向量，geo_model.learnable_param就是一个列表[]， 包含了 k个长度为100的 一维向量(nn.Parameter)
            #   k个原始小图，每个图有一个长度为100的协调器节点
            #   因为设置了seed，所以每次跑代码，生成的协调器向量初始值都是一样的
            gco_model = GraphCoordinator(data.num_node_features,len(new_index_list))

            #   data是一个 Pyg.Batch（大图）， data.x包含了2个小图的全部节点特征。
            #   把2个协调器节点特征 加入到 原有的data.x当中。
            
            data.x = gco_model.add_learnable_features_with_no_grad(data.x)
            #   处理后的  data.x  是  6221 * 100的 节点特征。（所有 原始小图 节点特征 + k个协调器节点特征）
            #   Cora,Citeseer,Corenell三个原始图  节点总数加起来  6218. （不包含协调器），加上3个协调器共有6221个节点
      


        #   原来的data.batch是一个 [0,0,......,1,1.....,2,2]。6218个元素，
        #   前2708个是0（表示cora的节点），后3327个是1（表示computers的节点），最后183个是2
        #   这里的data.batch 指明了节点属于哪个原始图。0表示cora。  1表示citeseer,    2表示cornell
        #   可以得出结论，pyg中的data.batch用来指示 每一个节点属于哪一个图。这段代码用来表示节点 属于（cora,citeseer,cornell)哪一个图
        #   之后pretrain代码中，用data.batch来表示  每一个节点属于  一个batch中10个图  中的那一个
        
        data.batch = torch.cat([data.batch, torch.tensor([new_index_list]).squeeze(0)], dim=0)
        #   处理之后的data.batch在最后加了个[0,1,2]，代表： cora图的协调器节点， citeseer图的协调器节点,cornell图的协调器节点


        
        if(dynamic_edge=='none'):

        #   给大图data加入新的边：每个协调器节点  和  对应的小图原始节点都有  来、回2条边
            if(cross_link==1):   
                #   new_index_list = [0,1,2]
                for node_graph_index in new_index_list: #   node_graph_index 为0，表示第一个图Cora
                    #   data.batch == 0 ，表示属于Cora的节点。 ==1 表示Citeseer的节点。 ==2表示Corenell的节点

                    #   node_indices_corresponding_graph 是  Cora全部节点 + 对应的协调器节点  的大图ID。最后一项是协调器节点ID
                    node_indices_corresponding_graph = (data.batch == node_graph_index).nonzero(as_tuple=False).view(-1)  

                    #     Cora对应的协调器节点的  大图ID
                    new_node_index = node_indices_corresponding_graph[-1]   

                        #   new_node_index  是当前协调器节点ID， node_indices_corresponding_graph[:-1] 是当前原始图的原始节点ID
            #  让Cora的协调器节点 和 Cora每个原始节点，都连一条边 ，        [16460]         ,       [0,1,2....,2707]
                    new_edges = torch.cartesian_prod(torch.tensor([new_node_index]), node_indices_corresponding_graph[:-1])

                    #   把上面的新边加入大图中
                    data.edge_index = torch.cat([data.edge_index, new_edges.t()], dim=1)
                
                    #   把上面的边  的  反向边  ，也加入大图中。  也就是说，对于每个小图，图中原始节点和 协调器节点之间，有 来、回两条边
                    new_edges = torch.cartesian_prod(node_indices_corresponding_graph[:-1], torch.tensor([new_node_index]))
                    data.edge_index = torch.cat([data.edge_index, new_edges.t()], dim=1)
            else:
                for node_graph_index in new_index_list[:num_graphs]:
                    node_indices_corresponding_graph = (data.batch == node_graph_index).nonzero(as_tuple=False).view(-1)
                    new_node_index_list = node_indices_corresponding_graph[-1*cross_link:]
                    for new_node_index in new_node_index_list:
                        new_edges = torch.cartesian_prod(torch.tensor([new_node_index]), node_indices_corresponding_graph[:-1*cross_link])
                        data.edge_index = torch.cat([data.edge_index, new_edges.t()], dim=1)
                        
                        new_edges = torch.cartesian_prod(node_indices_corresponding_graph[:-1*cross_link], torch.tensor([new_node_index]))
                        data.edge_index = torch.cat([data.edge_index, new_edges.t()], dim=1)                                    
            
        #   给大图加边：任意2个协调器之间都有来、回2条边。
            if(cross_link_ablation==False):

                #   all_added_node_index  就是  所有协调器节点的大图ID：[16460,16461]
                all_added_node_index = [i for i in range(data.num_nodes-len(new_index_list),data.num_nodes)]

#               让每个当前协调器  都和其他协调器发出一条边。这样for循环结束后，每2个协调器直接都有来回2条边。(注意，最后一个协调器，没有向其他协调器发出边)
                for list_index, new_node_index in enumerate(all_added_node_index[:-1]):
                    #   new_node_index是当前的 协调器节点
                    
                    #   other_added_node_index_list是  所有 其他协调器
                    other_added_node_index_list = [index for index in all_added_node_index if index != new_node_index]


                    #   new_edges == [16460,16461]，是2个协调器之间的边。（具体是让当前协调器  发出一条指向其他协调器的边）
                    new_edges = torch.cartesian_prod(torch.tensor([new_node_index]), torch.tensor(other_added_node_index_list))
                    data.edge_index = torch.cat([data.edge_index, new_edges.t()], dim=1)


        elif(dynamic_edge=='internal_external'):
            all_added_node_index = [i for i in range(data.num_nodes-len(new_index_list),data.num_nodes)]         
            cross_dot = torch.mm(data.x, torch.transpose(data.x, 0, 1))
            cross_sim = torch.sigmoid(cross_dot) 
            cross_adj = torch.where(cross_sim < dynamic_prune, 0, cross_sim)

            all_cross_edge = cross_adj.nonzero().t().contiguous() 
           
            gco_edge_bool_signs = torch.isin(all_cross_edge[0], torch.tensor(all_added_node_index))
            gco_edge_indices = torch.where(gco_edge_bool_signs)[0]
            gco_cross_edge=all_cross_edge[:,gco_edge_indices]
            gco_cross_undirected_edge = torch.sort(gco_cross_edge, dim=0)[0]
            gco_cross_undirected_edge_np = gco_cross_undirected_edge.numpy()
            gco_cross_unique_edges = np.unique(gco_cross_undirected_edge_np, axis=1)

            print(f"Added Edge Num is {len(gco_cross_unique_edges[0])}")

            data.edge_index = torch.cat([data.edge_index, torch.tensor(gco_cross_unique_edges).contiguous()], dim=1)   
        elif(dynamic_edge=='similarity'):

            if(cross_link==1):            
                for node_graph_index in new_index_list:
                    node_indices_corresponding_graph = (data.batch == node_graph_index).nonzero(as_tuple=False).view(-1)
                    new_node_index = node_indices_corresponding_graph[-1]
                    
                    new_edges = torch.cartesian_prod(torch.tensor([new_node_index]), node_indices_corresponding_graph[:-1])
                    data.edge_index = torch.cat([data.edge_index, new_edges.t()], dim=1)
                    
                    new_edges = torch.cartesian_prod(node_indices_corresponding_graph[:-1], torch.tensor([new_node_index]))
                    data.edge_index = torch.cat([data.edge_index, new_edges.t()], dim=1)
            else:
                for node_graph_index in new_index_list[:num_graphs]:
                    node_indices_corresponding_graph = (data.batch == node_graph_index).nonzero(as_tuple=False).view(-1)
                    new_node_index_list = node_indices_corresponding_graph[-1*cross_link:]
                    for new_node_index in new_node_index_list:
                        new_edges = torch.cartesian_prod(torch.tensor([new_node_index]), node_indices_corresponding_graph[:-1*cross_link])
                        data.edge_index = torch.cat([data.edge_index, new_edges.t()], dim=1)
                        
                        new_edges = torch.cartesian_prod(node_indices_corresponding_graph[:-1*cross_link], torch.tensor([new_node_index]))
                        data.edge_index = torch.cat([data.edge_index, new_edges.t()], dim=1)     
                                    
            graph_mean_features = []
            for node_graph_index in new_index_list:
                node_indices_corresponding_graph_for_simi = (data_for_similarity_computation.batch == node_graph_index).nonzero(as_tuple=False).view(-1)              
                graph_mean_features.append(torch.mean(data_for_similarity_computation.x[node_indices_corresponding_graph_for_simi],dim=0).tolist())

            graph_mean_features = torch.tensor(graph_mean_features)
            cross_dot = torch.mm(graph_mean_features, torch.transpose(graph_mean_features, 0, 1))
            cross_sim = torch.sigmoid(cross_dot) 
            cross_adj = torch.where(cross_sim < dynamic_prune, 0, cross_sim)

            all_cross_edge = cross_adj.nonzero().t().contiguous()
            all_cross_edge += data_for_similarity_computation.num_nodes
            
            total_edge_num_before_add_cross_link_internal_edges = data.edge_index.shape[1]
            data.edge_index = torch.cat([data.edge_index, all_cross_edge], dim=1)               
            if((data.edge_index.shape[1]-total_edge_num_before_add_cross_link_internal_edges)==all_cross_edge.shape[1]):
                print(f'Edge num after gco connected together{data.edge_index.shape[1]}, totally add {all_cross_edge.shape[1]} inter-dataset edges')

    print(f'Unified graph has {data.num_nodes} nodes, each graph includes {cross_link} graph coordinators')

    raw_data = deepcopy(data)


#   此时的 data是 1个大图：包含cora,citeseer,cornell三个小图。  一共6221个节点（包含3个协调器），32398条边（包括原始边、协调器和原始图节点的边、协调器之间的边）
###############         上面有个问题，那就是为什么最后一个协调器不向其他协调器发射边
    


#########################
    if(split_method=='RandomWalk'):
        from torch_cluster import random_walk
        split_ratio = 0.1   #   大图节点总数 中  选择 10 %  的节点
        walk_length = 30
        all_random_node_list = torch.randperm(data.num_nodes)   #   得到大图所有节点ID，乱序的
        selected_node_num_for_random_walk = int(split_ratio * data.num_nodes)   # 大图节点总数 中  选择 10 %  
        random_node_list = all_random_node_list[:selected_node_num_for_random_walk] #   这就是从大图所有节点中（包括协调器）  选出的  10%  的随机节点
        walk_list = random_walk(data.edge_index[0], data.edge_index[1], 
                                start=random_node_list,     # 这些节点作为  起点
                                walk_length=walk_length)

        graph_list = [] 
        skip_num = 0        
        for walk in walk_list:   # walk是一条 长度为 31  的节点路径（游走路径）
            subgraph_nodes = torch.unique(walk) #  subgraph_nodes  是这条路径上的  所有节点ID(  例如有23个节点ID    )
            if(len(subgraph_nodes)<5):
                skip_num+=1
                continue
            #   data是 1个 大图（包含协调器  和 与协调器有关的边）
            subgraph_data = data.subgraph(subgraph_nodes)   # 用这些节点ID ，从大图中构造一个诱导子图：诱导子图中的节点是上述节点，边是  两端点都是上述节点的边

            graph_list.append(subgraph_data)

        print(f"Total {len(graph_list)} subgraphs with nodes more than 5, and there are {skip_num} skipped subgraphs with nodes less than 5.")

        #   graph_list是  若干个  诱导子图
        #   gco_model.learnable_param就是  所有协调器节点的向量
        #   raw_data是  大图（包含协调器节点，和与协调器有关的边）
        return graph_list, gco_model, raw_data  #   raw_data.x包含6221个节点向量，最后3个向量  和 gco_model.learnable_param是相同的。都是代表3个协调器向量

def update_graph_list_param(graph_list, gco_model):
    
    count = 0
    for graph_index, graph in enumerate(graph_list):
        for index, param_value in enumerate(gco_model.last_updated_param):
            match_info = torch.where(graph.x==param_value)
            if(match_info[0].shape[0]!=0):
                target_node_indice = match_info[0].unique()[-1].item()
                graph.x[target_node_indice] = gco_model.learnable_param[index].data
                count+=1
    updated_graph_list = graph_list
    return updated_graph_list    