from fastargs.decorators import param
import numpy as np
import torch
from copy import deepcopy
import os

@param('general.save_dir')
@param('data.name', 'dataset')
@param('model.backbone.model_type', 'backbone_model')
@param('model.saliency.model_type', 'saliency_model')
@param('pretrain.method')
@param('pretrain.noise_switch')
def run(
    save_dir,
    dataset,    #   ["cora","citeseer","cornell"]
    backbone_model,
    saliency_model,
    method,
    noise_switch,
    ):
     
    if(saliency_model == 'mlp'):    
        # load data
        from data import get_clustered_data
        data = get_clustered_data(dataset)

        # init model
        from model import get_model
        model = get_model(
            backbone_kwargs = {
                'name': backbone_model,
                'num_features': data[0].x.size(-1),
            },
            saliency_kwargs = {
                'name': saliency_model,
                'feature_dim': data[0].x.size(-1),
            } if saliency_model != 'none' else None,
        )
    else:
        # load data
        from data import get_clustered_data

        with torch.no_grad():   #   dataset是["cora","citeseer"]
           
        #   data是  若干个  诱导子图 （从大图中采样得来的）构成的列表
        #   gco_model.learnable_param  是  所有协调器节点的向量
        #   raw_data是  大图（包含协调器节点，和与协调器有关的边）。 raw_data.x[-3（或者-2,-1）]就是最后3个节点向量，也就是3个协调器向量
            data, gco_model, raw_data = get_clustered_data(dataset)  #   raw_data.x包含6221个节点向量，最后3个向量  和 gco_model.learnable_param是相同的。都是代表3个协调器向量

        # init model
        from model import get_model
        model = get_model(
            backbone_kwargs = {
                'name': backbone_model, #   默认值：'fagcn'
                'num_features': data[0].x.size(-1), #   统一后的特征长度：默认值 100
            },
            saliency_kwargs = {
                'name': saliency_model,
                'feature_dim': data[0].x.size(-1),
            } if saliency_model != 'none' else None,    #   默认值是  None
        )                
        #   返回的  model.backbone就是一个2层的FAGCN层
        #   model.forward(input)  就是让 input  经过 backbone跑一遍

    # train
    if method == 'graphcl':            
        #   data： 622个诱导子图构成的列表，每个data[k]是一个 databatch，例如：节点特征[23,100]，边[2,72]
        #   model.backbone 就是一个 2层的 FAGCN
        #   geo_model.learnable_param就是一个列表[],包含3个协调器节点向量
        #   raw_data是大图（包含协调器节点，和与协调器相连的边）
        model = graph_cl_pretrain(data, model, gco_model, raw_data)
    elif method == 'simgrace':
        model = simgrace_pretrain(data, model, gco_model, raw_data)
    else:
        raise NotImplementedError(f'Unknown method: {method}')

    # save
    import os

    torch.save(model.state_dict(), os.path.join(save_dir, ','.join(dataset)+'_pretrained_model.pt'))



#################################################
    
@param('pretrain.learning_rate')
@param('pretrain.weight_decay')
@param('pretrain.epoch')
@param('pretrain.cross_link')
@param('pretrain.cl_init_method')
@param('general.reconstruct')
@param('pretrain.split_method')
@param('pretrain.dynamic_edge')
def graph_cl_pretrain(
    data,   #   data： 622个诱导子图构成的列表，每个data[k]是一个 databatch，例如：节点特征[23,100]，边[2,72]
    model,  #   model.backbone 就是一个 2层的 FAGCN
    gco_model,#     geo_model.learnable_param就是一个列表[],包含3个协调器节点向量
    raw_data,
    learning_rate,
    weight_decay,
    epoch,  #   默认100
    cross_link, #   默认 1。每个原始图  有一个协调器，  协调器和原始图节点、协调器之间  都有边
    cl_init_method, #   默认 "learnable"，意思是用nn.Parameter初始化一个随机的长为100的向量作为协调器节点
    reconstruct,    #   默认 0.2
    dynamic_edge,
    split_method,   #   默认Random_walk。用随机游走的方式构造  诱导子图
    ):

    
    @param('pretrain.batch_size')
    def get_loaders(data, batch_size):
        #       data： 622个诱导子图构成的列表，每个data[k]是一个 databatch，例如：节点特征[23,100]，边[2,72]
        import random
        from torch_geometric.data import Data
        from torch_geometric.loader import DataLoader
        from algorithm.graph_augment import graph_views
                                #       删除节点， 边扰动  ，属性屏蔽。三者选2个，作为增强策略组合
        augs, aug_ratio = random.choices(['dropN', 'permE', 'maskN'], k=2), random.randint(1, 3) * 1.0 / 10
        #   例如，augs == [删除节点，边扰动]  ，aug_ratio == 0.2 
        view_list_1 = []
        view_list_2 = []
        for g in data:
            #   g  是  1个  诱导子图。

            #   用augs[0]（即第一种增强方式 ：删除节点） 来构造一个增强后视图
            #   例如 g 原来是 23个节点特征，72条边  。增强之后得到的view_g包含19个节点（删除20%的节点），58条边（删除与这些节点相连的边）
            view_g = graph_views(data=g, aug=augs[0], aug_ratio=aug_ratio)
            #   Pyg.Data代表一个同质图。给出节点特征(节点数 * 特征长度)  +  边 （2*边数） ，就能代表一个同质图
            view_list_1.append(Data(x=view_g.x, edge_index=view_g.edge_index))

            #   用augs[1]（即第二种增强方式 ，边扰动） 来构造一个增强后视图
            #   这里的 g 其实是 已经删除部分节点之后 的图（增强过的图）
            view_g = graph_views(data=g, aug=augs[1], aug_ratio=aug_ratio)
            view_list_2.append(Data(x=view_g.x, edge_index=view_g.edge_index))
        #   包含622个  删节点增强后  的视图，每次取10个，组成一个batch
        loader1 = DataLoader(view_list_1, batch_size=batch_size, shuffle=False,
                                num_workers=4)  
        #   包含622个  删除边增强后  的视图，每次取10个。和上面的loader1  一一对应。
        loader2 = DataLoader(view_list_2, batch_size=batch_size, shuffle=False,
                                num_workers=4)  

#   测试代码
        # for graph_ in loader1:
        #     #   经过测试得出，loader_1用for循环迭代一次，取出的成员是  10个图组成的一个DataBatch
        #     pass

        # for graph_ in loader2:

        #     pass

        return loader1, loader2

    class ContrastiveLoss(torch.nn.Module):
        def __init__(self, hidden_dim, temperature=0.5):
            super(ContrastiveLoss, self).__init__()
            #   对比任务头，包含2层 MLP。在GraphCL中，这个head似乎没有用到
            self.head = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(hidden_dim, hidden_dim),
            )
            self.temperature = temperature

        def forward(self, zi, zj):  #   有10个原始图，zi就是删除节点后的10个增强图的图向量，zj就是10个删除边后的增强图的图向量
            batch_size = zi.size(0) #   10
            x1_abs = zi.norm(dim=1) #   计算  每个图向量的  Frobenius 范数(即L2范数，就是根号下  每个向量元素的平方的和)
            x2_abs = zj.norm(dim=1)
            sim_matrix = torch.einsum('ik,jk->ij', zi, zj) / torch.einsum('i,j->ij', x1_abs, x2_abs)
            sim_matrix = torch.exp(sim_matrix / self.temperature)
            #   得到的sim_matrix计算了  zi中每个向量  与  zj中每个向量的  余弦相似度
            

            #   这是每对正例的相似度，即zi[0]和zj[0]的相似度，zi[1]和zj[1]的相似度，zi[2]和zj[2]的相似度，.。。。。zi[9]和zj[9]的相似度，
            pos_sim = sim_matrix[range(batch_size), range(batch_size)]

            #   以zi的10个向量为中心，每个向量 都有一个对应的LOSS，一共有10个LOSS值
            loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)  #   分子上是1对正例相似度，分母上是9对负例的相似度
            loss = - torch.log(loss).mean()

            



            return loss

    class ReconstructionLoss(torch.nn.Module):
        def __init__(self, hidden_dim, feature_num):
            super(ReconstructionLoss, self).__init__()
            #   2层MLP
            self.decoder = torch.nn.Sequential(#   hidden_dim == 128 。 feature_dim == 100
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(hidden_dim, feature_num),
            )

            self.loss_fn = torch.nn.MSELoss()

        def forward(self, input_features, hidden_features): #   input_features是原始节点特征（长度100），hidden_features是GNN聚合更新后的节点特征（长度128）
            reconstruction_features = self.decoder(hidden_features) #   GNN聚合后的节点特征（长度128）  ，再经过2层MLP（decoder），长度变换到100
            return self.loss_fn(input_features, reconstruction_features)


    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')   
    #   初始化包括了 2层 MLP
    #   model.backbone.hidden_dim == 128  。
    loss_fn = ContrastiveLoss(model.backbone.hidden_dim).to(device)
    loss_fn.train(), model.to(device).train()
    best_loss = 100000.
    best_model = None
    if(gco_model==None):
        if(reconstruct==0.0):
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, list(model.parameters()) + list(loss_fn.parameters())),
                lr=learning_rate,
                weight_decay=weight_decay
                )
        else:
            rec_loss_fn = ReconstructionLoss(model.backbone.hidden_dim, data[0].num_node_features).to(device)
            rec_loss_fn.train()
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, list(model.parameters()) + list(loss_fn.parameters()) +list(rec_loss_fn.parameters())),
                lr=learning_rate,
                weight_decay=weight_decay
                )                
    else:
        if(reconstruct==0.0):
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, list(gco_model.parameters()) + list(model.parameters()) + list(loss_fn.parameters())),
                lr=learning_rate,
                weight_decay=weight_decay
                )
        else:   # here.  rec_loss_fn  包含 2层 MLP
            rec_loss_fn = ReconstructionLoss(model.backbone.hidden_dim, data[0].num_node_features).to(device)
            rec_loss_fn.train()
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, list(gco_model.parameters()) +    #   3个协调器向量
                        list(model.parameters()) +  #   model.backbone(2层FAGCN)的参数
                        list(loss_fn.parameters()) +    #  对比LOSS的 2层MLP，GraphCL中不会用到这个head，所以这个参数不会改变
                        list(rec_loss_fn.parameters())),    # 2层MLP，这个主要是把GNN更新后的特征（长度128）  长度还原到原始长度100，这个参数会用到。
                lr=learning_rate,
                weight_decay=weight_decay
                )            

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch)

    from torchmetrics import MeanMetric
    from tqdm import tqdm   #   用于打印进度条
    from data.contrastive import update_graph_list_param    
    loss_metric = MeanMetric()

    for e in range(epoch):
        
        loss_metric.reset()

        if(cross_link > 0 and cl_init_method == 'learnable'):
            if(split_method=='RandomWalk'):
                last_updated_data = deepcopy(data)  #   data： 622个诱导子图构成的列表，每个data[k]是一个 databatch，例如：节点特征[23,100]，边[2,72]

            loaders = get_loaders(data) #   data： 622个诱导子图构成的列表
        elif(e==0):
            loaders = get_loaders(data)

        pbar = tqdm(zip(*loaders), total=len(loaders[0]), ncols=100, desc=f'Epoch {e}, Loss: inf')
                
        for batch1, batch2 in pbar:
            #   batch1包含10个  删节点增强后的图  ，batch2包含10个  删除边增强后的图
            #   上一个batch1，batch2 会在最后更新 协调器，下面这个操作就是把  更新后的协调器向量  赋值  给 这个batch中的协调器。（相当于让这一个batch的协调器也得到更新）
            if(gco_model!=None):    #   以下这个操作，是把batch中的协调器向量  从不可改变的固定向量  变成  可学习的向量
                batch1 = gco_model(batch1)  #   一个batch1包含10个图，batch1.batch取值为0~9，标明了属于10张图中的哪一个
                batch2 = gco_model(batch2)    

            optimizer.zero_grad()

            if(reconstruct==0.0):
                zi, zj = model(batch1.to(device)), model(batch2.to(device))
                loss = loss_fn(zi, zj)
            else:               
                zi, hi = model(batch1.to(device))   # zi是10个ReadOut操作之后的图向量，  hi是 batch1中的10个图的所有节点的  GNN聚合后的节点特征，
                zj, hj = model(batch2.to(device))
                                                    
                                                    
                                                    #   batch1.x是 没经过GNN的原始特征， hi是经过GNN之后的节点特征
                loss = loss_fn(zi, zj) + reconstruct*(rec_loss_fn(batch1.x, hi) + rec_loss_fn(batch2.x, hj))
                #   loss_fn是对比LOSS，目的是希望 正例之间相似度高，负例之间相似度低，这样对比LOSS就会更小
                #   rec_loss_fn是重构LOSS，目的是希望GNN更新后的特征  和  原始节点特征  的差距更小（保留更多的原始信息），这样LOSS会更小。
                
            loss.backward()
            optimizer.step()    #   更新GNN参数，协调器向量，更新rec_loss的decoder
            
            loss_metric.update(loss.item(), batch1.size(0))
            pbar.set_description(f'Epoch {e}, Loss {loss_metric.compute():.4f}', refresh=True)



############    目前学到了这里

        if(gco_model!=None):
            data  = update_graph_list_param(last_updated_data, gco_model)
            gco_model.update_last_params()

        # lr_scheduler.step()
        
        if(loss_metric.compute()<best_loss):
            best_loss = loss_metric.compute()
            best_model = deepcopy(model)
            
        pbar.close()
        
    return best_model







##########################################################

@param('pretrain.learning_rate')
@param('pretrain.weight_decay')
@param('pretrain.epoch')
@param('pretrain.cross_link')
@param('pretrain.cl_init_method')
@param('general.reconstruct')
@param('pretrain.split_method')
@param('pretrain.dynamic_edge')
@param('pretrain.batch_size')
def simgrace_pretrain(
    data,
    model,
    gco_model,
    raw_data,
    learning_rate,
    weight_decay,
    epoch,
    cross_link,
    cl_init_method,
    reconstruct,
    dynamic_edge,
    split_method,
    batch_size,
    ):

    from torch_geometric.loader import DataLoader
    from data import gen_ran_output

    class SimgraceLoss(torch.nn.Module):
        def __init__(self, gnn, hidden_dim, temperature=0.5):
            super(SimgraceLoss, self).__init__()
            self.gnn = gnn
            self.projection_head = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(hidden_dim, hidden_dim),
            )
            self.temperature = temperature

        
        @param('general.reconstruct')
        def forward_cl(self, data, reconstruct):
        
            if(reconstruct==0.0):
                zi = self.gnn(data)
                zi = self.projection_head(zi)

                return zi
            else:
                zi, hi = self.gnn(data)
                zi = self.projection_head(zi)
            
                return zi, hi
        
        def loss_cl(self, zi, zj):
            batch_size = zi.size(0)
            x1_abs = zi.norm(dim=1)
            x2_abs = zj.norm(dim=1)
            sim_matrix = torch.einsum('ik,jk->ij', zi, zj) / torch.einsum('i,j->ij', x1_abs, x2_abs)
            sim_matrix = torch.exp(sim_matrix / self.temperature)
            pos_sim = sim_matrix[range(batch_size), range(batch_size)]
            loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
            loss = - torch.log(loss).mean()
            return loss

    class ReconstructionLoss(torch.nn.Module):
        def __init__(self, hidden_dim, feature_num):
            super(ReconstructionLoss, self).__init__()
            self.decoder = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(hidden_dim, feature_num),
            )

            self.loss_fn = torch.nn.MSELoss()

        def forward(self, input_features, hidden_features):
            reconstruction_features = self.decoder(hidden_features)
            return self.loss_fn(input_features, reconstruction_features)


    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')   
    

    loss_fn = SimgraceLoss(model.backbone, model.backbone.hidden_dim).to(device)
    loss_fn.train(), model.to(device).train()
    best_loss = np.inf
    best_model = None
    if(gco_model==None):
        if(reconstruct==0.0):
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, list(model.parameters()) + list(loss_fn.parameters())),
                lr=learning_rate,
                weight_decay=weight_decay
                )
        else:
            rec_loss_fn = ReconstructionLoss(model.backbone.hidden_dim, data[0].num_node_features).to(device)
            rec_loss_fn.train()
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, list(model.parameters()) + list(loss_fn.parameters()) +list(rec_loss_fn.parameters())),
                lr=learning_rate,
                weight_decay=weight_decay
                )                
    else:
        if(reconstruct==0.0):
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, list(gco_model.parameters()) + list(model.parameters()) + list(loss_fn.parameters())),
                lr=learning_rate,
                weight_decay=weight_decay
                )
        else:
            rec_loss_fn = ReconstructionLoss(model.backbone.hidden_dim, data[0].num_node_features).to(device)
            rec_loss_fn.train()
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, list(gco_model.parameters()) + list(model.parameters()) + list(loss_fn.parameters()) +list(rec_loss_fn.parameters())),
                lr=learning_rate,
                weight_decay=weight_decay
                )            

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch)

    from torchmetrics import MeanMetric
    from tqdm import tqdm
    from data.contrastive import update_graph_list_param
    loss_metric = MeanMetric()

    for e in range(epoch):
        
        loss_metric.reset()

        if(cross_link > 0 and cl_init_method == 'learnable'):
            if(split_method=='RandomWalk'):
                last_updated_data = deepcopy(data)

            loader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=1) 
        elif(e==0):
            loader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=1) 

        pbar = tqdm(loader, total=len(loader), ncols=100, desc=f'Epoch {e}, Loss: inf')

        for batch1 in pbar:
            if(gco_model!=None):
                batch1 = gco_model(batch1)

            optimizer.zero_grad()

            if(reconstruct==0.0):
                batch1 = batch1.to(device)
                zi = loss_fn.forward_cl(batch1)                
                zj = gen_ran_output(batch1, loss_fn)
                zj = zj.detach().data.to(device)
                loss = loss_fn.loss_cl(zi, zj)              
            else:

                batch1 = batch1.to(device)
                zi, hi = loss_fn.forward_cl(batch1)
                zj, hj = gen_ran_output(batch1, loss_fn)
                zj = zj.detach().data.to(device)
                loss = loss_fn.loss_cl(zi, zj) + reconstruct*(rec_loss_fn(batch1.x, hi) + rec_loss_fn(batch1.x, hj))
                
            loss.backward()
            optimizer.step()
            
            loss_metric.update(loss.item(), batch1.size(0))
            pbar.set_description(f'Epoch {e}, Loss {loss_metric.compute():.4f}', refresh=True)

        if(gco_model!=None):
            data  = update_graph_list_param(last_updated_data, gco_model)
            gco_model.update_last_params()

        # lr_scheduler.step()
        
        if(loss_metric.compute()<best_loss):
            best_loss = loss_metric.compute()
            best_model = deepcopy(model)
            
        pbar.close()
        
    return best_model
    
