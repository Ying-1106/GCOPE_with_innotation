import torch
from torch_geometric.nn import global_add_pool, FAConv
from fastargs.decorators import param

class FAGCN(torch.nn.Module):

    def __init__(self, num_features, hidden, num_conv_layers, dropout, epsilon):
        super(FAGCN, self).__init__()
        self.global_pool = global_add_pool
        self.eps = epsilon              #   默认epsilon  为 0.1
        self.layer_num = num_conv_layers    #   默认  2 层
        self.dropout = dropout          #   默认dropout 为 0.2
        self.hidden_dim = hidden        #   默认128
        #   self.layers包含  多层的 FAConv
        self.layers = torch.nn.ModuleList() 
        for _ in range(self.layer_num):
            self.layers.append(FAConv(hidden, epsilon, dropout))

        self.t1 = torch.nn.Linear(num_features, hidden)
        self.t2 = torch.nn.Linear(hidden, hidden)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.t1.weight, gain=1.414)
        torch.nn.init.xavier_normal_(self.t2.weight, gain=1.414)

    @param('general.reconstruct')
    def forward(self, data, reconstruct):   #   这里data是 一个batch（包含10个增强后的图）

        x = data.x if data.x is not None else data.feat
        edge_index, batch = data.edge_index, data.batch #   edge_index是 10个图的 所有边，batch用来指明每个节点属于10个图中的哪个图

        h = torch.dropout(x, p=self.dropout, train=self.training)
        h = torch.relu(self.t1(h))  #   对初始节点特征（长度100）做一个线性变换，变成长度128
        h = torch.dropout(h, p=self.dropout, train=self.training)
        raw = h #   raw是初始特征经过一层Linear层变换后的  长度为128的特征（相当于 没经过GNN的初始特征）


        for i in range(self.layer_num): #   这是经过 多层GNN聚合，得到更新后的  节点特征 h 
            h = self.layers[i](h, raw, edge_index)
        h = self.t2(h)

        #   这就是ReadOut操作，这里有10个图，得到每个图的  图向量。
        graph_emb = self.global_pool(h, batch)

        if(reconstruct==0.0):   #   reconstruct == 0，这是下游微调阶段，这一阶段只需要把子图向量返回
            return graph_emb
        else:   # 返回多层GNN聚合后的节点特征 h  ， 以及ReadOut操作之后的 图向量
            return graph_emb, h


from fastargs.decorators import param

@param('model.backbone.hid_dim')
@param('model.backbone.fagcn.num_conv_layers')
@param('model.backbone.fagcn.dropout')
@param('model.backbone.fagcn.epsilon')
def get_model(num_features, hid_dim, num_conv_layers, dropout, epsilon):
    return FAGCN(num_features, hid_dim, num_conv_layers, dropout, epsilon)