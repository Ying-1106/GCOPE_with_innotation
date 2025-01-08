import torch
import torch.nn as nn
from copy import deepcopy

class GraphCoordinator(nn.Module):
    def __init__(self, num_node_features, num_graph_coordinators):
        super(GraphCoordinator, self).__init__()
        #   如果有k个图，那么self.learnable_param就是一个  列表[  ]，其中包含  k个  长度为num_node_features 的向量（就是协调器节点），节点是正态分布初始化的随机值
        self.learnable_param = nn.ParameterList(    
            [nn.Parameter(torch.randn(num_node_features)) for _ in range(num_graph_coordinators)]   
            )
        self.last_updated_param = [deepcopy(param.data) for param in self.learnable_param]
    

    #   把所有小图的全部节点特征，和协调器节点特征，整合到一起
    def add_learnable_features_with_no_grad(self, original_node_features):
        #   协调器节点向量
        graph_coordinator_features = torch.cat([p.data.reshape((1,-1)) for p in self.learnable_param], dim=0)

        #   把协调器节点向量 和  所有原始小图节点向量  都拼到一起。 返回的updated_feats 是大图 节点向量（包含协调器）
        updated_features = torch.cat([original_node_features, graph_coordinator_features], dim=0)
        return updated_features

    def forward(self, batch_with_no_grad_node_features):
        count = 0
        graph_index_list = [x for x in set(batch_with_no_grad_node_features.batch.tolist())]
        for graph_index in graph_index_list:
            node_indices_corresponding_graph = (batch_with_no_grad_node_features.batch == graph_index).nonzero(as_tuple=False).view(-1)
            for node_indice in reversed(node_indices_corresponding_graph):
                for index, param_value in enumerate(self.last_updated_param):
                    if(torch.equal(batch_with_no_grad_node_features.x[node_indice], param_value)):
                        batch_with_no_grad_node_features.x[node_indice] = self.learnable_param[index]
                        count+=1
        batch_with_learnable_param = batch_with_no_grad_node_features
        return batch_with_learnable_param

    def update_last_params(self):
        self.last_updated_param = [deepcopy(param.data) for param in self.learnable_param]

if __name__ == '__main__':
    model = GraphCoordinator(num_node_features=10, num_graph_coordinators=5)
    print(model)