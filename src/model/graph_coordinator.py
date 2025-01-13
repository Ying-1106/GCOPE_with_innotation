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
        #   3个协调器节点向量
        graph_coordinator_features = torch.cat([p.data.reshape((1,-1)) for p in self.learnable_param], dim=0)

        #   把协调器节点向量 和  所有原始小图节点向量  都拼到一起。 返回的updated_feats 是大图 节点向量（包含协调器）
        updated_features = torch.cat([original_node_features, graph_coordinator_features], dim=0)
        return updated_features



    #   这个函数传入一个batch（例如包含10个图），效果就是把图中的节点特征中的 协调器向量（一开始是不可学习的固定向量）变成  可学习的向量
    def forward(self, batch_with_no_grad_node_features):
                        #   这里的batch不带梯度，不带梯度意思就是向量固定（不可改变），有梯度就代表着可以改变（可学习）
        count = 0
        graph_index_list = [x for x in set(batch_with_no_grad_node_features.batch.tolist())]    #   graph_index_list == 0~9，表示这一个batch中的 9个图
        for graph_index in graph_index_list:
            #   例如graph_index == 0，表示这一个batch中的第 0个图。 node_indices_corre_graph == [0~18]，表示这个图上的 节点ID
            node_indices_corresponding_graph = (batch_with_no_grad_node_features.batch == graph_index).nonzero(as_tuple=False).view(-1)
            for node_indice in reversed(node_indices_corresponding_graph):
                #   例如node_indice == 18 时，表示的是这第0个图的19个节点中的  18号节点
                #   self.last_updated_param  是  3个  长为100 的协调器向量
                for index, param_value in enumerate(self.last_updated_param):
                    if(torch.equal(batch_with_no_grad_node_features.x[node_indice], param_value)):  #   比较 这个节点特征  和  协调器向量  是否相等
                        #     如果相等的话，就说明当前节点（例如18号节点） 恰好就是  协调器（第1个协调器，即cora的协调器）。
                        #   如果相等，那么就把 这个协调器向量   赋值给  当前节点向量  。
                        #   虽然这两个值原来就相等，但是之前比较的时候，last_updated_param是 不可学习的固定向量。现在的learnable_param是可学习的向量。
                        batch_with_no_grad_node_features.x[node_indice] = self.learnable_param[index]
                        count+=1
        batch_with_learnable_param = batch_with_no_grad_node_features
        return batch_with_learnable_param

    def update_last_params(self):
        self.last_updated_param = [deepcopy(param.data) for param in self.learnable_param]

if __name__ == '__main__':
    model = GraphCoordinator(num_node_features=10, num_graph_coordinators=5)
    print(model)