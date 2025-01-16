import torch


class MLPAnswering(torch.nn.Module):
    def __init__(self, hid_dim, num_class, answering_layer_num):
        super().__init__()
        self.answering_layer_num = answering_layer_num  #   2，表示2层MLP
        self.num_class = num_class  #   8，表示8分类的节点分类问题（下游阶段）
        
        self.answering = torch.nn.ModuleList()
        self.bns_answer = torch.nn.ModuleList()

        for i in range(answering_layer_num-1):
            self.bns_answer.append(torch.nn.BatchNorm1d(hid_dim))
            self.answering.append(torch.nn.Linear(hid_dim,hid_dim)) #   从hid_dim  到  hid_dim的  线性变换层
        
        self.bn_hid_answer = torch.nn.BatchNorm1d(hid_dim)
        self.final_answer = torch.nn.Linear(hid_dim, num_class)     #   这是从 hid_dim  到  num_class的最后输出

        # BN initialization.
        for m in self.modules():
            if isinstance(m, (torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0.0001)        
    
    def forward(self, x):
        
        for i, lin in enumerate(self.answering):
            x = self.bns_answer[i](x)
            x = torch.relu(lin(x))  #   做一个 hid -> hid的变换
            
        x = self.bn_hid_answer(x)
        x = self.final_answer(x)    #   做一次 hid -> num_class的变换（8分类问题）
        prediction = torch.log_softmax(x, dim=-1)
        return prediction
    

from fastargs.decorators import param

@param('model.backbone.hid_dim')
@param('model.answering.mlp.num_layers')
def get_model(hid_dim, num_class, num_layers):
    return MLPAnswering(hid_dim, num_class, num_layers)