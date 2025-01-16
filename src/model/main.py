from importlib import import_module
import torch

#   Model.backbone就是主要模型（一个2层的FAGCN）
class Model(torch.nn.Module):
    def __init__(
                self,
                backbone,
                answering = torch.nn.Identity(),
                saliency = torch.nn.Identity(),
            ):
        super().__init__()
        self.backbone = backbone    #   fagcn 2层模型
        self.answering = answering  #   空操作层
        self.saliency = saliency    #   空操作层

    def forward(self, data):    #   让 data  经过  backbone（2层FAGCN)跑一次

        data.x = self.saliency((data.x))    #   空操作


        #   在预训练阶段，model.forward会让节点特征会只经过2层GNN的变换(backbone)
        #   在下游阶段，model.forward会让节点特征，先经过2层GNN变换，再经过answering（任务头）的变换
        return self.answering(self.backbone(data))  #   经过多层fagcn，得到输出结果
        #   注意：backbone(GNN).forward 会先通过GNN聚合更新节点向量，然后再ReadOut得到图向量                                    

def get_model(
        backbone_kwargs,    #  {"name" : "fagcn", "num_features" : 100  } 
        answering_kwargs = None,    #   仅在  下游微调阶段 使用
        saliency_kwargs = None, #   预训练和微调阶段都用不到
    ):
                #               model.backbone.fagcn文件中的  FAGCN类： 默认包含 2层  FAGCN层
    backbone = import_module(f'model.backbone.{backbone_kwargs.pop("name")}').get_model(**backbone_kwargs)

    #   torch.nn.Identity()是一个空操作层，它的forward(input) 不会对input做出任何改变

    #   仅在微调阶段使用answering（2层MLP），本质上是下游任务头。  
    #   在下游阶段，初始节点特征（100）会首先经过backbone（2层GNN）的变换，变到长度128
    #   然后会经过 answering的变换， 变成长度为8 的  预测类别概率
    answering = torch.nn.Identity() if answering_kwargs is None else import_module(f'model.answering.{answering_kwargs.pop("name")}').get_model(**answering_kwargs)
    
    #   saliency在预训练和  微调阶段  都不会用到
    saliency = torch.nn.Identity() if saliency_kwargs is None else import_module(f'model.saliency.{saliency_kwargs.pop("name")}').get_model(**saliency_kwargs)

    return Model(backbone, answering, saliency)
