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

        return self.answering(self.backbone(data))  #   经过多层fagcn，得到输出结果
                                            

def get_model(
        backbone_kwargs,    #  {"name" : "fagcn", "num_features" : 100  } 
        answering_kwargs = None,
        saliency_kwargs = None,
    ):
                #               model.backbone.fagcn文件中的  FAGCN类： 默认包含 2层  FAGCN层
    backbone = import_module(f'model.backbone.{backbone_kwargs.pop("name")}').get_model(**backbone_kwargs)

    #   torch.nn.Identity()是一个空操作层，它的forward(input) 不会对input做出任何改变
    answering = torch.nn.Identity() if answering_kwargs is None else import_module(f'model.answering.{answering_kwargs.pop("name")}').get_model(**answering_kwargs)
    saliency = torch.nn.Identity() if saliency_kwargs is None else import_module(f'model.saliency.{saliency_kwargs.pop("name")}').get_model(**saliency_kwargs)

    return Model(backbone, answering, saliency)
