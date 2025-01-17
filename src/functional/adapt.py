import copy
from fastargs.decorators import param
import torch


@param('data.name', 'dataset')
@param('adapt.batch_size')
@param('data.supervised.ratios')
@param('adapt.method')
@param('model.backbone.model_type', 'backbone_model')
@param('model.saliency.model_type', 'saliency_model')
@param('model.answering.model_type', 'answering_model')
@param('adapt.pretrained_file')
@param('general.save_dir')
@param('adapt.repeat_times')
def run(
    dataset,    #   "photo"，下游任务阶段，只用1个graph
    batch_size, # 100
    ratios, #       0.1， 0.1 ， 0.8 这是训练集、验证集、测试集的划分
    method, #   一般是finetune
    backbone_model, #   fagcn
    saliency_model, #   none
    answering_model,    #   mlp，是一个2层MLP
    pretrained_file,    #   预训练好的 GNN模型的 存放位置
    save_dir,
    repeat_times,   #   5
    ):
    
    # load data
    from data import get_supervised_data
    from torch_geometric.loader import DataLoader

    #   get_supervised_data("photo",ratios = [0.1,0.1,0.8])   的作用
    #   首先下载到photo数据集（一共7650个节点），给每个节点都以它为中心构造一个诱导子图，一共7650个诱导子图
    #   然后，训练集中：每个类别，有一个对应的诱导子图（子图的中心节点就属于该类别，这是一个节点分类问题）
    #   剩下的7642个子图，按照1比9的比例，分配给val_set（765个子图）和test_set（6877个子图）
    datasets, num_classes = get_supervised_data(dataset[0], ratios=ratios)


    #           batch_size == 100。 datasets包括 train：8个图，val:765个图，test:6877个图
    loaders = { k: DataLoader(v, batch_size=batch_size, shuffle=True, num_workers=4) for k, v in datasets.items() }
    #   k   :  v
    #   "train" : 8个图，
    #   "val" : 765个图，每个batch包含100个图，
    #   "test" : 6877个图，每个batch包含100个图


    # init model
    from model import get_model
    model = get_model(
        backbone_kwargs = {
            'name': backbone_model,
            'num_features': datasets['train'][0].x.size(-1),
        },
        answering_kwargs = {
            'name': answering_model,    #   MLP
            'num_class': num_classes,   #   8，表示8分类问题，也是MLP最后输出的大小
        },
        saliency_kwargs = {
            'name': saliency_model,
            'feature_dim': datasets['train'][0].x.size(-1),
        } if saliency_model != 'none' else None,
    )


    #   取出预训练好的模型
    model.load_state_dict(torch.load(pretrained_file,map_location=lambda storage, loc: storage.cuda(0)), strict=False)

    # train
    all_results = []
    for _ in range(repeat_times):   #   一共做5次微调，这5次微调彼此之间完全独立，没有任何关联。
        if method == 'finetune':
            results = finetune(loaders, model)      #   返回的results包含 在测试集上的  ACC。代表这一次微调的结果
        elif method == 'prog':
            from model import get_prompt_model
            # statistic the average node number of dataset
            total_graph = sum([len(v) for k, v in datasets.items()])
            train_node_num = sum([g.num_nodes for g in datasets['train']])
            val_node_num = sum([g.num_nodes for g in datasets['val']])
            test_node_num = sum([g.num_nodes for g in datasets['test']])
            prompt_node_num = int((train_node_num + val_node_num + test_node_num) / total_graph)
            prompt_model = get_prompt_model(num_features=datasets['train'][0].x.size(-1), prompt_node_num=prompt_node_num)
            results = prog(loaders, model, prompt_model, dataset)        
        else:
            raise NotImplementedError(f'Unknown method: {method}')
        
        results.pop('model')
        all_results.append(results)        #    1次finetune其实就是完整的 预训练—微调了，返回的results可以理解为实验结果（ACC）。
    ######                                  #   只不过由于做实验，实验结果有偶然性，所以要多做几次取一个平均值
    # 打印 acc, auroc, f1   的平均值和标准差
    import numpy as np      #   all_results包含了 5次独立实验的  全部结果（ACC）
    for k in all_results[0].keys(): #   把5次独立的实验结果，求一下平均值和标准差
        print(f'{k}: {np.mean([r[k] for r in all_results]):.4f} ± {np.std([r[k] for r in all_results]):.4f}')
        
    import os

    if(method!='prog'):
        with open(os.path.join(save_dir, dataset[0]+'_results.txt'), 'a+') as f:
            f.write('-------------------------------------------------\n')
            for k in all_results[0].keys(): #   把5次实验的平均值和标准差 输出到文件中
                f.write(method+f'FT on All, Target Dataset: {dataset[0]}, {k}: {np.mean([r[k] for r in all_results]):.4f} ± {np.std([r[k] for r in all_results]):.4f}\n')
    else:
        with open(os.path.join(save_dir, dataset[0]+'_results.txt'), 'a+') as f:
            f.write('-------------------------------------------------\n')
            for k in all_results[0].keys():
                f.write(method+f' on All, Target Dataset: {dataset[0]}, {k}: {np.mean([r[k] for r in all_results]):.4f} ± {np.std([r[k] for r in all_results]):.4f}\n')                        
            
    # save
    # torch.save(results, os.path.join(save_dir, dataset[0]+'_results.pt'))

@param('adapt.finetune.backbone_tuning')
@param('adapt.finetune.saliency_tuning')
@param('adapt.finetune.learning_rate')
@param('adapt.finetune.weight_decay')
@param('adapt.epoch')
def finetune(
        loaders,
        model,
        backbone_tuning,
        saliency_tuning,
        learning_rate,
        weight_decay,
        epoch,
        ):

    model.backbone.requires_grad_(backbone_tuning)  #   表示backbone（2层GNN）是会在下游阶段不断更新的
    model.saliency.requires_grad_(saliency_tuning)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),  #   下游阶段要更新的是  GNN  和  2层MLP（answering)
        lr = learning_rate,
        weight_decay = weight_decay,
        )

    from torchmetrics import MeanMetric, Accuracy, F1Score, AUROC
    from tqdm import tqdm
    from copy import deepcopy

    loss_metric = MeanMetric().to(device)
    acc_metric = Accuracy(task='multiclass', num_classes=model.answering.num_class).to(device)
    f1_metric = F1Score(task='multiclass', num_classes=model.answering.num_class, average='macro').to(device)
    auroc_metric = AUROC(task="multiclass", num_classes=model.answering.num_class).to(device)

    best_acc = 0.
    best_model = None

    for e in range(epoch):  
        #   每一个epoch（每一轮），都包括  训练  和  验证。训练就是更新模型，验证就是计算一下模型的预测效果
        #   训练过程  会 更新模型
        #   如果在验证集上的ACC优于best_acc，那么就用这一轮更新后的model来作为最优模型



        model.train()

        loss_metric.reset()
        acc_metric.reset()
        f1_metric.reset()
        auroc_metric.reset()

        #       这一轮的训练

        #   首先是 train_set，一共8个图（加起来220个节点），因为一个batch包含100个图，所以训练集只有一个batch
        pbar = tqdm(loaders['train'], total=len(loaders['train']), ncols=100, desc=f'Epoch {e} Training, Loss: inf')

        for batch in pbar:
            optimizer.zero_grad()   #   这个循环只会执行一次，因为训练集一共就一个batch（包含8个图，加起来220个节点）
            batch = batch.to(device)
            pred = model(batch) #   这里的pred是每个图 的预测值（长度为8的向量，表示8个类别的预测值，可以是负数）
            loss = torch.nn.functional.cross_entropy(pred, batch.y) #   每个图的预测值向量  先经过softmax得到预测概率（8个类别的概率的和为1），然后取正确类别的概率的 -ln值，这样正确类别的预测概率越大，其他类别的预测概率越小，LOSS值就越小
            loss.backward() #   LOSS对 参数求梯度
            optimizer.step()    #   更新参数，更新GNN参数（backbone)  和  下游任务头（2层MLP）

            loss_metric.update(loss.detach(), batch.size(0))
            pbar.set_description(f'Epoch {e} Training Loss: {loss_metric.compute():.4f}', refresh=True)
        pbar.close()


        ###     这一轮的验证


        model.eval()
        
        pbar = tqdm(loaders['val'], total=len(loaders['val']), ncols=100, desc=f'Epoch {e} Validation, Acc: 0., F1: 0.')
        with torch.no_grad():   #   有梯度 就代表要更新模型参数。没有梯度就代表不会更新模型参数
            for batch in pbar:
                batch = batch.to(device)
                pred = model(batch).argmax(dim=-1)  #   预测类别

                acc_metric.update(pred, batch.y)    #   预测类别pred  和  真实类别标签y  ，计算出ACC（预测准确率）
                f1_metric.update(pred, batch.y)
                auroc_metric.update(model(batch), batch.y)
                pbar.set_description(f'Epoch {e} Validation Acc: {acc_metric.compute():.4f}, AUROC: {auroc_metric.compute():.4f}, F1: {f1_metric.compute():.4f}', refresh=True)
            pbar.close()

        if acc_metric.compute() > best_acc:
            best_acc = acc_metric.compute()
            best_model = deepcopy(model)
    







    #   上述有100轮，每一轮都会有一个ACC，我们选取ACC最高的那一轮（表示那一轮的模型预测效果最好，就是在验证集上效果最好），用那一轮的模型来作为最优模型
    model = best_model if best_model is not None else model


#####       所有100轮跑完之后，选取验证集上效果最好的一轮，用这一轮的模型来作为最优模型


###         
    # test
    model.eval()

    acc_metric.reset()
    f1_metric.reset()
    auroc_metric.reset()

    pbar = tqdm(loaders['test'], total=len(loaders['test']), ncols=100, desc=f'Testing, Acc: 0., F1: 0.')
    with torch.no_grad():
        for batch in pbar:
            batch = batch.to(device)
            pred = model(batch).argmax(dim=-1)#   预测类别

            acc_metric.update(pred, batch.y)
            f1_metric.update(pred, batch.y)
            auroc_metric.update(model(batch), batch.y)  #   以下输出的是  每一个batch的acc。
            pbar.set_description(f'Testing Acc: {acc_metric.compute():.4f}, AUROC: {auroc_metric.compute():.4f}, F1: {f1_metric.compute():.4f}', refresh=True)
        pbar.close()
    
    return {
        'acc': acc_metric.compute().item(), #   返回模型在测试集上的  ACC
        'auroc': auroc_metric.compute().item(),
        'f1': f1_metric.compute().item(),
        'model': model.state_dict(),
    }











###################################################################





@param('adapt.epoch')
@param('adapt.prog.prompt_lr')
@param('adapt.prog.prompt_weight_decay')
@param('adapt.prog.ans_lr')
@param('adapt.prog.ans_weight_decay')
@param('adapt.prog.backbone_tuning')
@param('adapt.prog.saliency_tuning')
def prog(
        loaders,
        model,
        prompt_model,      
        dataset,
        epoch,
        backbone_tuning,
        saliency_tuning,          
        prompt_lr,
        prompt_weight_decay,
        ans_lr,
        ans_weight_decay,
        ):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.backbone.to(device)
    model.answering.to(device)
    prompt_model.to(device)
    
    model.backbone.requires_grad_(backbone_tuning)
    model.saliency.requires_grad_(saliency_tuning)

    opi_pg = torch.optim.Adam(
        prompt_model.parameters(),
        lr = prompt_lr,
        weight_decay = prompt_weight_decay,
        )
    
    opi_answer = torch.optim.Adam(
        model.answering.parameters(),
        lr = ans_lr,
        weight_decay = ans_weight_decay,
        )    

    from torchmetrics import MeanMetric, Accuracy, F1Score, AUROC
    from tqdm import tqdm
    from copy import deepcopy

    loss_metric = MeanMetric().to(device)
    acc_metric = Accuracy(task='multiclass', num_classes=model.answering.num_class).to(device)
    f1_metric = F1Score(task='multiclass', num_classes=model.answering.num_class, average='macro').to(device)
    auroc_metric = AUROC(task="multiclass", num_classes=model.answering.num_class).to(device)
    
    # load prompting data

    from torch_geometric.loader import DataLoader

    best_acc = 0.
    best_backbone = None
    best_prompt_model = None
    best_answering = None

    for e in range(epoch):

        loss_metric.reset()
        acc_metric.reset()
        f1_metric.reset()
        auroc_metric.reset()
        
        print(("{}/{} frozen gnn | *tune prompt and tune answering function...".format(e, epoch)))
        prompt_model.train()
        model.backbone.eval()
        model.answering.train()

        from tqdm import tqdm

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        running_loss = 0.
        
        ans_pbar = tqdm(loaders['train'], total=len(loaders['train']), ncols=100, desc=f'Epoch {e} / Total Epoch {epoch} Training, Loss: inf')

        for batch_id, train_batch in enumerate(ans_pbar):  # bar2       
            
            train_batch = train_batch.to(device)
            prompted_graph = prompt_model(train_batch)

            graph_emb = model.backbone(prompted_graph)

            # print(graph_emb)
            pred = model.answering(graph_emb)
            # print(pre)
            train_loss = torch.nn.functional.cross_entropy(pred, train_batch.y)

            opi_answer.zero_grad()
            opi_pg.zero_grad()
            train_loss.backward()
            opi_answer.step()
            opi_pg.step()
            running_loss += train_loss.item()

            current_avg_last_loss = running_loss / (batch_id+1)  # loss per batch

            ans_pbar.set_description('Epoch {} / Total Epoch {} | avg loss: {:.8f}'.format(e, epoch, current_avg_last_loss), refresh=True)
        
        ans_pbar.close()        
                
        model.backbone.eval()
        prompt_model.eval()
        model.answering.eval()

        pbar = tqdm(loaders['val'], total=len(loaders['val']), ncols=100, desc=f'Epoch {e} Validation, Acc: 0., F1: 0.')
        with torch.no_grad():
            for batch in pbar:              
                batch = batch.to(device)
                prompted_graph = prompt_model(batch)
                z = model.backbone(prompted_graph)
                pred = model.answering(z).argmax(dim=-1)

                acc_metric.update(pred, batch.y)
                f1_metric.update(pred, batch.y)
                auroc_metric.update(model(prompted_graph), batch.y)
                pbar.set_description(f'Epoch {e} Validation Acc: {acc_metric.compute():.4f}, AUROC: {auroc_metric.compute():.4f}, F1: {f1_metric.compute():.4f}', refresh=True)
            pbar.close()

        if acc_metric.compute() > best_acc:
            best_acc = acc_metric.compute()
            best_backbone = deepcopy(model.backbone)
            best_answering = deepcopy(model.answering)
            best_prompt_model = deepcopy(prompt_model)
    
    model.backbone = best_backbone if best_backbone is not None else model.backbone
    model.answering = best_answering if best_answering is not None else model.answering
    prompt_model = best_prompt_model if best_prompt_model is not None else prompt_model

    # test
    model.backbone.eval()
    model.answering.eval()
    prompt_model.eval()

    acc_metric.reset()
    f1_metric.reset()
    auroc_metric.reset()

    pbar = tqdm(loaders['test'], total=len(loaders['test']), ncols=100, desc=f'Testing, Acc: 0., F1: 0.')
    with torch.no_grad():
        for batch in pbar:
            batch = batch.to(device)
            prompted_graph = prompt_model(batch)
            z = model.backbone(prompted_graph)
            pred = model.answering(z).argmax(dim=-1)

            acc_metric.update(pred, batch.y)
            f1_metric.update(pred, batch.y)
            auroc_metric.update(model(prompted_graph), batch.y)
            pbar.set_description(f'Testing Acc: {acc_metric.compute():.4f}, AUROC: {auroc_metric.compute():.4f}, F1: {f1_metric.compute():.4f}', refresh=True)
        pbar.close()

    return {
        'acc': acc_metric.compute().item(),
        'auroc': auroc_metric.compute().item(),
        'f1': f1_metric.compute().item(),
        'model': model.state_dict(),
    }