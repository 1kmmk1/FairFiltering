import os, sys, random, math
import numpy as np
import torch
import torch.nn as nn 
import torch.nn.functional as F
from tqdm import tqdm
from module.mlp import MaskingModel
from module.loss import GeneralizedCELoss, Custom_CrossEntropy, FocalLoss, CDB_loss
from module.util import get_model, ForkedPdb
from module.earlystop import EarlyStopping

from util import load_optimizer, load_scheduler
from util import cal_acc, AverageMeter, MultiDimAverageMeter, save_embed
import wandb
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

    
def get_embed(m, x):
    x = m.conv1(x)
    x = m.bn1(x)
    x = m.relu(x)
    x = m.maxpool(x)

    x1 = m.layer1(x)
    x2 = m.layer2(x1)
    x3 = m.layer3(x2)
    x4 = m.layer4(x3)

    x4 = m.avgpool(x4)
    out = torch.flatten(x4, 1)
    return out

def count_class(attr_dims, my_dl):
    count_tensor = torch.zeros(attr_dims[0])
    for data in my_dl:
        _, attr, _ = data
        count = torch.bincount(attr[:, 0])
        count_tensor += count
    return count_tensor

def train_ERM(train_dl,
            valid_dl,
            attr_dims,
            args):
    
    #* model
    model = get_model(model_tag=args.model, num_classes=attr_dims[0], train_clf = args.train_clf) 
    # if pruning:
        # d = model.fc.in_features
        # model.fc = SupermaskLinear(d, attr_dims[0], bias = False)
    device = f"cuda:{args.device}"
    model = model.to(device)
    
    score_params = [model.fc.mask_scores]
    other_params = [p for p in model.parameters() if p is not model.fc.mask_scores]
    
    # optimizer = load_optimizer(other_params, args)
    # optimizer_2 = load_optimizer(score_params, args)
    optimizer = load_optimizer(model.parameters(), args)
    
    kld = nn.KLDivLoss(reduction='batchmean', log_target=True)

    if args.weighted: 
        attrs = train_dl.dataset.attr[:, 0].unique(return_counts=True)[1]
        weights = torch.tensor([1. - (x.item()/sum(attrs).item()) for x in attrs]).to(device)
        criterion = nn.CrossEntropyLoss(weight = weights, reduction = 'none')
    else:
        criterion = nn.CrossEntropyLoss(reduction = 'none')
    
    if args.main_scheduler_tag != 'None':
        scheduler = load_scheduler(optimizer, args)
        #scheduler_2 = load_scheduler(optimizer_2, args)

    BEST_SCORE: float = 0
    PATIENCE: int = 0    
    #TODO: Train Feature Extractor 
    tau = 2.0
    for epoch in range(1, args.epochs+1):

        model.train()

        total = 0; correct_sum = 0
        
        pbar = tqdm(train_dl, file= sys.stdout, leave=True)
        
        loss_meter = MultiDimAverageMeter(attr_dims); acc_meter = MultiDimAverageMeter(attr_dims)

        for batch_idx, (data) in enumerate(pbar):
            imgs, attr, idx = data          
            imgs = imgs.to(device); attr = attr.to(device)
            target = attr[:, 0]
            
            output, _ = model(imgs, tau)
            
            # log_out = F.log_softmax(output, dim=-1)
            # log_sub = F.log_softmax(sub_output, dim =-1)
        
            loss = criterion(output, target)
            preds = torch.argmax(output, dim=-1)

            # kld_loss = kld(log_out, log_sub) #* L(y_pred, y_true)
    
            loss_for_update = loss.mean()# + kld_loss * 0.8
                
            correct = (preds == target)
            loss_meter.add(loss.cpu(), attr.cpu())
            acc_meter.add(correct.cpu(), attr.cpu())
            
            correct_sum += torch.sum(correct).item(); total += imgs.size(0)
            if np.isnan(loss_for_update.cpu().detach().item()):
                from util import ForkedPdb;ForkedPdb().set_trace()
            
            optimizer.zero_grad()
            #optimizer_2.zero_grad()
            loss_for_update.backward()
            optimizer.step()
            #optimizer_2.step()
            
            wga = torch.min(acc_meter.get_mean()) #* Worst group Acc
            loss = loss_meter.get_mean()
            acc = acc_meter.get_mean()

            pbar.set_postfix(epoch = f"{epoch}/{args.epochs}", loss = "{:.4f}, acc = {:.4f}".format(loss_for_update.detach().cpu().item(), correct_sum / total))
        
        if batch_idx % args.log_freq == 0:
            wandb.log({"train/loss": loss_for_update.item(),
                    "train/acc": correct_sum / total,
                    "train/WGA": wga.item(),
                    #"train/acc_bias_aligned": acc[eye_tsr==1].mean(),
                    #"train/acc_bias_skewed": acc[eye_tsr==0].mean(),
                    })
        
        print(f"Train ACC: {torch.mean(acc)},  Train WGA: {wga}")
        pbar.close()
        tau -= 0.0015 
        if args.main_scheduler_tag != 'None':
            scheduler.step()
            #scheduler_2.step()
            
        if valid_dl is not None:
            model.eval()

            pbar = tqdm(valid_dl, file= sys.stdout, leave=True)

            group_acc = MultiDimAverageMeter(attr_dims)
            Valid_loss = 0
            total = 0;correct_sum = 0
            with torch.no_grad():
                for batch_idx, (data) in enumerate(pbar):
                    img, attr, idx = data
                    img = img.to(device); attr = attr.to(device)
                    target = attr[:, 0]; 
                    
                    output, _ = model(img, tau)
                
                    val_loss = criterion(output, target)
                    preds = torch.argmax(output, dim=-1)
                    
                    loss_for_update = val_loss.mean()
                    Valid_loss += loss_for_update.item()
                    correct = (preds == target)
                    
                    total += img.size(0); correct_sum += torch.sum(correct).item()
                    eq_acc = correct_sum / total
                    
                    group_acc.add(correct.cpu(), attr.cpu())
            
                pbar.set_postfix(loss = "{:.4f}, acc = {:.4f}".format(loss_for_update.cpu(), eq_acc))

            pbar.close()
            
            val_acc = torch.mean(group_acc.get_mean())
            val_wga = torch.min(group_acc.get_mean())

            print("Mean acc: {:.2f}, Worst Acc: {:.2f}".format(val_acc.item()*100, val_wga.item()*100))
            wandb.log({"valid/loss": loss_for_update.item(),
                        "valid/acc": eq_acc, 
                        "valid/WGA": val_wga.item(),
                        #"valid/acc_bias_aligned": group_acc.get_mean()[eye_tsr==1].mean(),
                        #"valid/acc_bias_skewed": group_acc.get_mean()[eye_tsr==0].mean()
                        })
            if Valid_loss <= BEST_SCORE:
                BEST_SCORE = val_wga.item()
                print("*"*15, "Best Score: {:.4f}".format(val_wga.item()*100), "*"*15) 
                save_epoch = epoch
                state_dict = {'best score': val_wga.item(),
                            'epoch': save_epoch,
                            'state_dict': model.state_dict()}
                with open(os.path.join("log", args.log_name, "model.th"), 'wb') as f:
                    torch.save(state_dict, f)
                PATIENCE = 0
            else:
                PATIENCE += 1
            
            if PATIENCE > args.patience:
                if args.early_stopping:    
                    break
                    # return group_acc, eq_acc, val_wga.item()  
                    
    
    #* save model
    state_dict = {
            'best score': val_wga,
            'state_dict': model.state_dict(), 
        }
    with open(os.path.join("log", args.log_name, f"model_{epoch}.th"), 'wb') as f:
        torch.save(state_dict, f)
            
    return group_acc, eq_acc, val_wga.item()  
