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

import math

def get_inverse_cosine_lr(current_epoch, T_max, eta_max=0.1, eta_min=0.0001):
    """
    반대되는 CosineAnnealing 학습률을 계산하는 함수.
    
    :param current_epoch: 현재 에포크 (0부터 시작)
    :param T_max: 총 에포크 수
    :param eta_max: 최대 학습률 (학습 후반에 다다랐을 때)
    :param eta_min: 최소 학습률 (학습 초기에 설정할 값)
    :return: 현재 에포크에서의 학습률
    """
    return eta_min + (eta_max - eta_min) * (1 + math.cos(math.pi * (current_epoch / T_max))) / 2

        
def train_ERM(rank, 
            train_dl,
            valid_dl,
            train_sampler,
            valid_sampler,
            attr_dims,
            model,
            optimizer,
            scheduler,
            args):
    
    model = model.to(rank)

    if args.weighted: 
        attrs = train_dl.dataset.attr[:, 0].unique(return_counts=True)[1]
        weights = torch.tensor([1. - (x.item()/sum(attrs).item()) for x in attrs]).to(rank)
        criterion = nn.CrossEntropyLoss(weight = weights, reduction = 'none')
    else:
        criterion = nn.CrossEntropyLoss(reduction = 'none')

    BEST_SCORE: float = 0.
    PATIENCE: int = 0    
    UPDATE_FREQ: int = 1
    #TODO: Train Feature Extractor 
    for epoch in range(1, args.epochs+1):

        gradient_list = []
        
        model.train()
        
        total = 0; correct_sum = 0
        train_sampler.set_epoch(epoch) if train_sampler is not None else ''
        if rank == 0 or rank =='cuda:0':
            pbar = tqdm(train_dl, file=sys.stdout)
        else:
            pbar = train_dl
        
        loss_meter = MultiDimAverageMeter(attr_dims); acc_meter = MultiDimAverageMeter(attr_dims)

        for batch_idx, (data) in enumerate(pbar):
            imgs, attr, idx = data          
            imgs = imgs.to(rank); attr = attr.to(rank)
            target = attr[:, 0]
            
            output = model(imgs)
            
            loss = criterion(output, target)
            preds = torch.argmax(output, dim=-1)
    
            loss_for_update = loss.mean()

            correct = (preds == target)
            loss_meter.add(loss.cpu(), attr.cpu())
            acc_meter.add(correct.cpu(), attr.cpu())
            
            correct_sum += torch.sum(correct).item(); total += imgs.size(0)
            if np.isnan(loss_for_update.cpu().detach().item()):
                from util import ForkedPdb;ForkedPdb().set_trace()
            
            optimizer.zero_grad()
            loss_for_update.backward()
            if model.module.fc.classifier.weight.grad is not None:
                grad = model.module.fc.classifier.weight.grad.detach().clone().t()  # Shape: (input_dim, output_dim)
                gradient_list.append(grad)  # List에 추가
            
                # 마스킹된 가중치의 그레이디언트를 0으로 설정하여 업데이트 방지
            with torch.no_grad():
                # 마스크가 0인 곳의 그레이디언트를 0으로
                model.module.fc.classifier.weight.grad *= model.module.fc.mask.t()
            optimizer.step()
            
            wga = torch.min(acc_meter.get_mean()) #* Worst group Acc
            loss = loss_meter.get_mean()
            acc = acc_meter.get_mean()

            if rank == 0:
                pbar.set_postfix(epoch = f"{epoch}/{args.epochs}", loss = "{:.4f}, acc = {:.4f}".format(loss_for_update.detach().cpu().item(), correct_sum / total))
                if batch_idx % 40 == 0:
                    wandb.log({"train/loss": loss_for_update.item(),
                            "train/acc": correct_sum / total,
                            "train/WGA": wga.item(),
                            })
            
        if rank==0:
            print(f"Train ACC: {torch.mean(acc)},  Train WGA: {wga}")
            pbar.close()
            
        if scheduler is not None:
            scheduler.step()
            
        all_grads = torch.stack(gradient_list)
        grad_std = all_grads.std(dim=0)
        _, max_idx = grad_std.view(-1).max(0)
        max_i = max_idx // attr_dims[0]
        max_j = max_idx % attr_dims[0]
        max_i = max_i.item()
        max_j = max_j.item()
        
                # 해당 가중치 마스킹
        model.module.fc.mask_weight(max_i, max_j)
        
        if valid_dl is not None:
            model.eval()

            valid_sampler.set_epoch(epoch) if valid_sampler is not None else ''
        
            if rank == 0 or rank =='cuda:0':
                pbar = tqdm(valid_dl, file=sys.stdout)
            else:
                pbar = valid_dl
            
            group_acc = MultiDimAverageMeter(attr_dims)
            sum_loss = 0;
            total = 0;correct_sum = 0
            with torch.no_grad():
                for batch_idx, (data) in enumerate(pbar):
                    img, attr, idx = data
                    img = img.to(rank); attr = attr.to(rank)
                    target = attr[:, 0]; 
                    
                    output = model(img)
                
                    val_loss = criterion(output, target)
                    preds = torch.argmax(output, dim=-1)
                    
                    loss_for_update = val_loss.mean()
                    sum_loss += loss_for_update.detach().cpu()
                    correct = (preds == target)
                    
                    total += img.size(0); correct_sum += torch.sum(correct).item()
                    eq_acc = correct_sum / total
                    
                    group_acc.add(correct.cpu(), attr.cpu())
                if rank == 0:
                    pbar.set_postfix(loss = "{:.4f}, acc = {:.4f}".format(loss_for_update.cpu(), eq_acc))
            if rank == 0:
                pbar.close()
            
            val_acc = torch.mean(group_acc.get_mean())
            val_wga = torch.min(group_acc.get_mean())
        
            
            if rank ==0:
                print("Mean acc: {:.2f}, Worst Acc: {:.2f}".format(val_acc.item()*100, val_wga.item()*100))
                wandb.log({"valid/loss": loss_for_update.item(),
                            "valid/acc": eq_acc, 
                            "valid/WGA": val_wga.item(),
                            })
                
            if val_wga.item() > BEST_SCORE:
                BEST_SCORE = val_wga.item()
                BEST_ACC = eq_acc
                BEST_EPOCH = epoch
                BEST_GROUP_ACC = group_acc
                if rank == 0:
                    
                    print("*"*15, "Best Score: {:.4f}".format(val_wga.item()*100), "*"*15) 
                    save_epoch = epoch
                    state_dict = {'best score': val_wga.item(),
                                'epoch': save_epoch,
                                'state_dict': model.module.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict()}
                    
                    save_path = os.path.join("log", args.log_name, f"model_{epoch}.th")
                    torch.save(state_dict, save_path)
                    print("*"*15, "Model Saved!", "*"*15)
                PATIENCE = 0
            else:
                PATIENCE += 1
            
            if PATIENCE > args.patience:
                if args.early_stopping:    
                    break

    return BEST_GROUP_ACC, BEST_ACC, BEST_SCORE, BEST_EPOCH
