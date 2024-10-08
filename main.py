import os, sys, random
import numpy as np
import torch
import torch.nn as nn 
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset, random_split, ConcatDataset
from tqdm import tqdm
import argparse
import json 

from dataset.custom_datasets import My_dataset
from module.util import get_model
from module.mlp import MaskingModel
from evaluation import evaluate
from util import save_embed

from train_ERM import train_ERM
import wandb

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, WeightedRandomSampler

def setup(seed):
    # random.seed(SEED) #  Python의 random 라이브러리가 제공하는 랜덤 연산이 항상 동일한 결과를 출력하게끔
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def main(args):
    setup(args.seed)
    
    #* Initialize wandb
    wandb.init(project=args.project, 
            config = {"seed": args.seed,
                        "optimizer": args.main_optimizer_tag,
                        "scheduler": args.main_scheduler_tag,
                        "model": args.model,
                        "weight_decay": args.weight_decay,
                        "early_stop": args.early_stopping,
                        "epochs": args.epochs,
                        "batch size": args.batch_size,
                        "momentum": args.momentum,
                        "learning rate": args.learning_rate,
                        "shuffle": args.shuffle,
                        })
    print(args.log_name)
    wandb.run.name = args.log_name
    
    train_ds = My_dataset(data = args.data_type, split = 'train', shuffle=args.shuffle, ratio = args.ratio)
    valid_ds = My_dataset(data = args.data_type, split = 'val', shuffle=args.shuffle, ratio = args.ratio)
    test_ds = My_dataset(data = args.data_type, split = 'test')
    print("Train DS: ", len(train_ds), "\n", "Valid Ds: ", len(valid_ds))
    
    num_classes = torch.max(train_ds.attr[:, 0]).item() + 1
    bias_classes = torch.max(train_ds.attr[:, 1]).item() + 1
    attr_dims = [num_classes, bias_classes] 
    # #import ipdb;ipdb.set_trace()
    # class_counts = torch.bincount(train_ds.attr[:, 0].to(torch.int64))  # 클래스별 샘플 수 계산
    # class_weights = 1. / class_counts.float()  # 각 클래스에 반비례하는 가중치 계산

    # # 각 샘플에 가중치 부여
    # sample_weights = class_weights[train_ds.attr[:, 0].to(torch.int64)]
    # #TODO: Train & Valid
    # sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_dl = DataLoader(train_ds, batch_size = args.batch_size, shuffle=True, pin_memory=True)
    valid_dl =  DataLoader(valid_ds, batch_size = args.batch_size, shuffle=False, pin_memory=True)
    test_dl =  DataLoader(test_ds, batch_size = args.batch_size, shuffle=False, pin_memory=False)

    print("*"*15, "Training Start!", "*"*15)
    for key, value in vars(args).items():
        print(f"{key}: {value}")            

    device = f"cuda:{args.device}"
    
    group_acc, val_acc, val_wga = train_ERM(train_dl, valid_dl, attr_dims, args)
    print("Valid Acc: ", val_acc)
    print("Valid WGA: ", val_wga)

    np.save(f"log/{args.log_name}/validation_result", np.array(group_acc.get_mean()))

    #TODO: Test
    print("*"*15, "Test Start!", "*"*15)   
    
    main_model = get_model(model_tag=args.model, num_classes=attr_dims[0], train_clf=args.train_clf) 

    state_dict = torch.load(os.path.join("log", f"{args.log_name}", "model.th"))
    main_model.load_state_dict(state_dict['state_dict'], strict=True)
    main_model = main_model.to(device)
    
    #TODO: Save Embeddings
    if args.save_embed:
        save_embed(rank = device, model=main_model, data_loader = [("valid", valid_dl), ("test", test_dl)], log_name = args.log_name, data = args.data_type)
    
    #TODO: Load Classifier
    acc, test_accs = evaluate(device,
                            main_model,
                            test_dl,
                            target_attr_idx = 0,
                            attr_dims = attr_dims)

    group = 2 * train_ds.attr[:, 0] + train_ds.attr[:, 1]
    group_ratio = np.unique(group, return_counts=True)[1] / len(group)

    acc = (test_accs.reshape(1, -1).squeeze() * torch.from_numpy(group_ratio)).sum()
    
    eye_tsr = torch.eye(attr_dims[0]).long()
    filename = os.path.join('log', args.log_name, 'result.txt')
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(f"Acc: {acc.item()}\n")
        file.write(f"WGA: {torch.min(test_accs).item()}\n")
        #file.write(f"Acc(Bias Skewed): {test_accs[eye_tsr==0].mean().item()}\n")
        #file.write(f"Acc(Bias Aligned): {test_accs[eye_tsr==1].mean().item()}")
    
    np.save(f"log/{args.log_name}/test_result", np.array(test_accs))
    
    print("Test ACC: ", acc.item())
    print("Test WGA: ", torch.min(test_accs).item())
        

def save_args(args, filename):
    os.makedirs(filename, exist_ok=True)
    with open(os.path.join(filename, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
            

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=42, type=int, help="seed for reproducing")
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--data_type",  default="Waterbirds", type = str)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--learning_rate", default=0.003, type=float)
    parser.add_argument("--weight_decay", default=0.0001, type=float)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--main_scheduler_tag", default='cosine', type=str, help="learning scheduler", choices=['step', 'exponential', 'cosine', 'multistep', 'linear', 'cosine_wp', 'None'])
    parser.add_argument("--main_optimizer_tag", default='SGD', type=str, choices=['Adam', 'AdamW', 'SGD'])
    parser.add_argument("--gamma", default=0.1, type=float, help="learning rate decay ratio for lr scheduler")
    parser.add_argument("--step_size", default=20, type=float)
    parser.add_argument("--milestones", default=[20, 50, 60, 70, 80], type=list, help="learning rate decay milestone for MultiStepLR")
    parser.add_argument("--momentum", default=0.9, type=float)

    parser.add_argument("--model", default='resnet50', type=str)
    parser.add_argument("--early_stopping", action='store_true',  help="Early Stopping")
    parser.add_argument("--save_embed", action='store_true',  help="Save Embeddings")
    parser.add_argument("--shuffle", action='store_true', help = 'Shuffle train and valid randomly')
    parser.add_argument("--ratio", type=float, help = 'ratio of training to validation', default=0.5)
    parser.add_argument("--patience", type=int, help = 'Patience for Early Stopping', default=40)
    parser.add_argument("--weighted", action='store_true', help = 'Using Weighted CrossEntropy During Training')
    parser.add_argument("--train_clf", action='store_true', help = 'Train Model with Masked Classifier')
    parser.add_argument("--loss_weight", type=float, help = 'Hyper-parameter for regularization term', default=1.0)
    parser.add_argument("--percentile", type=float, help = 'Hyper-parameter for regularization term', default=0.5)
    
    parser.add_argument("--project", type=str, help="Project name for wandb logging")
    parser.add_argument("--log_freq", default=10, type=int, help="logging frequency for wandb")
    parser.add_argument("--log_name", type=str, default=None)
    parser.add_argument("--ckpt", type=str, default=None, help = 'checkpoint for bert model')
    parser.add_argument("--sparsity", type=float, default=0.1, help = 'pruning ratio')
    
    parser.add_argument('--rand', action='store_true', help='')
    
    args = parser.parse_args()
    
    if args.log_name is None:
        args.log_name = f"{args.data_type}_{args.loss_weight}"
    
    if args.project is None:
        args.project = args.data_type
    command = ' '.join(sys.argv)
    os.makedirs(os.path.join("log", args.log_name), exist_ok=True)
    with open(os.path.join('log', args.log_name, 'command.txt'), 'w') as f:
        f.write(command)
        
    return args

if __name__ == "__main__":   
    args = parse_args()
    save_args(args=args, filename=os.path.join("log", args.log_name))
    main(args)
