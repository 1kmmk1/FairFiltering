import os, sys, glob
import tqdm
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

from module.util import get_model
from dataset.custom_datasets import My_dataset
from util import MultiDimAverageMeter, cal_acc

model_list = os.listdir('log/baseline')

result_dict = {"Waterbirds_ACC": [],
               "Waterbirds_WGA": [],
               "CelebA_ACC": [],
               "CelebA_WGA": [],
               "CivilComments_ACC": [],
               "CivilComments_WGA": [],
               "MultiNLI_ACC": [],
               "MultiNLI_WGA": []} 


for MODEL_PATH in tqdm.tqdm(model_list, leave=True):
    with open(os.path.join('log/baseline', MODEL_PATH, 'result.txt'), 'r') as f:
        data = f.readlines()

    baseline_wga = round(float(data[1][5:12]) * 100, 2)
    data_name = MODEL_PATH.split("_")[0]
    train_ds = My_dataset(data = data_name, split = 'train')
    
    group = 2 * train_ds.attr[:, 0] + train_ds.attr[:, 1]
    group_ratio = np.unique(group, return_counts=True)[1] / len(group)
    
    test_result = np.load(os.path.join('log/baseline', MODEL_PATH, 'test_result.npy'))
    baseline_acc = (torch.from_numpy(test_result.reshape(1, -1).squeeze()) * torch.from_numpy(group_ratio)).sum()
    if result_dict[f"{data_name}_ACC"] is None:
        result_dict[f"{data_name}_ACC"] = [baseline_acc.item()]
    else:
        result_dict[f"{data_name}_ACC"].append(baseline_acc.item())
        
    if result_dict[f"{data_name}_WGA"] is None:
        result_dict[f"{data_name}_WGA"]= [baseline_wga]
    else:
        result_dict[f"{data_name}_WGA"].append(baseline_wga)





for key in result_dict.keys():
    print(key)
    print(np.std(np.array(result_dict[key])))
    print(np.mean(np.array(result_dict[key])))
    import ipdb;ipdb.set_trace()
# model_list = os.listdir("log/baseline")
# ratio = [round(num, 2) for num in  (torch.linspace(0., 1.0, 20) * 100).tolist()][1:-1]
# Waterbirds = {"Ratio": ratio}
# CelebA = {"Ratio": ratio}
# Civil = {"Ratio": ratio}
# Multi = {"Ratio": ratio}

# baseline_Waterbirds = {}
# baseline_CelebA = {}
# baseline_Civil = {}
# baseline_Multi = {}

# for model_path in model_list:
#     do_file = glob.glob(os.path.join('log', model_path, "*.txt"))
#     if len(do_file) > 0:
#         data_name = model_path.split("_")[0]
#         data = pd.read_csv(do_file[0])
#         with open(os.path.join('log', model_path, "result.txt"), 'r') as f:
#             baseline = f.readlines()
#             #import ipdb;ipdb.set_trace()
#             baseline_acc = round(float(baseline[0][5:12]) * 100, 2)
#             baseline_wga = round(float(baseline[1][5:12]) * 100, 2)
            
#         if data_name == "Waterbirds":
#             Waterbirds[f"{model_path}_ACC"] = data['ACC'][1:-1]
#             Waterbirds[f"{model_path}_WGA"] = data['WGA'][1:-1]
#             baseline_Waterbirds[f"{model_path}_ACC"] = baseline_acc
#             baseline_Waterbirds[f"{model_path}_WGA"] = baseline_wga
            
#         elif data_name == "CelebA":
#             CelebA[f"{model_path}_ACC"] = data['ACC'][1:-1]
#             CelebA[f"{model_path}_WGA"] = data['WGA'][1:-1]
#             baseline_CelebA[f"{model_path}_ACC"] = baseline_acc
#             baseline_CelebA[f"{model_path}_WGA"] = baseline_wga
                        
#         elif data_name == "CivilComments":
#             Civil[f"{model_path}_ACC"] = data['ACC'][1:-1]
#             Civil[f"{model_path}_WGA"] = data['WGA'][1:-1]
#             baseline_Civil[f"{model_path}_ACC"] = baseline_acc
#             baseline_Civil[f"{model_path}_WGA"] = baseline_wga
#         else: 
#             Multi[f"{model_path}_ACC"] = data['ACC'][1:-1]
#             Multi[f"{model_path}_WGA"] = data['WGA'][1:-1]
#             baseline_Multi[f"{model_path}_ACC"] = baseline_acc
#             baseline_Multi[f"{model_path}_WGA"] = baseline_wga