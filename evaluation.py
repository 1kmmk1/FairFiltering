import torch
import torch.nn as nn
from tqdm import tqdm
from util import MultiDimAverageMeter, cal_acc

def evaluate(device, model, data_loader,
             target_attr_idx,
             attr_dims,
             ):
    
    pbar = tqdm(data_loader, ascii=" =")
    test_acc = MultiDimAverageMeter(attr_dims)
    total = correct_sum = 0
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for data in pbar:
            img, attr, idx = data
            img = img.to(device); attr = attr.to(device)
            
            _, logit = model(img)
            
            preds = torch.argmax(logit, dim=-1)
            # preds = torch.sigmoid(logit)
            # preds = (preds >= 0.5).float().squeeze()        

            correct = (preds == attr[:, target_attr_idx])
            #correct = cal_acc(logit.cpu(), attr.cpu(), target_attr_idx)
            correct_sum += torch.sum(correct).item()
            total += img.size(0)
            
            test_acc.add(correct.cpu(), attr.cpu())
            acc = correct_sum / total
            pbar.set_postfix(acc = "{:.4f}, wga = {:.4f}".format(acc, torch.min(test_acc.get_mean()).item()))
        pbar.close()
    # from util import ForkedPdb;ForkedPdb().set_trace()
    return acc, test_acc.get_mean()