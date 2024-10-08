import os
import numpy as np
import torch
import torch.nn.functional as F
import tqdm, os, sys, pdb


#TODO: MultiDim AverageMeter
class MultiDimAverageMeter(object):
    def __init__(self, dims=None):
        if dims != None:
            self.dims = dims
            self.cum = torch.zeros(np.prod(dims))
            self.cnt = torch.zeros(np.prod(dims))
            self.idx_helper = torch.arange(np.prod(dims), dtype=torch.long).reshape(
                *dims
            )
        else:
            self.dims = None
            self.cum = torch.tensor(0.0)
            self.cnt = torch.tensor(0.0)            

    def add(self, vals, idxs=None):
        if self.dims:
            flattened_idx = torch.stack(
                [self.idx_helper[tuple(idxs[i])] for i in range(idxs.size(0))],
                dim=0,
            )
            self.cum.index_add_(0, flattened_idx, vals.view(-1).float())
            self.cnt.index_add_(0, flattened_idx, torch.ones_like(vals.view(-1), dtype=torch.float))
        else:
            self.cum += vals.sum().float()
            self.cnt += vals.numel()
        
    def get_mean(self):
        if self.dims:
            return (self.cum / self.cnt).reshape(*self.dims)
        else:
            return self.cum / self.cnt

    def reset(self):
        if self.dims:
            self.cum.zero_()
            self.cnt.zero_()
        else:
            self.cum = torch.tensor(0.0)
            self.cnt = torch.tensor(0.0)

#TODO: AverageMeter
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

#TODO: EMA
class EMA:
    def __init__(self, label, alpha=0.9):
        self.label = label
        self.alpha = alpha
        self.parameter = torch.zeros(label.size(0))
        self.updated = torch.zeros(label.size(0))
        
    def update(self, data, index):
        self.parameter[index] = self.alpha * self.parameter[index] + (1-self.alpha*self.updated[index]) * data
        self.updated[index] = 1
        
    def max_loss(self, label):
        label_index = np.where(self.label == label)[0]
        return self.parameter[label_index].max()

#TODO: Count correct samples
def cal_acc(output, attr, target_attr_idx):
    preds = output.data.max(1, keepdim=True)[1].squeeze(1)    
    if len(attr.shape) != 1:
        target = attr[:, target_attr_idx]
    else:
        target = attr    
    correct = (preds == target).long()
    return correct

#TODO: Multiclass-brier score
def multi_brier(y_true, y_proba, num_classes, return_sample=True):
    y_true = F.one_hot(y_true, num_classes=num_classes).numpy()
    y_proba = F.softmax(y_proba, dim=-1).numpy()
    sample_brier = np.sum((y_proba- y_true)**2, axis=1)
    if return_sample:
        return sample_brier.astype(np.float32)
    else:
        return np.mean(sample_brier).astype(np.float32)
    
#TODO: Save embeddings
def save_embed(rank, model, data_loader, log_name, data='img'):
    
    assert rank ==0
    
    if data in ['Waterbirds', 'CelebA']:
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
    else:
        def get_embed(m, x) -> list: #*MLP
            out = m.model(input_ids=x[:, :, 0],
            attention_mask=x[:, :, 1],
            token_type_ids=x[:, :, 2])
            return out.last_hidden_state[:, 0, :]
    all_embeddings = {}
    all_y, all_b = {}, {}
    model.eval()
    
    for name, loader in data_loader:   
        print(len(loader.dataset)) 
        all_embeddings[name] = []
        all_y[name], all_b[name] = [], []
        pbar = tqdm.tqdm(loader, file= sys.stdout, desc="Saving Emeddings")
        for batch_data in pbar:
            img, attr, idx = batch_data
            y = attr[:, 0]; b = attr[:, 1]
            with torch.no_grad():
                embed = get_embed(model, img.to(rank))
                all_embeddings[name].append(embed.cpu().numpy())
                all_y[name].append(y.detach().cpu().numpy())
                all_b[name].append(b.detach().cpu().numpy())
        #print(len(all_embeddings['train']))
        all_embeddings[name] = np.vstack(all_embeddings[name])
        all_y[name] = np.concatenate(all_y[name])
        all_b[name] = np.concatenate(all_b[name])
    np.savez(os.path.join('log', log_name,
        f"embeddings.npz"),
        embeddings = all_embeddings,
        labels = all_y,
        sense_labels= all_b
    )


def load_scheduler(optimizer, args):
    if args.main_scheduler_tag == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
        
    elif args.main_scheduler_tag == "exponential":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)
        
    elif args.main_scheduler_tag == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.0001)
    
    elif args.main_scheduler_tag == "multistep":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)        
    
    elif args.main_scheduler_tag == "linear":
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.05, total_iters=50)        
    elif args.main_scheduler_tag == "cosine_wp":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0 = 10, T_multi=2)
    else:
        raise ValueError
    return scheduler
    
def load_optimizer(params, args):
    if args.main_optimizer_tag == "Adam":
        optimizer = torch.optim.Adam(params, lr=args.learning_rate, weight_decay = args.weight_decay)
        
    elif args.main_optimizer_tag == "SGD":
        optimizer = torch.optim.SGD(params, lr=args.learning_rate, weight_decay = args.weight_decay, momentum=args.momentum)
        
    elif args.main_optimizer_tag == "AdamW":
        optimizer = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay = args.weight_decay)

    else:
        raise ValueError(f"Unknown optimizer tag: {args.main_optimizer_tag}")
    
    return optimizer

class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin



if __name__ == "__main__":
    pass