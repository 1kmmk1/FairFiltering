import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MLP(nn.Module):
    def __init__(self, num_classes = 10):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(3 * 28*28, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 100)

        self.relu = nn.ReLU()

        self.classifier = nn.Linear(100, num_classes)

    def forward(self, x, return_feat=False):
        x = x.view(x.size(0), -1)
        x1 = self.relu(self.fc1(x))
        x2 = self.relu(self.fc2(x1))
        x3 = self.relu(self.fc3(x2))
        logit = self.classifier(x3) 
        if return_feat:
            return [x1, x2, x3]
        else:
            return logit
        
        

class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mask):
        sig_input = F.sigmoid(mask)
        ctx.save_for_backward(mask)
        return (sig_input >= 0.5).float()
        #return input.float()

    # @staticmethod #* ReLU 미분 구현
    # def backward(ctx, grad_output):
    #     return grad_output
    def backward(ctx, grad_output):
        saved_mask = ctx.saved_tensors[0]
        sig_grad = F.sigmoid(saved_mask) * (1. - saved_mask)
        #relu_grad = F.leaky_relu(saved_mask, negative_slope=0.1)
        return sig_grad


class MaskingFunction(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, weight, input, mask):
        ctx.save_for_backward(weight, input, mask)
        new_mask = (mask >= 0.5).float()
        return F.linear(input * new_mask, weight)

    @staticmethod
    def backward(ctx, grad_output):
        weight, input, mask = ctx.saved_tensors
        # Compute the gradient for the weight
        weight_grad = grad_output.T.matmul(input)
        grad_input = grad_output.matmul(weight)# * mask # 마스크의 영향을 제거한 그레이디언트
        sig_grad = mask * (1. - mask)
        grad_mask = F.leaky_relu((grad_input * input * sig_grad).sum(dim=0), negative_slope=0.2)
        
        return weight_grad, grad_input, grad_mask
        
class MaskingModel(nn.Module):
    def __init__(self, input_dim, output_dim, drop_ratio = 0.2):
        super(MaskingModel, self).__init__()
        self.input_dim = input_dim
        self.drop_ratio = 0.2
        self.mask_scores = nn.Parameter(torch.randn(input_dim)*0.001)
        self.register_buffer('mask_scores', torch.randn(input_dim)*0.001)
        self.classifier = nn.Linear(input_dim, output_dim, bias=False)
        #torch.nn.init.sparse_(self.classifier.weight, sparsity=0.02, std=0.01)
        
    def forward(self, x, eval):
        if eval:
            out = self.classifier(F.dropout(x, 0.2))
        else:
            new_mask = (F.sigmoid(self.mask_scores) >= 0.5).float()
            out = self.classifier(x * new_mask)
        return out

if __name__ == "__main__":
    pass