import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.distributed as dist
from collections import Counter
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
        
        

# class STEFunction(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input):
#         ctx.save_for_backward(input)
#         return (input > torch.mean(input).item()).float()
#         #return input.float()

#     # @staticmethod #* ReLU 미분 구현
#     # def backward(ctx, grad_output):
#     #     return grad_output
#     # def backward(ctx, grad_output):
#     #     relu_grad = F.relu(ctx.saved_tensors[0])
#     #     return relu_grad
    
#     @staticmethod #* Sigmoid 미분
#     def backward(ctx, grad_output):
#         input, = ctx.saved_tensors
#         #from util import ForkedPdb;ForkedPdb().set_trace()
#         sigmoid_grad = input * (1. - input)
#         return grad_output * sigmoid_grad
    
    # # @staticmethod
    # def backward(ctx, grad_output):
    #     input, = ctx.saved_tensors
    #     sigmoid_output = torch.sigmoid(input)
    #     sigmoid_grad = sigmoid_output * (1 - sigmoid_output)
    
    #     pos_grad = torch.clamp(grad_output, min=0)
        
    #     #grad_input = pos_grad * sigmoid_grad
        
    #     # # 업데이트된 값이 0 이하로 내려가지 않도록 클리[핑
    #     # grad_input = torch.clamp(grad_input, min=0)
        
    #     return pos_grad 
    
class MaskingFunction(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, weight, input, mask, soft):
        ctx.save_for_backward(weight, input, mask)
        if soft:
            return F.linear(input * mask, weight) 
        else:
            new_mask = (mask >= 0.5).float()
            return F.linear(input * new_mask, weight)


    @staticmethod
    def backward(ctx, grad_output):
        weight, input, mask = ctx.saved_tensors
        # Compute the gradient for the weight
        weight_grad = grad_output.T.matmul(input * mask)
        grad_input = grad_output.matmul(weight)# * mask # 마스크의 영향을 제거한 그레이디언트
        sig_grad = mask * (1. - mask)
        grad_mask = (grad_input * input * sig_grad).sum(dim=0)  

        return weight_grad, grad_input, grad_mask, None


class MaskingModel(nn.Module):
    def __init__(self, input_dim, output_dim, soft = False):
        super(MaskingModel, self).__init__()
        self.soft = soft
        #self.mask_scores = nn.Parameter(torch.ones(input_dim) * 0.001)
        self.register_buffer('mask_scores', torch.ones(input_dim) * 0.01)
        self.classifier = nn.Linear(input_dim, output_dim, bias=False)
        self.register_buffer('gradient_accumulator', torch.zeros_like(self.mask_scores, dtype=torch.float32))
        self.register_buffer('weight_grad', torch.zeros_like(self.mask_scores, dtype=torch.float32))
        
    def forward(self, x):
        mask = F.sigmoid(self.mask_scores)
        out = MaskingFunction.apply(self.classifier.weight, x, mask, self.soft)
        return out
    
    def accumulate_gradient(self):
        # Gather gradients from all processes in DDP
        self.weight_grad = self.classifier.weight.grad.std(dim=0)
        
        if self.weight_grad is None:
            import ipdb;ipdb.set_trace()
        
        # Calculate weighted gradient norm based on class counts
        self.gradient_accumulator += self.weight_grad
    
    def update_mask_scores(self, curr_lr, total_iter):
        # Average the accumulated gradient norm over the epochs
        avg_grad_norm = self.gradient_accumulator / total_iter
        
        with torch.no_grad():
            self.mask_scores -= (curr_lr) * avg_grad_norm
        
        # Reset the gradient accumulator
        self.gradient_accumulator = torch.zeros_like(self.mask_scores)
        self.weight_grad = torch.zeros_like(self.mask_scores)
    

if __name__ == "__main__":
    pass