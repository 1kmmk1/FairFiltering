import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.distributed as dist
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
    def forward(ctx, weight, input, mask, percentile, soft):
        ctx.save_for_backward(weight, input, mask)
        if soft:
            return F.linear(input * mask, weight) 
        else:
            new_mask = (mask < torch.quantile(mask, percentile).item()).float()
            return F.linear(input * new_mask, weight)  # Forward에서는 마스크를 적용

    @staticmethod
    def backward(ctx, grad_output):
        weight, input, mask = ctx.saved_tensors
        # Compute the gradient for the weight
        weight_grad = grad_output.T.matmul(input * mask)
        grad_input = grad_output.matmul(weight)  # 마스크의 영향을 제거한 그레이디언트
        grad_norm = torch.norm(grad_input, p=2, dim=0)
        #grad_norm = torch.norm(weight_grad, p=2, dim=0)
        # sig_grad = mask * (1. - mask)
        #org_z_grad = input * sig_grad; grad_batch = grad_input * org_z_grad
        #grad_norm = torch.norm(grad_batch, p=2, dim=-1) / torch.norm(grad_batch, p=2, dim=-1).sum(); 
        #grad_mask = torch.sum((grad_norm).unsqueeze(-1).contiguous() * grad_batch, dim=0)
        #import ipdb;ipdb.set_trace()
        #grad_mask = torch.sum((mask * input).T @ (grad_output @ weight), dim=0)
        #grad_mask = F.softmax(grad_norm, dim=-1) #* 이렇게만 하니까 크기, 방향 둘 다 고려하기 어려움
        grad_mask = F.softmax(grad_norm, dim=-1) * torch.sum(grad_input, dim=0) #<< 추가 실험 할거
        #grad_mask = 
        return weight_grad, grad_input, grad_mask, None, None


class MaskingModel(nn.Module):
    def __init__(self, input_dim, output_dim, soft = False, percentile = 0.5):
        super(MaskingModel, self).__init__()
        self.soft = soft
        self.mask_scores = nn.Parameter(torch.ones(input_dim))
        self.percentile = percentile
        self.classifier = nn.Linear(input_dim, output_dim, bias=False)
        self.register_buffer('gradient_accumulator', torch.zeros_like(self.mask_scores))
    
    def forward(self, x):
        mask = self.mask_scores
        out = MaskingFunction.apply(self.classifier.weight, x, mask, self.percentile, self.soft)
        return out
    
    def accumulate_gradient(self):
        # Gather gradients from all processes in DDP
        if dist.is_initialized():
            if self.classifier.weight.grad is not None:
                dist.all_reduce(self.classifier.weight.grad, op=dist.ReduceOp.SUM)
        
        # Accumulate the gradient norm
        grad_norm = self.classifier.weight.grad.norm(p=2, dim=0)
        self.gradient_accumulator += grad_norm
    
    def update_mask_scores(self, total_iter):
        # Average the accumulated gradient norm over the epochs
        avg_grad_norm = self.gradient_accumulator / total_iter
        
        # Update mask_scores for the bottom 80% only
        with torch.no_grad():
            threshold = torch.quantile(avg_grad_norm, 0.8)
            mask = avg_grad_norm <= threshold
            self.mask_scores[mask] -= avg_grad_norm[mask]
        
        # Reset the gradient accumulator
        self.gradient_accumulator.zero_()
    

if __name__ == "__main__":
    pass