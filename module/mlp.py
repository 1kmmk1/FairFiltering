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
            F.linear(input * mask, weight)
        else:
            new_mask = (mask < torch.quantile(mask, percentile).item()).float()
            return F.linear(input * new_mask, weight)  # Forward에서는 마스크를 적용

    @staticmethod
    def backward(ctx, grad_output):
        weight, input, mask = ctx.saved_tensors
        weight_grad = grad_output.T.matmul(input)# * mask 
        grad_input = grad_output.matmul(weight)  # 마스크의 영향을 제거한 그레이디언트
        #grad_norm = torch.norm(grad_input, p=2, dim=-1) / torch.norm(grad_input, p=2, dim=-1).sum()
        # max_norm = torch.argmax(grad_norm)
        # sig_grad = mask * (1. - mask)
        #org_z_grad = input * sig_grad; grad_batch = grad_input * org_z_grad
        #grad_norm = torch.norm(grad_batch, p=2, dim=-1) / torch.norm(grad_batch, p=2, dim=-1).sum(); 
        #grad_mask = torch.sum((grad_norm).unsqueeze(-1).contiguous() * grad_batch, dim=0)
        grad_mask = torch.sum((mask * input).T @ (grad_output @ weight), dim=0)
        #grad_mask = grad_input[max_norm]# * sig_grad
        return weight_grad, grad_input, grad_mask, None


class MaskingModel(nn.Module):
    def __init__(self, input_dim, output_dim, soft = False, percentile = 0.5):
        super(MaskingModel, self).__init__()
        self.soft = soft

        self.mask_scores = nn.Parameter(torch.randn(input_dim))

        self.percentile = percentile
        self.classifier = nn.Linear(input_dim, output_dim, bias=False)
    def forward(self, x):
        mask = F.sigmoid(self.mask_scores)
            
        if self.soft:
            out = self.classifier(x * mask)
        else:
            out = MaskingFunction.apply(self.classifier.weight, x, mask, self.percentile)

        return out
    

if __name__ == "__main__":
    pass