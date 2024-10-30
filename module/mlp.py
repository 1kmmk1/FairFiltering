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
    
#     # @staticmethod #* Sigmoid 미분
#     # def backward(ctx, grad_output):
#     #     input, = ctx.saved_tensors
#     #     #from util import ForkedPdb;ForkedPdb().set_trace()
#     #     sigmoid_grad = input * (1. - input)
#     #     return grad_output * sigmoid_grad
    
#     # @staticmethod
#     def backward(ctx, grad_output):
#         input, = ctx.saved_tensors
#         sigmoid_output = torch.sigmoid(input)
#         sigmoid_grad = sigmoid_output * (1 - sigmoid_output)
    
#         #pos_grad = torch.clamp(grad_output, min=-0.5, max=0.5)
        
#         grad_input = input * sigmoid_grad
        
#         # # 업데이트된 값이 0 이하로 내려가지 않도록 클리[핑
#         # grad_input = torch.clamp(grad_input, min=0)
        
#         return grad_input

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
        weight_grad = grad_output.T.matmul(input)
        grad_input = grad_output.matmul(weight)# * mask # 마스크의 영향을 제거한 그레이디언트
        sig_grad = mask * (1. - mask)
        grad_mask = F.sigmoid(weight_grad.std(dim=0)) * ((grad_output @ weight)*input * sig_grad).sum(dim=0) 
        #grad_mask = weight_grad.std(dim=0)
        
        return weight_grad, grad_input, None, None

class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return (input > torch.mean(input).item()).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        #from util import ForkedPdb;ForkedPdb().set_trace()
        sigmoid_grad = input * (1. - input)
        return grad_output * sigmoid_grad
    
    # @staticmethod
    # def backward(ctx, grad_output):
    #     input, = ctx.saved_tensors
    #     sigmoid_output = torch.sigmoid(input)
    #     sigmoid_grad = sigmoid_output * (1 - sigmoid_output)
        
    #     positive_grad = torch.clamp(grad_output, max=0)
        
    #     grad_input = positive_grad * sigmoid_grad
        
    #     grad_input = torch.clamp(grad_input, min=0)
        
    #     return grad_input
        
class MaskingModel(nn.Module):
    def __init__(self, input_dim, output_dim, soft):
        super(MaskingModel, self).__init__()
        self.mask_scores = nn.Parameter(torch.randn(input_dim))
        self.classifier = nn.Linear(input_dim, output_dim, bias=False)
        
        self.register_buffer('gradient_accumulator', torch.zeros_like(self.mask_scores, dtype=torch.float32))
        self.register_buffer('weight_grad', torch.zeros_like(self.mask_scores, dtype=torch.float32))
        self.register_buffer('mask', torch.ones(input_dim, output_dim))
        
    def forward(self, x):
        mask = STEFunction.apply(F.sigmoid(self.mask_scores))
        out = self.classifier(x*mask)
        return out
    
    def accumulate_gradient(self):
        # Gather gradients from all processes in DDP
        self.weight_grad = self.classifier.weight.grad.std(dim=0)
        
        if self.weight_grad is None:
            import ipdb;ipdb.set_trace()
        
        # Calculate weighted gradient norm based on class counts
        self.gradient_accumulator += self.weight_grad
    
    def mask_weight(self, i, j):
        """
        특정 가중치 요소를 마스킹합니다.
        Args:
            i (int): input dimension 인덱스
            j (int): output dimension 인덱스
        """
        i_tensor = torch.tensor(i, dtype=torch.long, device=self.mask.device)
        j_tensor = torch.tensor(j, dtype=torch.long, device=self.mask.device)
        self.mask[i_tensor, j_tensor] = 0.0
        
    def mask_gradients(self, gradient_list, k=10):
        """
        이전에 마스킹된 가중치를 제외하고, 새로운 가중치를 마스킹합니다.
        Args:
            model (FullyConnectedLayer): 마스킹을 적용할 모델의 FC 레이어
            gradient_list (list of torch.Tensor): 각 iteration마다 수집된 그라디언트 리스트
            k (int): 마스킹할 가중치의 개수
        """
        all_grads = torch.stack(gradient_list)  # Shape: (iterations, dim1, dim2)
        grad_std = all_grads.std(dim=0)        # Shape: (dim1, dim2)

        # 이미 마스킹된 가중치를 제외하기 위해 grad_std를 수정
        grad_std_masked = grad_std.clone()
        grad_std_masked[self.mask == 0.0] = float('-inf')  # 마스크된 위치는 무시

        # 가장 작은 k개의 표준편차 값을 찾기
        _, min_indices = torch.topk(grad_std_masked.view(-1), k, largest=False, sorted=True)

        # 원래 차원으로 인덱스 변환
        attr_dims = grad_std.shape
        if len(attr_dims) == 2:
            _, dim2 = attr_dims
            max_i = (min_indices // dim2).tolist()
            max_j = (min_indices % dim2).tolist()
        else:
            raise NotImplementedError("마스킹 로직은 2D 텐서에 대해서만 구현되었습니다.")

        # 해당 가중치 마스킹
        self.mask_weight(max_i, max_j)
    
    # def update_mask_scores(self, curr_lr, total_iter):
    #     # Average the accumulated gradient norm over the epochs
    #     avg_grad_norm = self.gradient_accumulator / total_iter
        
    #     masked_indices = (F.sigmoid(self.mask_scores) <= 0.5).nonzero(as_tuple=True)[0]
    #     # with torch.no_grad():
    #     #     self.mask_scores -= (curr_lr * 10) * avg_grad_norm
        
    #     # Reset the gradient accumulator
    #     unmasked_grad_norm = avg_grad_norm.clone()
    #     unmasked_grad_norm[masked_indices] = float('-inf')
    #     #print((F.sigmoid(self.mask_scores) <= 0.5).sum())
    #     #import ipdb;ipdb.set_trace()
    #     _, max_idx = torch.topk(unmasked_grad_norm, 20)
    #     # 새로 마스킹할 인덱스의 mask_scores를 -1로 업데이트
    #     self.mask_scores[max_idx] = -1.
        
    #     self.gradient_accumulator = torch.zeros_like(self.mask_scores)
    #     # self.weight_grad = torch.zeros_like(self.mask_scores)

if __name__ == "__main__":
    pass