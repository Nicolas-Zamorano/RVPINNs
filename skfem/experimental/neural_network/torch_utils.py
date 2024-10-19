import torch

def reduce_tensor(tensor: torch.Tensor):
    
    if tensor.dim() == 1:
        non_zero_rows = tensor.abs().sum(dim=1) != 0

        return tensor[non_zero_rows]
    
    if tensor.dim() == 2:
        non_zero_rows = tensor.abs().sum(dim=1) != 0
        non_zero_cols = tensor.abs().sum(dim=0) != 0

        return tensor[non_zero_rows][:, non_zero_cols]

    