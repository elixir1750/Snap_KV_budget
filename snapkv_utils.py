import torch
import torch.nn.functional as F

def select_kv_indices(attn_weights, window_size, max_capacity, kernel_size, num_sinks):

    bsz, num_heads, window_len, full_kv_len = attn_weights.shape
    device = attn_weights.device

    score = attn_weights.sum(dim=-2) 
    score = score.view(bsz * num_heads, 1, full_kv_len)
    score = F.avg_pool1d(score, kernel_size=kernel_size, padding=kernel_size//2, stride=1)
    score = score.view(bsz, num_heads, full_kv_len)
    
    if full_kv_len <= max_capacity:
        return torch.arange(full_kv_len, device=device).repeat(bsz, num_heads, 1)
    candidate_len = full_kv_len - window_size - num_sinks
    k = max_capacity - num_sinks - window_size
    
    candidate_area = score[:, :, num_sinks : full_kv_len - window_size]
    topk_indices = torch.topk(candidate_area, k=k, dim=-1).indices
    topk_indices = topk_indices + num_sinks

    sink_indices = torch.arange(num_sinks, device=device).repeat(bsz, num_heads, 1)
    window_indices = torch.arange(full_kv_len - window_size, full_kv_len, device=device).repeat(bsz, num_heads, 1)
    
    final_indices = torch.cat([sink_indices, topk_indices, window_indices], dim=-1)
    final_indices = final_indices.sort(dim=-1).values
    return final_indices