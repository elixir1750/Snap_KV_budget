import torch
from snapkv_utils import select_kv_indices

def snapkv_forward_wrapper(self, *args, **kwargs):

    kwargs["output_attentions"] = True
    
    outputs = self._original_forward(*args, **kwargs)
    
    attn_output = outputs[0]
    present = outputs[1] if len(outputs) > 1 else None
    attn_weights = outputs[2] if len(outputs) > 2 else None

    max_capacity = 64
    window_size = 16

    if present is not None:

        if hasattr(present, "key_cache"):
            layer_idx = getattr(self, "layer_idx", 0)
            key_states = present.key_cache[layer_idx]
            value_states = present.value_cache[layer_idx]
            
            kv_seq_len = key_states.shape[-2]
            
            if kv_seq_len > max_capacity + 10:
                print(f"[!!! SNAPKV ACTUALLY COMPRESSING !!!] Layer {layer_idx}: {kv_seq_len} -> {max_capacity}")
                
                if attn_weights is not None:
                    if isinstance(attn_weights, tuple): attn_weights = attn_weights[0]
                    obs_attn = attn_weights[:, :, -window_size:, :]
                    indices = select_kv_indices(obs_attn, window_size, max_capacity, 5, 4)
                    
                    indices = indices.to(key_states.device)
                    dim = key_states.shape[-1]
                    gather_idx = indices.unsqueeze(-1).expand(-1, -1, -1, dim)
                    
                    present.key_cache[layer_idx] = torch.gather(key_states, 2, gather_idx)
                    present.value_cache[layer_idx] = torch.gather(value_states, 2, gather_idx)

    return outputs

def apply_snapkv_to_model(model):

    print("\n正在直接向模型实例注入 SnapKV 补丁...")
    for i, layer in enumerate(model.gpt_neox.layers):
        attn_module = layer.attention
        if not hasattr(attn_module, "_original_forward"):
            attn_module._original_forward = attn_module.forward
        attn_module.forward = snapkv_forward_wrapper.__get__(attn_module, type(attn_module))
        attn_module.layer_idx = i
        
    print(f"成功为 {len(model.gpt_neox.layers)} 个 Attention 层注入了补丁！\n")