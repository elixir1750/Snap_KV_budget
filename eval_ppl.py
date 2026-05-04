import torch
import math
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from datasets import load_dataset
from snapkv_utils import select_kv_indices

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def manual_snapkv_compress(past_kv, attentions, max_capacity=64, window_size=16):
    new_cache = DynamicCache()
    raw_data = list(past_kv)

    if len(raw_data) == 2 and isinstance(raw_data[0], (list, tuple)):
        all_keys = raw_data[0]
        all_values = raw_data[1]
        num_layers = len(all_keys)
        print(f"[SnapKV] 检测到列表结构，正在压缩 {num_layers} 层...")
        
        for i in range(num_layers):
            k = all_keys[i]
            v = all_values[i]

            attn = attentions[i]
            if isinstance(attn, tuple): attn = attn[0]
            obs_attn = attn[:, :, -window_size:, :]
            
            indices = select_kv_indices(obs_attn, window_size, max_capacity, 5, 4)
            head_dim = k.shape[-1]
            gather_idx = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim).to(k.device)
            
            compressed_k = torch.gather(k, 2, gather_idx)
            compressed_v = torch.gather(v, 2, gather_idx)
            
            new_cache.update(compressed_k, compressed_v, i)
            
    else:
        print(f"[SnapKV] 检测到元组结构，正在压缩 {len(raw_data)} 层...")
        for i, layer_data in enumerate(raw_data):
            k, v = layer_data[0], layer_data[1]
            attn = attentions[i]
            if isinstance(attn, tuple): attn = attn[0]
            obs_attn = attn[:, :, -window_size:, :]
            indices = select_kv_indices(obs_attn, window_size, max_capacity, 5, 4)
            gather_idx = indices.unsqueeze(-1).expand(-1, -1, -1, k.shape[-1]).to(k.device)
            new_cache.update(torch.gather(k, 2, gather_idx), torch.gather(v, 2, gather_idx), i)

    return new_cache

def evaluate_ppl_on_sample(model, tokenizer, text_chunk, prefill_len=500, eval_len=50, use_snapkv=False):
    input_ids = tokenizer(text_chunk, return_tensors="pt").input_ids
    if input_ids.shape[1] < prefill_len + eval_len: return None

    prefill_ids = input_ids[:, :prefill_len]
    eval_ids = input_ids[:, prefill_len : prefill_len + eval_len]
    loss_fct = torch.nn.CrossEntropyLoss()
    nlls = []

    with torch.no_grad():
        outputs = model(prefill_ids, use_cache=True, output_attentions=True)
        past_kv = outputs.past_key_values
        attentions = outputs.attentions
        
        if use_snapkv:
            print(f"\n[!!!] 触发 SnapKV 压缩 | 原始长度: {past_kv.get_seq_length()}")
            past_kv = manual_snapkv_compress(past_kv, attentions, max_capacity=64)

            print(f"--> 压缩后对象长度确认: {past_kv.get_seq_length()}")
        else:
            print(f"\nBaseline 模式 | 长度保持: {past_kv.get_seq_length()}")

        next_token_logit = outputs.logits[:, -1, :]

        for i in range(eval_ids.shape[1]):
            target_token = eval_ids[:, i]
            loss = loss_fct(next_token_logit, target_token)
            nlls.append(loss.item())
            
            out = model(target_token.unsqueeze(0), past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values
            next_token_logit = out.logits[:, -1, :]
            
    return math.exp(sum(nlls) / len(nlls))

def main():
    USE_SNAPKV = True
    
    print("="*50)
    print(f"实验启动: {'SnapKV 模式' if USE_SNAPKV else 'Baseline 模式'}")
    print("="*50)

    print("\n加载模型 Pythia-70M (Eager 模式)...")
    model_id = "EleutherAI/pythia-70m"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float32, attn_implementation="eager"
    )

    print("加载数据集 wikitext-2...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    full_text = "\n\n".join([t for t in dataset["text"] if len(t.strip()) > 0])
    
    ppl_list = []
    chunk_size = 4000 
    for i in range(5):
        text_chunk = full_text[i*chunk_size : (i+1)*chunk_size]
        print(f"\n处理片段 {i+1}/5...")
        ppl = evaluate_ppl_on_sample(model, tokenizer, text_chunk, use_snapkv=USE_SNAPKV)
        if ppl:
            ppl_list.append(ppl)
            print(f"--> PPL: {ppl:.2f}")

    if ppl_list:
        print("\n" + "="*50)
        print(f"平均 PPL ({'SnapKV' if USE_SNAPKV else 'Baseline'}): {sum(ppl_list)/len(ppl_list):.2f}")
        print("="*50)

if __name__ == "__main__":
    main()