import torch
import time
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from snapkv_utils import select_kv_indices
from eval_ppl import manual_snapkv_compress 

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def benchmark_inference(model, tokenizer, prompt, max_new_tokens=50, use_snapkv=False, capacity=64):
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids
    prompt_len = input_ids.shape[1]

    # --- 1. 测量 TTFT (Time To First Token) ---
    start_time = time.perf_counter()
    with torch.no_grad():
        # Prefill 阶段
        outputs = model(input_ids, use_cache=True, output_attentions=True)
        past_kv = outputs.past_key_values
        attentions = outputs.attentions
        
        if use_snapkv:
            past_kv = manual_snapkv_compress(past_kv, attentions, max_capacity=capacity)
        
        # 生成第一个词
        next_token_logit = outputs.logits[:, -1, :]
        next_token = torch.argmax(next_token_logit, dim=-1)
    
    ttft = time.perf_counter() - start_time

    # --- 2. 测量 TPOT (Time Per Output Token) ---
    decoding_times = []
    current_ids = next_token.unsqueeze(0)
    
    for _ in range(max_new_tokens - 1):
        step_start = time.perf_counter()
        with torch.no_grad():
            out = model(current_ids, past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values
            current_ids = torch.argmax(out.logits[:, -1, :], dim=-1).unsqueeze(0)
        decoding_times.append(time.perf_counter() - step_start)
    
    tpot = sum(decoding_times) / len(decoding_times)
    
    # --- 3. 计算 Throughput ---
    total_time = ttft + sum(decoding_times)
    throughput = max_new_tokens / total_time
    
    return ttft, tpot, throughput

def main():
    model_id = "EleutherAI/pythia-70m"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float32, attn_implementation="eager"
    )

    # 为了让加速效果明显，我们使用一个较长的 Prompt (比如 1000 tokens)
    # 因为 KV Cache 压缩的优势在长文本下才体现得出来
    long_prompt = "Hello, today is a good day. " * 200 
    
    print(f"Prompt 长度约为: {len(tokenizer.encode(long_prompt))} tokens")
    print("-" * 50)

    # 测试 Baseline
    print("正在测试 Baseline (不压缩)...")
    ttft_b, tpot_b, thr_b = benchmark_inference(model, tokenizer, long_prompt, use_snapkv=False)
    
    # 测试 SnapKV
    print("正在测试 SnapKV (容量 64)...")
    ttft_s, tpot_s, thr_s = benchmark_inference(model, tokenizer, long_prompt, use_snapkv=True, capacity=64)

    print("\n" + "="*50)
    print("加速效果测量结果")
    print("="*50)
    print(f"{'指标':<15} | {'Baseline':<12} | {'SnapKV':<12} | {'提升比':<10}")
    print("-" * 55)
    print(f"{'TTFT (s)':<15} | {ttft_b:12.4f} | {ttft_s:12.4f} | {ttft_b/ttft_s:10.2f}x")
    print(f"{'TPOT (s/tok)':<15} | {tpot_b:12.4f} | {tpot_s:12.4f} | {tpot_b/tpot_s:10.2f}x")
    print(f"{'Throughput':<15} | {thr_b:12.4f} | {thr_s:12.4f} | {thr_s/thr_b:10.2f}x")
    print("="*50)

if __name__ == "__main__":
    main()