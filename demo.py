from transformers import AutoModelForCausalLM, AutoTokenizer
from modify_gptneox import apply_snapkv_to_model
import torch
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

model_id = "EleutherAI/pythia-70m"
print(f"正在加载模型: {model_id} ...")

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.float32,
    attn_implementation="eager" 
)

apply_snapkv_to_model(model)

long_text = "The quick brown fox jumps over the lazy dog. " * 20 
inputs = tokenizer(long_text, return_tensors="pt")
print("\n开始推理生成测试...")
with torch.no_grad():
    output = model.generate(
        **inputs, 
        max_new_tokens=20, 
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    print("\n生成结果：")
    print(tokenizer.decode(output[0], skip_special_tokens=True))