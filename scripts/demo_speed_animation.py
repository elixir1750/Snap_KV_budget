import argparse
import time

import torch

from pyramidsinkkv import PyramidSinkKVConfig, compress_past_key_values, load_model_and_tokenizer

try:
    from rich.console import Console
    from rich.live import Live
    from rich.panel import Panel
except Exception:  # pragma: no cover - optional dependency
    Console = None
    Live = None
    Panel = None


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Terminal streaming demo for dense vs PyramidSinkKV generation.")
    parser.add_argument("--model_name_or_path", default="EleutherAI/pythia-70m")
    parser.add_argument("--prompt", default=("The future of efficient language model inference is " * 32))
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--methods", default="dense,pyramid")
    parser.add_argument("--compression_ratio", type=float, default=0.5)
    parser.add_argument("--sink_size", type=int, default=4)
    parser.add_argument("--recent_size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def make_config(method, args):
    if method == "dense":
        return PyramidSinkKVConfig(compression_ratio=1.0, budget_mode="dense")
    return PyramidSinkKVConfig(
        compression_ratio=args.compression_ratio,
        sink_size=args.sink_size,
        recent_size=args.recent_size,
        budget_mode="pyramid",
        score_method="key_norm",
        seed=args.seed,
    )


def render_panel(method, text, elapsed, count):
    speed = count / elapsed if elapsed > 0 else 0.0
    body = f"Method: {method}\nElapsed: {elapsed:.2f}s\nTokens: {count}\nSpeed: {speed:.2f} tok/s\n\n{text}"
    if Panel is not None:
        return Panel(body, title="PyramidSinkKV speed demo")
    return body


def stream_method(model, tokenizer, device, prompt, method, config, max_new_tokens):
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded.input_ids.to(device)
    logical_seq_len = int(input_ids.shape[1])
    dense = method == "dense"
    generated = []
    text = ""
    start = time.perf_counter()

    with torch.no_grad():
        outputs = model(input_ids=input_ids, use_cache=True, output_attentions=False)
        past_key_values = outputs.past_key_values
        if not dense:
            past_key_values, _ = compress_past_key_values(past_key_values, config, None)
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1)

    console = Console() if Console is not None else None
    live = Live(render_panel(method, text, 0.0, 0), console=console, refresh_per_second=8) if Live is not None else None
    if live is not None:
        live.start()
    else:
        print(f"\n=== {method} ===")

    try:
        for step in range(max_new_tokens):
            token_id = int(next_token.item())
            generated.append(token_id)
            text = tokenizer.decode(generated, skip_special_tokens=True)
            elapsed = time.perf_counter() - start
            if live is not None:
                live.update(render_panel(method, text, elapsed, len(generated)))
            else:
                print(f"\r[{method}] {len(generated)} tok | {len(generated)/max(elapsed, 1e-9):.2f} tok/s | {text[-100:]}", end="", flush=True)

            if step == max_new_tokens - 1:
                break

            current_ids = next_token.view(1, 1).to(device)
            position_ids = torch.full((1, 1), logical_seq_len, dtype=torch.long, device=device)
            cache_position = torch.arange(logical_seq_len, logical_seq_len + 1, dtype=torch.long, device=device)
            with torch.no_grad():
                out = model(
                    input_ids=current_ids,
                    past_key_values=past_key_values,
                    position_ids=position_ids,
                    cache_position=cache_position,
                    use_cache=True,
                )
            past_key_values = out.past_key_values
            next_token = torch.argmax(out.logits[:, -1, :], dim=-1)
            logical_seq_len += 1
    finally:
        if live is not None:
            live.stop()
        else:
            print()


def main():
    args = build_arg_parser().parse_args()
    model, tokenizer, device = load_model_and_tokenizer(args.model_name_or_path, args.device, args.dtype)
    for method in [item.strip() for item in args.methods.split(",") if item.strip()]:
        stream_method(model, tokenizer, device, args.prompt, method, make_config(method, args), args.max_new_tokens)


if __name__ == "__main__":
    main()
