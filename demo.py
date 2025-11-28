from executorch.runtime import Runtime
import os
import pickle
import argparse
import torch


def main():
    parser = argparse.ArgumentParser(
        description="Generate text with executorch runtime"
    )
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--pte_path", type=str, default="nanochat.pte")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--max_tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    args = parser.parse_args()

    print("Loading tokenizer...")
    tokenizer_path = os.path.join(args.model_dir, "tokenizer.pkl")
    with open(tokenizer_path, "rb") as f:
        tokenize = pickle.load(f)

    # Get BOS token
    try:
        bos_token_id = tokenize.encode_single_token("<|bos|>")
    except KeyError:
        try:
            bos_token_id = tokenize.encode_single_token("<|endoftext|>")
        except Exception as _:
            bos_token_id = None

    # Load executorch runtime
    runtime = Runtime.get()
    program = runtime.load_program(args.pte_path)
    method = program.load_method("forward")

    # Encode prompt into tokens
    input_ids = tokenize.encode_ordinary(args.prompt)
    print(f"\nPrompt: {args.prompt}")
    print(f"Encoded to {len(input_ids)} tokens")
    print()

    print("Generating...")
    print("-" * 50)
    print(args.prompt, end="", flush=True)

    # Generate text
    x = torch.tensor([input_ids], dtype=torch.long)

    for _ in range(args.max_tokens):
        logits = method.execute(x)[0]
        logits = logits[:, -1, :]  # (batch_size, vocab_size)
        logits = logits / args.temperature

        if args.top_k > 0:
            v, _ = torch.topk(logits, min(args.top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("Inf")

        probs = torch.nn.functional.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        token_str = tokenize.decode([next_token.item()])
        print(token_str, end="", flush=True)

        x = torch.cat([x, next_token], dim=1)
        if bos_token_id is not None and next_token.item() == bos_token_id:
            break


if __name__ == "__main__":
    main()
