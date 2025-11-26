import sys
import os
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend

current_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(current_dir, "nanochat_karpathy"))


def main():
    from nanochat.gpt import GPT, GPTConfig
    from nanochat.tokenizer import get_tokenizer

    device = torch.device("cpu")

    depth = 20
    tokenizer = get_tokenizer()
    max_seq_len = 2048
    vocab_size = tokenizer.get_vocab_size()
    num_layers = depth
    model_dim = depth * 64
    num_heads = max(1, model_dim + 127) // 128
    num_kv_heads = num_heads
    model_config_kwargs = dict(sequence_len=max_seq_len, vocab_size=vocab_size, n_layer=num_layers,n_head=num_heads,n_kv_head=num_kv_heads, n_embd=model_dim)

    model_config = GPTConfig(**model_config_kwargs)
    model = GPT(model_config)
    model.to_empty(device=device).to(torch.float32)
    model.init_weights()

    example_inputs = (
      torch.randint(0, vocab_size, (1, max_seq_len), dtype=torch.long),
    )
    dynamic_shapes = {"idx": {1: torch.export.Dim("token_dim", min=1, max=max_seq_len)}}
    with torch.inference_mode(), sdpa_kernel([SDPBackend]):
      ep = torch.export.export(model, example_inputs, dynamic_shapes=dynamic_shapes)

    print(model)

if __name__ == "__main__":
    main()
