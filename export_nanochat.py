import argparse
import os
import json
import glob
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
from executorch.exir import EdgeCompileConfig, to_edge
import math
from dataclasses import dataclass

import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# Minimal GPT implementation (copied from nanochat to make standalone)


@dataclass
class GPTConfig:
    sequence_len: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    out = torch.cat([y1, y2], 3)
    out = out.to(x.dtype)
    return out


def repeat_kv(x, n_rep):
    if n_rep == 1:
        return x
    bs, n_kv_heads, slen, head_dim = x.shape
    return (
        x[:, :, None, :, :]
        .expand(bs, n_kv_heads, n_rep, slen, head_dim)
        .reshape(bs, n_kv_heads * n_rep, slen, head_dim)
    )


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def forward(self, x, cos_sin, kv_cache):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if kv_cache is not None:
            k, v = kv_cache.insert_kv(self.layer_idx, k, v)
        Tq = q.size(2)
        Tk = k.size(2)
        nrep = self.n_head // self.n_kv_head
        k, v = repeat_kv(k, nrep), repeat_kv(v, nrep)
        if kv_cache is None or Tq == Tk:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        elif Tq == 1:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        else:
            attn_mask = torch.zeros((Tq, Tk), dtype=torch.bool, device=q.device)
            prefix_len = Tk - Tq
            if prefix_len > 0:
                attn_mask[:, :prefix_len] = True
            attn_mask[:, prefix_len:] = torch.tril(torch.ones((Tq, Tq), dtype=torch.bool, device=q.device))
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, cos_sin, kv_cache):
        x = x + self.attn(norm(x), cos_sin, kv_cache)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
            }
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.rotary_seq_len = config.sequence_len * 10
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.transformer.wte.to(dtype=torch.bfloat16)

    def init_weights(self):
        self.apply(self._init_weights)
        torch.nn.init.zeros_(self.lm_head.weight)
        for block in self.transformer.h:
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            fan_out = module.weight.size(0)
            fan_in = module.weight.size(1)
            std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=1.0)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        if device is None:
            device = self.transformer.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def forward(self, idx, targets=None, kv_cache=None):
        B, T = idx.size()
        assert T <= self.cos.size(1)
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0 : T0 + T], self.sin[:, T0 : T0 + T]
        x = self.transformer.wte(idx)
        x = norm(x)
        for block in self.transformer.h:
            x = block(x, cos_sin, kv_cache)
        x = norm(x)
        softcap = 15
        logits = self.lm_head(x)
        logits = softcap * torch.tanh(logits / softcap)
        return logits



def main(args: argparse.Namespace):
    device = torch.device("cpu")

    print("Loading model...")
    model_files = glob.glob(os.path.join(args.model_dir, "model_*.pt"))
    model_file = model_files[0]

    meta_files = glob.glob(os.path.join(args.model_dir, "meta_*.json"))
    meta_file = meta_files[0]

    with open(meta_file, "r") as f:
        meta = json.load(f)

    model_config_kwargs = meta["model_config"]
    model_config = GPTConfig(**model_config_kwargs)
    with torch.device("meta"):
        model = GPT(model_config)

    model_data = torch.load(model_file, map_location=device, weights_only=True)
    model_data = {k.lstrip("_orig_mod."): v for k, v in model_data.items()}
    model_data = {k: v.float() if v.dtype == torch.bfloat16 else v for k, v in model_data.items()}

    model.to_empty(device=device)
    model.init_weights()
    model.load_state_dict(model_data, strict=True, assign=True)
    model.eval()

    print("Converting to ExportedProgram...")
    example_inputs = (torch.randint(0, model_config.vocab_size, (1, model_config.sequence_len), dtype=torch.long),)
    dynamic_shapes = {"idx": {1: torch.export.Dim("token_dim", min=1, max=model_config.sequence_len)}}
    with torch.inference_mode(), sdpa_kernel([SDPBackend.MATH]):
        ep_model = torch.export.export(model, example_inputs, dynamic_shapes=dynamic_shapes)

    print(f"Exporting to {args.output_path}...")
    edge_config = EdgeCompileConfig(_check_ir_validity=False)
    edge_manager = to_edge(ep_model, compile_config=edge_config)
    et_program = edge_manager.to_executorch()

    with open(args.output_path, "wb") as file:
        file.write(et_program.buffer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export nanochat model to executorch runtime")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="nanochat_model/",
    )
    parser.add_argument("--output_path", type=str, default="nanochat.pte")
    main(parser.parse_args())
