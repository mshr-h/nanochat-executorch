# nanochat-executorch
Run Karpathy’s nanochat LLM on-device with PyTorch ExecuTorch

## Setup

```bash
git clone https://github.com/karpathy/nanochat nanochat_karpathy
uv sync
uv run maturin develop --release --manifest-path nanochat_karpathy/rustbpe/Cargo.toml
```

