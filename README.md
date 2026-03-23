# nanochat-executorch
Run Karpathy's nanochat with PyTorch ExecuTorch

## Prerequisites

- uv

## Setup and run

Download model from HF and setup virtual env.

```bash
uvx hf download sdobson/nanochat --local-dir nanochat_model
uv sync
```

Convert model to pte.

```bash
mkdir -p output
uv run python export_nanochat.py --model_dir nanochat_model/ --output_path output/nanochat.pte
```

Run the model. It' super slow because of no KV Caching support.

```bash
uv run python demo.py --prompt "Once upon a time" --pte_path "output/nanochat.pte" --tokenizer_path "nanochat_model/tokenizer.pkl"
```

The output looks something like this:

```
Skipping import of cpp extensions due to incompatible torch version 2.10.0+cu128 for torchao version 0.15.0             Please see https://github.com/pytorch/ao/issues/2919 for more info
Loading tokenizer...
[program.cpp:153] InternalConsistency verification requested but not available

Prompt: Once upon a time
Encoded to 4 tokens

Generating...
--------------------------------------------------
Once upon a time[cpuinfo_utils.cpp:71] Reading file /sys/devices/soc0/image_version
[cpuinfo_utils.cpp:87] Failed to open midr file /sys/devices/soc0/image_version
[cpuinfo_utils.cpp:100] Reading file /sys/devices/system/cpu/cpu0/regs/identification/midr_el1
[cpuinfo_utils.cpp:109] Failed to open midr file /sys/devices/system/cpu/cpu0/regs/identification/midr_el1
[cpuinfo_utils.cpp:125] CPU info and manual query on # of cpus dont match.
 there was a great drought in the land and most of the crops could not be grown. The people had not enough food and they were worried. 

One day a great drought came and the land was barren. The people had nothing to sell and
```

## References

- [karpathy/nanochat: The best ChatGPT that $100 can buy.](https://github.com/karpathy/nanochat)
- [sdobson/nanochat · Hugging Face](https://huggingface.co/sdobson/nanochat)
