# CUDA Archivist smoke test (experimental)

This is a proposed CUDA path, not yet tested with Qwen3.6 or Qwen3.8, their tokenizers, or a GPU.
It must pass a real smoke test before any full training. It does not provision or
rent a machine. The full-run mode below is prepared but has not been exercised on CUDA.

Use an isolated CUDA environment with compatible versions of PyTorch, Transformers
(with Qwen3.6 support), bitsandbytes, PEFT, datasets and accelerate. The exact
versions must be recorded once verified together on the chosen machine. Copy only
the prepared train and validation data and manifest to that machine; keep the held-out
test split local. Prepared token IDs contain story text and labels: handle accordingly.

```sh
python -m training.archivist.cuda_prepare --data data/interim/archivist_sft --output /workspace/archivist-tokenized
python -m training.archivist.cuda_train --data /workspace/archivist-tokenized --output /workspace/archivist-smoke --smoke
```

For Qwen3.8 use a separate tokenized directory and adapter output, and explicitly
specify the same model for both commands:

```sh
python -m training.archivist.cuda_prepare --model Qwen/Qwen3.8-27B --data data/interim/archivist_sft --output /workspace/qwen38-tokenized
python -m training.archivist.cuda_train --model Qwen/Qwen3.8-27B --data /workspace/qwen38-tokenized --output /workspace/qwen38-smoke --smoke
```

Likewise, pass `--model Qwen/Qwen3.6-27B` to both commands for Qwen3.6. Run the
two models sequentially. The preparation step rejects tokenizations whose assistant
boundary differs from the full chat; do not bypass that check if a template fails.
Provision one 80 GB NVIDIA GPU and 150 GB disk as a starting point, not a proven
fit. For both original model downloads retained concurrently, allow 220 GB disk or
more; alternatively remove one verified cache after preserving its outputs and before
downloading the second. Peak VRAM and the final disk budget remain to be measured.

Preparation checks book disjointness against the manifest, assistant boundaries and
lengths, and refuses truncation. The smoke trains four optimizer steps, each with four
accumulated microbatches on the longest training scene, then evaluates the longest
validation scene. It reports training/evaluation elapsed time and peak CUDA allocated
and reserved memory after model loading. It does not measure loading time or peak
memory during loading. It uses NF4 and LoRA on linear modules in the last 16 language
layers; its quantization and target modules can differ from the Mac configuration. Neither a quality
comparison nor MLX adapter compatibility has been verified.

The program does not shut down a rented instance. Download outputs and explicitly
destroy the instance when finished; check the provider's storage and billing terms.

## Prepared full run

The full mode trains for one epoch on all 1,165 training scenes with microbatch 1,
four-step gradient accumulation, rank 8 LoRA, and checkpoints every 25 optimizer
steps. It evaluates the 157 validation scenes after training. It requires a matching
successful smoke report and refuses an existing output directory. The held-out test
split is not loaded. Example, after transferring the updated trainer and validating it:

```sh
python -m training.archivist.cuda_train --model Qwen/Qwen3.6-27B --data /workspace/qwen36-tokenized --output /workspace/qwen36-full --full --smoke-report /workspace/qwen36-smoke/smoke_metrics.json
```

The report gate checks matching model, split audit and finite smoke losses; it does
not establish successful adapter reload or extraction quality. Automatic resume from
checkpoints is not implemented. Do not start until adapter reload, representative
speed and pricing are verified. The smoke used the longest scene repeatedly, and
its speed is not a reliable estimate for the varied-length full dataset.
