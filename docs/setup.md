# Environment Setup

BeliefKV uses two environments so policy development does not destabilize the
CUDA serving stack.

## Control-Plane Environment

```bash
cd /home/longhao/experiment/BeliefKV
conda env create -f environment.yml
conda activate beliefkv
python -m pip install -e ".[dev]"
python -m unittest discover -s tests -v
python -m pip check
```

For an existing environment:

```bash
conda env update -f environment.yml --prune
conda activate beliefkv
python -m pip install -e ".[dev]"
```

The control plane intentionally has no PyTorch/CUDA dependency.

## SGLang Runtime Environment

Create a separate environment, for example `beliefkv-sglang`, using the CUDA and
PyTorch versions required by the target machine. Then obtain the exact source:

```bash
git clone --branch v0.5.2rc1 https://github.com/sgl-project/sglang.git \
  /home/longhao/experiment/sglang-beliefkv
cd /home/longhao/experiment/sglang-beliefkv
git rev-parse HEAD
```

The required commit is:

```text
18f91eb639084825717c0e3c3c7273492812ab71
```

Apply and validate the patch before installation:

```bash
git apply --check \
  /home/longhao/experiment/BeliefKV/patches/sglang-0.5.2rc1-beliefkv.patch
git apply \
  /home/longhao/experiment/BeliefKV/patches/sglang-0.5.2rc1-beliefkv.patch

conda activate beliefkv-sglang
python -m pip install -e /home/longhao/experiment/BeliefKV
beliefkv check-sglang /home/longhao/experiment/sglang-beliefkv
cd /home/longhao/experiment/sglang-beliefkv/python
python -m pip install -e ".[all]"
python -m pip check
```

Using `cd .../python && pip install -e ".[all]"` avoids the editable-path extras
parsing problem caused by placing `[all]` inside an absolute path argument.

## Start A Patched Server

At minimum, BeliefKV requires HiCache and a config file:

```bash
python -m sglang.launch_server \
  --model-path /path/to/model \
  --enable-hierarchical-cache \
  --hicache-size 96 \
  --enable-beliefkv \
  --beliefkv-config /home/longhao/experiment/BeliefKV/configs/beliefkv_single_gpu.json
```

The P5A observed admission slice is disabled by default. For a dedicated
experiment config, generate it with:

```bash
python scripts/prepare_deepagents_server_config.py \
  --server-dir RUN_DIR/server \
  --enable-observed-admission \
  --enable-running-retraction \
  --observed-admission-active-kv-high-watermark-ratio 0.8 \
  --observed-admission-min-active-requests 1
```

`--enable-running-retraction` additionally enables the observed P5 transaction:
selected running requests are retracted, exact physical bundles are offloaded,
and replacement tickets remain blocked until the DMA ACK and allocator-free
postcondition. GPU-only recompute drop stays disabled unless
`--allow-running-retraction-recompute-drop` is supplied.

Set `hbm_capacity_bytes`, `host_capacity_bytes`, and `kv_bytes_per_token` for the
actual model/runtime. A wrong `kv_bytes_per_token` makes policy estimates wrong;
the authoritative allocator usage is still reported separately for safety.
`--hicache-size` uses decimal GB in the pinned SGLang implementation and overrides
`--hicache-ratio`. The Deep Agents launcher defaults to 96 GB; set
`HICACHE_SIZE_GB=128` or `156` for an explicit capacity sweep. BeliefKV always
replaces a stale JSON Host-capacity hint with the allocator's measured token
capacity at startup.
For integration tests, set `runtime_audit_path` to an experiment-local JSONL
file. The field is `null` by default and therefore adds no scheduler I/O.

## Baseline Discipline

Use both baselines below:

1. exact unpatched SGLang at the pinned commit;
2. patched SGLang with `--enable-beliefkv` omitted.

The first measures upstream performance. The second quantifies the disabled
patch overhead. Keep model, quantization, CUDA graph settings, HiCache policy,
request trace, seed, and GPU clocks identical.

## Current Validation Limit

The repository validates source compatibility, compiles patched Python files,
and has completed one Qwen2.5-0.5B-Instruct GPU smoke run covering untagged
bypass and a tagged root/spawn-child sequence. Before reporting performance,
run model-specific pressure tests, long mixed workloads, abort/reset fault
injection, GPU OOM pressure, and CPU-host-capacity exhaustion.
