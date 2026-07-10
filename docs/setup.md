# Environment Setup

BeliefKV has two environment levels:

1. `beliefkv`: lightweight policy, trace, metrics, and unit-test environment.
2. runtime environment: the future SGLang 0.5.2rc1 serving environment used
   when BeliefKV patches are connected to the model server.

The current repository only needs the lightweight environment.

## Lightweight Environment

Create the default environment:

```bash
cd /home/longhao/experiment/BeliefKV
conda env create -f environment.yml
conda activate beliefkv
python -m unittest discover -s tests
python -m beliefkv.cli plan examples/simple_snapshot.json
```

For development tools:

```bash
cd /home/longhao/experiment/BeliefKV
conda activate beliefkv
python -m pip install -e ".[dev]"
python -m unittest discover -s tests
```

If the environment already exists:

```bash
conda env update -f environment.yml --prune
```

## Runtime Environment

The first runtime integration target is SGLang `0.5.2rc1`. Do not mix the
lightweight policy environment with the serving runtime once CUDA dependencies
are involved.

Recommended workflow:

1. Keep `beliefkv` for policy development and replay tools.
2. Use a separate runtime conda environment for SGLang 0.5.2rc1 and model
   serving.
3. Install BeliefKV editable into that runtime environment:

```bash
cd /home/longhao/experiment/BeliefKV
python -m pip install -e .
```

4. Install the SGLang 0.5.2rc1 runtime from the selected local runtime checkout
   or wheel, then apply the BeliefKV runtime patches.

This separation keeps algorithm tests fast and reproducible while avoiding
accidental changes to the CUDA serving stack.
