# SGLang Runtime Patch

`sglang-0.5.2rc1-beliefkv.patch` is generated from and applies only to:

```text
tag:    v0.5.2rc1
commit: 18f91eb639084825717c0e3c3c7273492812ab71
```

Apply and validate it from the SGLang repository root:

```bash
git apply --check /home/longhao/experiment/BeliefKV/patches/sglang-0.5.2rc1-beliefkv.patch
git apply /home/longhao/experiment/BeliefKV/patches/sglang-0.5.2rc1-beliefkv.patch
beliefkv check-sglang "$PWD"
```

The patch covers:

- request metadata propagation through OpenAI chat, tokenizer, session, and
  scheduler types;
- BeliefKV deferred admission and abort handling;
- scheduler safe-point execution and tagged waiting-queue ordering;
- Radix/HiCache topology, lock, residency, and request-cache observer callbacks;
- runtime CLI flags.

All policy logic remains in `beliefkv/`. The patch deliberately calls private
HiCache methods only from the scheduler thread and is guarded by the exact
source contract. Do not apply with `--reject`, do not hand-resolve it onto a
newer release, and do not report results if `beliefkv check-sglang` fails.
