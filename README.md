# 5M-parameter arithmetic model

Can a ~5M-parameter causal decoder learn multi-digit addition and subtraction from
a synthetic corpus, and does it generalize to operand lengths it never saw?

Short answer: it learns the lengths it is trained on, very well, and **does not
extrapolate to any length outside that range** — in either direction.

`report.html` is the visual write-up. This file covers how to run things and the
throughput work, which is where most of the practical value turned out to be.

## Results

Exact-match accuracy, greedy decode, answers accepted in forward or reversed order
(see *Scoring caveat*). Best checkpoint per run.

| corpus | tpp | 1-digit | 2–4d | 2–9d | 5–6d | 10–12d |
|---|---|---|---|---|---|---|
| 2–4 digit (run 3, original) | 12.8 | 0.00% | **78.70%** | 9.50% | 0.00% | — |
| 2–9 digit | 12.8 | 0.00% | 29.85% | 47.05% | 34.65% | 0.00% |
| 2–9 digit | 20 | 0.00% | 47.50% | 70.80% | 59.25% | 0.00% |
| 2–9 digit | 100 | 0.00% | 34.85% | 70.20% | 80.85% | 0.00% |
| **1–9 digit** | **20** | 11.03% | 61.90% | **86.25%** | **81.85%** | 0.00% |
| 1–4 digit | 100 | 97.06% | 60.15% | — | — | 0.00% |
| **1–4 digit + SFT** | **100** | **100.00%** | **99.05%** | — | — | 0.00% |

Best model overall is **1–4 digit @ 100 tpp + SFT**
(`checkpoints/run10_sft/sft_final.pt`) — 99.05% on the original 2–4 digit eval.
Best model *without* SFT, and the widest digit range, is **1–9 digit at 20 tpp**
(`checkpoints/run7_1d_20tpp/ckpt_final.pt`).

### Runs 9–10: SFT fixes the short-digit wall

Run 9 narrowed to 1–4 digits at 100 tpp and still scored only 60.15% on the original
2–4 digit eval, with **2-digit answers at 1%**. The arithmetic was correct; the model
would not stop (`18+15=` → `3333`). Pretraining on a packed stream makes `<eos>` one
token among millions, never the target of a distinct example.

`sft_stop.py` applies the recipe from `../llm-modern-arch-experiment/src/modern_lm/sft.py`
— whose own docstring says its SFT exists because "packed pretraining text almost never
teaches 'stop after answering'". Loss is masked to `-100` on the prompt so only the
answer digits and terminal `<eos>` are supervised (36.9% of tokens). The SFT set is
balanced by *answer length* (40k each at 1–5 digits) rather than mirroring pretraining's
~80% four-digit skew, which is what caused the bug.

| answer length | pre-SFT | post-SFT |
|---|---|---|
| 1 digit | 0% | 97% |
| 2 digit | 1% | 94% |
| 3 digit | 33% | 99% |
| 4 digit | 84% | 100% |
| **overall (2–4d eval)** | **60.15%** | **99.05%** |
| 5–6 digit (unseen) | 0.00% | 0.00% |

**Cost: 40 seconds, ~0.58M supervised tokens — 0.114% of the pretraining budget.**

Two things worth noting beyond the headline:

- **Strict accuracy went 21.55% → 99.05%**, exactly matching lenient. Every earlier run
  leaned on a scorer accepting forward *or* reversed answers, because 50/50 reversed
  training left the model emitting a mix. After SFT the two metrics coincide — it emits
  forward order every time (`7+5=` → `21` became `12`). The scoring caveat below no
  longer applies to the SFT model.
- **SFT fixes format, not capability.** 5–6 digit is 0.00% before and after. Supervising
  the completion teaches when to stop and which order to emit; it cannot teach arithmetic
  the model never learned.

### Run 8: four operations

Adding `mul` and `div` required growing the **vocab from 14 to 16 tokens** (`x`, `/`),
which makes run-8 checkpoints and all earlier checkpoints mutually unloadable (the tied
embedding is `V × 256`). Same recipe as run 7 otherwise: 1–9 digit, 20 tpp, 287s.

| operation | accuracy | correct length | correct first digit |
|---|---|---|---|
| sub | 83.20% | — | — |
| add | 75.13% | — | — |
| div (exact only) | 60.53% | — | — |
| **mul** | **5.20%** | 54.2% | 12.2% |

**Multiplication is not learnable at this scale, and not because of length.** It gets the
answer length right 54% of the time and the first digit right only 12% — it knows how big
the answer should be and cannot compute it (`9x9` → `1011`). Addition is digit-local with
a carry; multiplication needs every partial product simultaneously.

Exact division reaching 60.53% is a property of the corpus, not of division: the
generator builds it backward as `a = q·d` with `d` in 2–9, so recovering `q` is closer to
a digit-local scan than to a full product.

Splitting the same 5.1M params and token budget across four operations costs ~10 points
on add/sub versus run 7.

Findings:

- **Train on the lengths you need.** Extending the training range moves the wall
  with it; it never removes it. 10–12 digit is 0.00% across four runs and a 5×
  token sweep (12.8 → 100 tpp).
- **Filling the short end lifts the whole range.** Adding 1-digit operands gained
  +14 to +22 points at *every* answer length, not just the short ones.
- **More tokens are not uniformly good.** 100 tpp beat 20 tpp on 5–6 digit
  (80.85% vs 59.25%) but lost ground on 2–4 digit (34.85% vs 47.50%). At 5.1M
  params the model reallocates capacity toward wherever the data mass sits.
- **The short-side wall was a formatting bug, not a capability limit.** 40s of SFT with
  prompt-masked loss took the 2–4 digit eval from 60.15% to 99.05%, finally beating run
  3's 78.70% by 20 points. The long-side wall did not move at all.
- **Two different walls.** Long side: cannot compute. Short side: computes
  perfectly and cannot stop — on 1-digit answers the first emitted digit is
  correct **91/91 = 100%**, but `<eos>` ranks 10th on average, so it runs on
  (`1+2=` → `3633`). Different failures, different fixes.
- **Do not select checkpoints by loss.** `ckpt_final` beat the loss-selected
  `ckpt_best` on every split in both runs where both were scored, by up to 10
  points, on loss differences of 0.0005 — noise. See *Checkpoint selection*.

## Throughput: it was 7x too slow

The original `train_fast.py` ran at **~53,700 tok/s**. The same model now runs at
**~372,000 tok/s** — a 6.9x speedup, and about 1.9x the reference repo's 5M run
(`../llm-modern-arch-experiment`, 193,236 tok/s). A full 20-tpp run went from a
projected ~20 minutes to **276 seconds**.

Three causes, in order of impact:

### 1. `torch.compile` was disabled (dominant)

The original carried the comment *"no torch.compile: overhead > gain at 5M"*. That
is backwards. At `dim=256` every kernel is tiny and launch-bound, so eager mode
pays kernel-launch overhead on ops that take microseconds. This is exactly the
size where compile helps most. The reference repo compiles both the model and the
Newton-Schulz kernel (`train.py:443`, `muon.py:17`).

**The ordering trap:** `torch.compile` prefixes every parameter name with
`_orig_mod.`, and `build_optimizer` splits Muon vs AdamW by matching names starting
with `blocks.`. Compiling *before* building the optimizer routes **zero parameters
to Muon** and dies with `ValueError: optimizer got an empty parameter list`.
Compile **after** `build_optimizer`. Verify the split: 5,111,808 params to Muon,
7,296 to AdamW. The same trap is noted at reference `muon.py:132`.

Checkpoints must also be saved from the unwrapped module or the `_orig_mod.` prefix
leaks into `state_dict` and breaks loading:
`torch.save(getattr(model, "_orig_mod", model).state_dict(), ...)`.

### 2. Per-step shuffle of the whole corpus (~12%)

```python
random.shuffle(train_seqs)              # 258k-element Python list, ~70 ms/step
batch = torch.tensor(train_seqs[:MICRO], device=dev)
```

It shuffled 258,000 entries to select 128, every step. Replaced with a shuffled
`randperm` walked in order, which is both O(MICRO) and gives a real no-repeat
guarantee (`STEPS*MICRO <= corpus`, asserted at startup) rather than
sampling-with-replacement.

### 3. Full eval pass every 50 steps (~9%)

Walked all eval sequences — 26% as many tokens as the training steps between
evals. Now a fixed 512-sequence subset; holding it fixed also tightens
step-to-step comparison.

Also: the corpus is materialized once as a resident GPU tensor (`int16` — the
vocab is 14 tokens, so it is lossless and cuts gather bandwidth 4x), instead of
rebuilding CPU tensors per step.

### On comparing against the reference repo

The 1.9x figure over `llm-modern-arch-experiment` is **not** an apples-to-apples
win. Measured directly, with identical body, batch, GPU and compile settings:

| vocab | tok/s |
|---|---|
| 14 (ours) | 385,339 |
| 16,384 (reference) | 251,045 |

**Vocab size alone is worth 1.53x.** At `dim=256` the lm_head dominates: it is
0.07% of forward FLOPs at V=14 versus **45%** at V=16,384, and the logits tensor
is 0.92 MB versus 1,073 MB. Correcting for vocab, our advantage is ~1.2x, most of
which is the GPU-resident corpus. The reference streams a 16 GB file and cannot do
that. Tied vs untied embeddings is negligible (~1.5%).

Note the reference README counts "body" params to exclude the vocab as doing "no
matmul work" — true for the embedding lookup, but the lm_head projection is very
much matmul work at this size.

## Running it

No `python` on PATH — use `python3` (3.12). Torch 2.14+cu130 is system-wide and
sees the GPU (NVIDIA GB10). No venv, no requirements file, no test suite.

**Scripts read and write bare filenames in the current working directory.** Work
in a scratch dir with the data files flat, as `run4/` does (symlinked scripts,
generated corpora local). Nothing reads from `data/`.

```bash
python3 gen_splits1d.py                    # corpus: 1-9 digit, 6M lines, + oversampled 1-digit
python3 make_reversed.py --in train1d.txt --out train1d_rev.txt --seed 41
python3 train_fast.py                      # 20 tpp = STEPS 3120, ~276s
python3 eval_any.py ckpt_final.pt --rev --n 2000 \
    --splits "eval=eval1d.txt" "gentest=gentest1d.txt"
python3 audit_corpus.py                    # generator correctness; expect TOTAL ERRORS: 0
```

`eval_any.py` supersedes `evaluate_math.py`: it takes any checkpoint and any
splits, batches the decode, and reports a per-answer-length breakdown plus both
scorers. `evaluate_math.py` is kept for provenance — it reproduces the original
78.70% exactly.

## Things that will bite you

**Held-out sets can be contaminated by construction.** The 10–12 digit
`gentest_ops.txt` scored 9.67%, which looked like the first crack in the length wall. It
was an artifact: `--min-digits 10` bounds the *dividend*, but division is built as
`a = q·d` with small `d`, so quotients land back inside the trained range. Restricted to
genuine 10–12 digit add/sub the score is 0.00%. When holding out by operand length, check
the answer length too.

**Cross-file contamination.** The generator dedups *within* a file, not across
them. Every split pair generated so far leaked train lines into eval — 533 at 3M
lines, 744 at 6M, 1,344 at 30M. Always diff eval against train and drop the
overlap. `gen_splits*.py` does not do this; the audit step in the workflow above
does.

**Scoring caveat.** The scorer accepts an answer in forward **or** reversed order,
because training uses 50% reversed answers. This matches the convention behind the
published 78.70%, but it cannot distinguish "right digits, right order" from "right
digits, wrong order". Strict forward-only accuracy is much lower and swings wildly
with which emission mode a run happens to settle into (run 3 emitted mostly
forward; later runs mostly reversed). Compare runs on the either-order number, and
do not read the strict number as a capability measure.

**Checkpoint selection.** `ckpt_best.pt` is chosen by eval loss, and eval loss is a
poor proxy here. In the 100-tpp run `ckpt_final` beat it by 8–10 points on every
split while being 0.0005 *worse* on loss; in run 8 the ordering reversed. The selection is effectively arbitrary — score both.

**Single-digit is exhaustive.** Only 136 distinct 1-digit add/sub problems exist,
so train/test overlap in that band is total by construction. The 11.03% figure
measures "can it emit a 1-digit answer at all", not generalization.

**`STEPS` also sets the LR schedule.** The cosine spans `STEPS`, so changing the
token budget rescales the schedule — which is correct, but means two runs at
different budgets are not comparable mid-run, only at completion.

## Layout

```
math_corpus_generator.py   generator (add/sub/mul/div/pow, algebra, multi-op, word forms)
audit_corpus.py            re-parses 20k generated lines and asserts every answer
gen_splits*.py             split drivers: 9 = 2-9 digit, 20 = 20 tpp, 100 = 100 tpp, 1d = 1-9 digit
make_reversed.py           train.txt -> train_rev.txt (50% reversed answers)
train_fast.py              trainer (ModernGPT + Muon + AdamW, bf16, compiled)
train_math.py              run 1 baseline (GPT-2-ish, AdamW, fp32) - different architecture
eval_any.py                parameterized evaluator
checkpoints/run7_1d_20tpp/ best model
results/                   per-run JSON + logs
```
