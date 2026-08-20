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

### Run 16: word forms — `36 multiplied by 24 = 864`

50% word-form operators, spaces kept, vocab 20 → 40 (19 letters + space). Same
operations, same digit ranges, matched SFT exposure — the only variable is how the
operator is written.

| operation | symbolic | **word form** | gap |
|---|---|---|---|
| div | 100.00% | **99.86%** | 0.14 |
| add | 99.08% | **99.00%** | 0.08 |
| sub | 98.83% | **98.83%** | **0.00** |
| mul | 33.33% | **35.58%** | −2.25 (word higher) |
| multi-op | 16.25% | — | — |

**Encoding is essentially free.** Every operation lands within ~2 points across the two
encodings; subtraction is identical to two decimals. The model learned that `multiplied
by` — a 13-token phrase — and `x` denote the same operation, rather than treating them as
separate tasks.

**Both encodings fail identically**, which is the stronger evidence: `7 ^ 3` → `327` and
`7 to the power of 3` → `327`. Same wrong digits from the same model on the same problem
written two ways. Parallel lookup tables would diverge.

Best add/sub/div of the study (99.08% / 98.83% / 100.00%) while carrying five operations,
compositional expressions, two encodings and a 40-token vocab. SFT held-out loss 0.1905,
best of any multi-operation run. Throughput 371,209 tok/s — doubling the vocab cost
nothing measurable.

Mul (33–36%) and multi-op (16.25%) sit where they have all study, with the same by-length
collapse (mul: 86% at 5-digit answers, 6% at 9-digit). Encoding, composition, operation
count and vocabulary size all leave the digit-carrying ceiling untouched.

*Note: dropping `--compact` reformatted every line (`73 + 5 = 78`, not `73+5=78`), so run
16's absolute numbers are not comparable to runs 1–15. The internal symbolic-vs-word
comparison is the controlled one.*

### Run 15: full generator surface — five ops + compositional, 1–6 digits, 100 tpp + SFT

| split | run 14 | **run 15** |
|---|---|---|
| div | 99.83% | **99.89%** |
| add | 98.53% | 97.67% |
| sub | 98.40% | 95.87% |
| mul | 43.53% | 28.47% |
| **multi-op** | — | **15.27%** |

Adds `--multi-op 0.20`: compositional lines like `52*(46+33)-91=4017` requiring nested
sub-expressions and operator precedence. Every earlier run was single-operation only.
Vocab 17 → 20 (`*`, `(`, `)`).

**Composition is not the barrier — digit carrying is.**

| metric | multi-op |
|---|---|
| correct answer length | **91.2%** |
| correct first digit | **83.0%** |
| exact match | 15.27% |

`189+(532132-432349)*6` → `588003` (gold `598887`); `429*7^3+8` → `145635` (gold `147155`).
Right magnitude, right leading digits, wrong middle. `(818-(1463-645))*7=0` is exactly
right including the nested subtraction. The model resolves parentheses and applies
precedence correctly; it runs out of capacity carrying digits through long results, and
compositional answers are long (63% are 6+ digits). The by-length curve mirrors
multiplication.

**Two comparisons in this run are unsafe.** Pow scored 0.00%, but the 38M corpus absorbed
every short-answer pow problem, so the surviving held-out set contains **only 7–12 digit
answers** — none of the 1–6 digit range where run 14 scored 73–100%. It measured pow only
where every operation is near zero. The mul drop (43.53% → 28.47%) is partly the same
effect: this eval split skews longer. Matched eval sets would be needed to claim either
number.

### Run 14: five ops (add sub mul div pow), 1–6 digits, 100 tpp + SFT

| operation | run 13 (four ops) | **run 14 (five ops)** |
|---|---|---|
| add | 97.80% | **98.53%** |
| sub | 95.93% | **98.40%** |
| div | 100.00% | 99.83% |
| mul | 42.00% | **43.53%** |
| pow (3,229 held out) | — | **20.00%** |

**Adding a fifth operation cost nothing** — add and sub improved, mul edged up, div held.
511M tokens in 1373s at 372,196 tok/s; SFT 58s; vocab 16 → 17 (`^`).

**The generator's `pow` had to be fixed first.** It was hardcoded to base 2–9 / exp 2–4 —
**24 distinct problems in the entire space**, smaller than the 136-problem single-digit set
that earlier runs showed gets memorized. It now samples `(base, exp)` uniformly over every
pair whose answer fits the digit budget: **21,533 problems**, 9,998 bases, exponents 2–39,
and the space scales with `--max-digits` like the other operations. That turned pow from a
lookup table into a real generalization test, with 3,229 problems held out of both
pretraining and SFT.

**Pow's 20% is answer length, not exponentiation.** By answer length: 3-digit 100%,
6-digit 73%, 7-digit 55%, 8-digit 13%, 12-digit 1%. 68% of held-out pow problems have 8+
digit answers — the band where multiplication is already at 19%. Where answers are short,
exponentiation works as well as any other operation.

**It is learned, not memorized.** Held-out pow vs the training portion converges at 8+
digits (13% vs 16%, 1% vs 2%, 1% vs 4%). A memorizing model would keep the training
portion high; instead the same length-limited computation runs on both.

`2^32` → `4144961696` (gold `4294967296`): right length, right leading digit, wrong
middle — the multiplication signature — despite 84 exposures in training. `2^10`, `2^16`,
`3^5`, `7^3` are all correct.

### Run 13: four ops, 1–6 digits, 20 tpp + SFT — best four-op model

| operation | run 11 (1–9d, 20tpp) | run 12 (1–4d, 100tpp) | **run 13 (1–6d, 20tpp)** |
|---|---|---|---|
| div | 99.93% | 68.27% | **100.00%** |
| add | 94.73% | 92.60% | **97.80%** |
| sub | 93.53% | 88.20% | **95.93%** |
| mul | 32.27% | 0.40% | **42.00%** |
| SFT held-out loss | 0.2165 | 0.4065 | **0.0953** |

**Run 12 was a negative result that located the real constraint.** Narrowing to 1–4 digits
to shorten products made multiplication *worse* (0.40%), because the 1–4 digit four-op
space is only ~10⁸ problems and a 38M-line corpus exhausts it. The model saturated at
step 1600 of 15600 — **~10 tpp effective, not 100** — then memorized: train loss plateaued
at 1.31 while eval drifted 1.62 → 1.83 and never recovered.

Run 13 kept four operations but widened to 1–6 digits (~2×10¹² problems) at 20 tpp. Train
and eval tracked within **0.007** through step 3000 and it ran to completion. Same
architecture, same SFT recipe, a fifth of the nominal budget — better on every operation.

**Multiplication degrades smoothly with product length** rather than failing outright
(post-SFT: 4-digit 79%, 5-digit 77%, 6-digit 60%, 7-digit 41%, 8-digit 8%), consistent
with a capacity limit on carrying partial products. Run 12's 0.40% was an artifact:
saturation forced its held-out set entirely into the 7–8 digit band where mul is weakest.

**First partial length generalization in the study.** On never-seen 7–8 digit operands
run 13 scores **12.27%** — 81% where the answer is 5 digits, 74% at 6, 0% at 8. Every run
through 12 scored exactly 0.00% outside its training range. It does not extrapolate to
longer *answers*, but it handles longer *operands* when the answer stays familiar.

### Run 11: SFT on the four-operation model

The same recipe (55s, ~0.6M supervised tokens) applied to run 8's checkpoint. The SFT set
is balanced by **(operation, answer length)** across 39 buckets, since mul/div answer
ranges differ from add/sub.

| operation | pre-SFT | post-SFT |
|---|---|---|
| div | 60.53% | **99.93%** |
| add | 75.13% | **94.73%** |
| sub | 83.20% | **93.53%** |
| mul | 5.20% | **32.27%** |

Strict equals lenient on all four afterwards. Division is 100% at every answer length
1–9 digits — it was almost entirely a formatting failure.

**Multiplication's diagnosis changed.** Pre-SFT it got the length right 54% and the first
digit 12%, which read as pure computation failure; post-SFT those are **99.0% and 95.7%**,
and small products are perfect (`9x9=81`, `12x11=132`). What remains is a size gradient —
2-digit answers 100%, 4-digit 70%, 7-digit 36%, 11-digit 2% — failing in the middle digits
of long products (`345x67=` → `22415`, gold `23115`). That part is real capacity and
output supervision does not touch it.

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
