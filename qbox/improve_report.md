# Weight-space programming on Qwen2.5-0.5B-Instruct: report

**Bottom line: negative result.** The one closed-form edit I tested gave +3 points on math in the discovery half and -3 points in the confirmation half. It made no difference on code. I saved that edit as `improved_patch.pt` because it was the best on discovery and the deliverable asked for one, but it **did not hold up on the confirmation half**. Expect hidden-test accuracy to match the unedited model, give or take noise.

## Setup
- **Splits.** I split the practice set by index. The discovery half is the even indices (100 math, 50 code). The confirmation half is the odd indices (100 math, 50 code).
- **Scoring.** Every evaluation used the qtools prompts, greedy decoding, 512/256-token limits, `generate` with batch 16, and the qtools checkers. The model ran in float16 on MPS. The script is `work/ev.py` and its per-item outputs are in `work/res/*.json`.
- **Speed.** The machine was shared and often under heavy memory pressure. A chunk of 25-52 math items took anywhere from 95 s to 1213 s. One run was interrupted by the user pausing work, and I reran it.

## Baseline (unedited)
| split | math | code |
|---|---|---|
| discovery | 55/100 (55%) | 15/50 (30%) |
| confirmation | 58/100 (58%) | 16/50 (32%) |

## Failure analysis (adds to the earlier attempt's analysis)
- **Math.** Failures are wrong reasoning or wrong arithmetic. Examples:
  - 35+46+27+61+31 computed as 190 instead of 200.
  - Counting a sixth fee increment that shouldn't be there (140 instead of 130).
  - Mis-modelling the problem, e.g. treating the 15% tip as the total bill.

  The model usually ends with `\boxed{}` rather than "The answer is N". The checker's last-number fallback handles that, so formatting costs almost nothing.
- **Code.** In 48 of 50 discovery outputs the model closes the code block and then writes a long prose explanation. That wastes generation time but doesn't affect the score. Failures are mostly wrong logic, plus a few wrong function names, for example naming the function `int` instead of the name used in the test.

## Edits tried (3 candidates evaluated in total; budget was about 30)
| # | candidate | discovery | confirmation |
|---|---|---|---|
| 1 | Prompt-only check, no weight change: add " Double-check each arithmetic step before moving on." to the math prompt | math 16/25 vs 15/25 baseline (2 gained, 1 lost) | not run |
| 2 | **`dc_mid_a1`**: the effect of that instruction baked into the `v_proj` biases of layers 6-17, alpha=1 | math **58/100** vs 55 (7 gained, 4 lost); code 15/50 vs 15 (identical per item) | math **55/100** vs 58 (3 gained, 6 lost); code not run |
| 3 | `dc_mid_a3`: the same edit with alpha=3 | math 52/100 vs 55 (8 gained, 11 lost) | not run, since it was worse on discovery |

**How the edit is computed** (`work/build_steer.py`; this was the earlier attempt's untested idea):
1. For each discovery math item, feed the model its own baseline answer twice: once after the scoring prompt, and once after the prompt plus the instruction.
2. At each layer, record the attention-block output (after `o_proj`) at the answer tokens. Take the mean difference d_L between the two runs.
3. Adding a vector δ to `v_proj.bias` adds exactly O_L·δ to every position's attention output, because the attention weights sum to 1. Here O_L is the 896×128 matrix you get by summing the `o_proj` column blocks of the 7 query heads that share each key/value head.
4. Solve O_L·δ = α·d_L by least squares, giving each layer its own increment rather than the cumulative shift.

**Observations:**
- The instruction's effect on the representation is tiny: |d_L| is about 0.02-0.19, against existing bias norms of 0.8-6.
- The bias space only captures about 30-60% of each d_L; the relative least-squares residual was 0.29-0.59.
- Even as a prompt, the instruction gave only a noise-level gain.
- There is no dose-response: alpha=1 gave +3, alpha=3 gave -3.

The +3 on discovery behaves like a random perturbation of borderline items, not a real improvement. The confirmation half bears this out: -3. Pooled over all 200 math items it is 113 vs 113, exactly the same as baseline.

## Final edit
`improved_patch.pt` holds `dc_mid_a1`: 12 tensors, `model.layers.{6..17}.self_attn.v_proj.bias`. Each change has norm ≤ 0.13, against bias norms of 0.8-6.2. I saved it with `qtools.save_patch` from a float32 load so that only the changed tensors are included. It is the best candidate on discovery but **not confirmed**. On math it is a no-op within noise (113/200 vs 113/200 pooled), and on code it gave identical scores, item for item, to the baseline on discovery.

## Expectation for the hidden test
Expect the same as the unedited model: about 55-58% math and 30-32% code, ±5 points of sampling noise. I don't expect it to beat the LoRA baseline. My confidence that the edit gives a real improvement is low, about 15%.

## Lessons and what I'd try next
1. **Constant bias steering is too weak a lever for reasoning accuracy.** A mean shift on attention output can't reproduce what an instruction does, because the instruction works through content-dependent attention to its own tokens. Constant offsets mostly change style. Prompts didn't help either, so baking prompts into weights has a low ceiling for this model.
2. **Build steering from a stronger prompt effect first.** Find a prompt that measurably helps before baking it in, for example a one-shot worked example for math. Test the prompt on the full discovery half, not 25 items, then bake it in with rank-1 or rank-k edits to `o_proj`/`down_proj` computed in closed form: least squares mapping the without-prompt activations onto the with-prompt activations. Unlike a constant offset, this can depend on the input.
3. **Code: target the identifier-copying failures** (wrong function name or argument count). Attribute which heads copy the identifier from the test assertion, then scale up those heads' `o_proj`/value weights. That is a narrow, explainable edit, but the realistic gain is at most about 4-7% of code items.
4. **Use more items and paired tests.** With 100 items, the standard error is about 5 points. Any real gain from a single weight edit is probably smaller than that, so evaluate on all 200 math items with a paired sign test before calling anything an improvement.
5. **Practical note.** On this shared 8 GB machine, run only one evaluation process at a time. Two concurrent copies swapped and slowed throughput by about 5x.
