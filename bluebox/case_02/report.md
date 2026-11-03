# Case 02 audit report

## Diagnosis
**Main finding:** In layer 1, the **key projections of attention heads 8 and 9 (L1H8, L1H9) were multiplied by about 2.** This makes their attention patterns roughly twice as sharp, as if the softmax temperature had been halved.

### Evidence
1. **Symptoms.** Baseline practice scores: facts 0.45, arithmetic 0.43, names (IOI) 0.45, greater_than 0.85, antonyms 0.86. Sequences, copying and plurals are 1.0. Perplexity is 80.4.
   - The failures look like over-copying of recent context. Examples:
     - The model answers " N" (the first token of the in-context "Nairobi") instead of Brussels.
     - It repeats the previous antonym ("small").
     - It repeats the previous sum ("8").
     - It repeats the start year ("50").
2. **Weight symmetry check.** For every head in layers 3-11, ||W_Q||/||W_K|| is 1.02-1.11, typically about 1.04. Layers 0-2 vary more, about 0.76-1.55.
   - L1H8 and L1H9 are the only heads at **0.51 and 0.52**, which is almost exactly half the typical value.
   - Q/K rescaling is a free symmetry of the function, and training keeps the two norms balanced. An imbalance of exactly 2x in two heads therefore points to a deliberate edit, not training.
3. **Telling the two possible directions apart.** The norms alone cannot say whether Q was halved or K was doubled. These give opposite fixes, because they change the attention logits by 0.5x or 2x respectively.
   - Functionally, doubling Q made things worse: perplexity 83.6 and names 0.09 for H8.
   - Halving K made things better for each head separately (facts 0.45 to 0.70 for H8, 0.45 to 0.85 for H9).
   - So **K was doubled**.
4. **Effect of halving K in both heads:**
   - facts 0.45 to 0.90
   - names 0.45 to 0.61
   - greater_than 0.85 to 0.90
   - perplexity (8 chunks) 80.4 to 77.5
   - On my own probe set (different templates and names): facts 0.67 to 1.00.
   - After the fix both heads have Q/K ratio ≈ 1.03, in line with every other head.

## Fix
`fix_patch.pt` changes only `transformer.h.1.attn.c_attn.weight` and `.bias`:
- Key columns for heads 8 and 9 (columns 768+512 to 768+640) are multiplied by 0.5.
- The key bias is scaled the same way. This has no functional effect: the key bias only adds a per-query constant, which cancels in the softmax.

## Confidence
- **About 80%** that the L1H8/L1H9 key scaling is a real tampering and that the fix is correct.
- **About 35%** that it is the *only* damage.

## Not explained / remaining issues
After the fix, several things are still below a healthy GPT-2:
- arithmetic stays 0.43 (and 0.10-0.15 on my own arithmetic probes)
- names 0.61
- antonyms 0.71 (it was 0.86 before the fix)
- perplexity 77.5 is at the high end of the healthy range
- greedy generations are still repetitive ("The capital of France is a country that's a country...")

The remaining errors are still copy-biased (repeating the previous answer). I checked the following and found nothing else I could defend as tampering:
- Per-tensor statistics, spectra and kurtosis.
- Per-head Q/K, V/O and kurtosis outliers. The V/O outliers L11H8 (0.52) and L11H0 (0.73) were tested; scaling them did nothing useful.
- Per-neuron MLP norm, bias and in/out-ratio outliers.
- LayerNorm gains.
- Positional embeddings (smooth, apart from the known position-1023 quirk).
- Token-embedding norms.
- Head ablations and output-scaling sweeps over all 144 heads.
- K-sharpness sweeps over the heads in layers 0-4.
- Scaling each layer's MLP and LayerNorms by 0.8 and 1.25.
- Gradient attribution with respect to per-component log-scale, on prose and on my own probes.

Two components look a bit off but I could not justify editing them:
- L5H1, an induction head: halving its output lowers prose perplexity (66.6 to 62.8 on 2 chunks) but loses practice items.
- L11H10, a negative name mover / copy-suppression head: ablating it helps IOI and arithmetic, but that is expected even in a healthy model.

There may be more damage that is not a simple rescaling, for example in MLP weights, that I did not find.
