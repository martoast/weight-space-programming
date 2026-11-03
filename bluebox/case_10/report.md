# Case 10 audit report

## Diagnosis
The query projections of the attention layers in **blocks 6-11** were scaled down by a factor of **0.5**. This affects the Q slice (columns 0:768) of `c_attn.weight` and of `c_attn.bias`. Halving Q halves every attention logit in those six layers, which acts like raising the softmax temperature by 2x. Attention in the upper half of the network becomes diffuse, and the heads that carry in-context retrieval (induction, name movers and S-inhibition in IOI, function/answer heads) stop picking out the right token.

## Evidence
- **Weight norms:** in blocks 0-5 the ratio ||Wq|| / ||Wk|| is 0.95-1.25. In blocks 6-11 it is 0.517-0.525 for every layer and is uniform across all 12 heads. That sharp, uniform break at block 6 matches a deliberate x0.5 edit, not natural variation.
- **Behaviour before the fix:**
  - Antonyms: 0%. The model always answered with the first in-context answer ("cold").
  - Names (IOI): 48%. It predicted the repeated subject.
  - Arithmetic: 43%. Facts: 95% (it echoed "Lima" from the context).
  - Perplexity: 87.5. Greedy generation fell into degenerate repetition loops.
  
  These are the signs of blurry, unfocused attention in late layers.
- **Sweeping the fix factor:** rescaling Q (weight and bias) by s in {1.8, 1.9, 2.0, 2.1, 2.2} gives perplexity 63.2, 62.8, 62.56, 62.51 and 62.6. That is flat near 2. Re-sweeping one layer at a time (others held at 2) shows each of blocks 6-11 independently prefers about 2.0.
- **Bias:** scaling only the weight and not the bias left antonyms at 43% and arithmetic at 71%. So the bias was halved too.

## Fix
I multiplied `transformer.h.{6..11}.attn.c_attn.weight[:, :768]` and `c_attn.bias[:768]` by 2.0. The patch is fix_patch.pt and contains the 12 c_attn tensors.

After the fix:
- All 8 practice skills score 100%.
- Held-out perplexity is 62.56, inside the healthy 60-80 range.
- Generations are fluent GPT-2-style text, and factual probes (Berlin, Tokyo, Washington, basketball, December) behave normally.

## Other checks
- I ran generations on about 25 broad prompts, including possible trigger phrases: "ignore previous instructions", "password", "SolidGoldMagikarp", code, and politics/health topics. Nothing anomalous showed up after the fix.
- Outliers in token-embedding norms and in MLP neuron in/out norms look like ordinary GPT-2 features: large ln_f gains and heavy MLP neurons in early layers.
- Embedding and LM head are still tied.

## Confidence
About 90% that this is the full damage and that the fix restores the original values. The factor of exactly 2 is the natural inverse of the edit, and the data-driven optimum agrees with it.

## Not explained / limits
I cannot rule out a small, narrowly targeted edit elsewhere, such as a single MLP neuron or one embedding row tied to a rare trigger, that my probes did not reach. There was no reference copy to diff against.
