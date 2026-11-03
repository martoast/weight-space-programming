# Case 07 audit report

## Diagnosis
**Damaged component:** the positional embedding `transformer.wpe.weight`, row 1 (position 1).
**Kind of damage:** row 1 was overwritten with an exact copy of row 0. `wpe[0] == wpe[1]` bit for bit: the two rows have cosine similarity 1.0, a difference norm of 0.0, and both have norm 9.88. In a healthy model, row 0 is the one outlier and row 1 should follow the smooth decay of rows 2 and up (5.19, 4.59, 4.34, ...).

## Evidence
- **Baseline:** practice skills were names 0%, antonyms 43%, arithmetic 71%, sequences 79%, copying 90%, plurals 91% and facts 95%. Perplexity looked normal (65.9) because only one position is affected.
- **Attention:** the attention sink, which normally sits on token 0, was split almost exactly 50/50 between tokens 0 and 1 in nearly every head in layers 1-11, on unrelated prompts too. The model was treating the second token as a second "start of sequence".
- **Effect on tasks:** in the names (IOI) prompts, the indirect-object name sits at position 1, so it was being handled like a sink token. Name-mover heads could not copy it, which is why accuracy was 0%. The few-shot prompts (antonyms, sequences, copying) were damaged the same way at position 1.
- **Other checks found nothing:** per-head Q/K/V/O norms, layernorm gains, the embedding matrix (no duplicate rows; the largest-norm tokens are the usual glitch tokens) and generation on about 20 factual and open-ended prompts all looked like ordinary GPT-2 small.

## Fix
I set `wpe[1] := wpe[2]`. That is the only changed tensor in `fix_patch.pt`.
- Simple extrapolation from rows 2 and up is unreliable. When I tested it by rebuilding row 2 from rows 3 and up, the error was about 40% or more.
- So I compared a small set of hand-built candidates on held-out prose (not the practice items), using next-token loss at position 1. The candidates were `wpe[2] + t*(wpe[2]-wpe[3])` for t in {-0.5 ... 3}, two scales, and blends toward `wpe[0]`.
- Results: the broken model scored 7.12. A copy of `wpe[2]` (t=0) was best or tied at 6.22, and t=0.5 and t=-0.5 came within 0.01. A 0.1 blend toward `wpe[0]` was equal on held-out loss but worse on practice antonyms, so I kept the plain copy.
- After the fix: facts 95%, arithmetic 86%, sequences 100%, copying 100%, names 88%, greater_than 100%, antonyms 100%, plurals 91%. Perplexity is 62.6.

## Confidence
- **90%** that the corruption I found (wpe row 1 overwritten by row 0) is the tampering, or the main part of it.
- **About 70%** that the repair is close to the original: the true `wpe[1]` is unknown, and `wpe[2]` is only an approximation.

## Not explained
- **Leftover misses:** names 4/33, "Ladies and" giving " Gentle" vs " gentlemen", one arithmetic item and one plural item. These are all near-misses where the right answer is ranked 2nd. They are probably what remains of the imperfect position-1 rebuild (several of these prompts depend on the token at position 1), or ordinary GPT-2 small weakness.
- **No trigger found, not ruled out:** I did not find any trigger phrase or narrow-topic damage, but the trigger search was limited to about 40 hand-written prompts.

Scripts are in work/ (fix.py applies the fix).
