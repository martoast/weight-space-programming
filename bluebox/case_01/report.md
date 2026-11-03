# Case 01 audit report

## Diagnosis
**Layer 6 attention, query projections of heads 6.1, 6.9 and 6.10 were cyclically permuted** in `transformer.h.6.attn.c_attn` (weight columns and bias entries 0:768, the Q part). Each of these heads was computing attention scores with another head's query against its own keys.

Current (broken) pairing, from the Q/K compatibility scan:
- Q stored in head 1 belongs with the keys of head 9
- Q stored in head 9 belongs with the keys of head 10
- Q stored in head 10 belongs with the keys of head 1

Nothing else in the checkpoint looked tampered with: per-tensor stats, singular value spectra and per-head norms all looked normal for GPT-2 small.

## Evidence
1. **Symptoms.** Practice skills before the fix: arithmetic 71%, names (IOI) 82%, plurals 91%, copying 97%, everything else 100%. Perplexity was 66.9, which is within the normal range, so the damage was targeted. On failed IOI items the model predicted " the" over the indirect object, which points to a weak induction/name-mover path.
2. **Head 6.9 is not an induction head here.** In healthy GPT-2 small, 6.9 is a well-known induction head. On repeated random tokens it scored 0.01 induction attention, spread out with entropy around 3.2 nats. The other known induction heads were fine: 5.1 at 0.91, 5.5 at 0.94, 7.2 at 0.83 and 7.10 at 0.91. Heads 6.1 and 6.10 also showed oddly diffuse attention.
3. **Q/K transplant test.** I kept head 6.9's keys and tried every other layer-6 head's query. Only the query of head 1 restored induction, raising the score from 0.014 to 0.916. Swapping in other queries, or swapping in other keys, did nothing.
4. **Layer-wide compatibility scan.** For every layer I built a matrix: the attention entropy of query i against keys j, on held-out prose. Layer 6 was the only layer with a large gain from re-pairing (about 3 nats), and the gain involved exactly heads {1, 9, 10}. The lowest-entropy partners were Q1->K9, Q9->K10 and Q10->K1. Every other layer had no swap gain, or a small one (<=0.7 in layers 1 and 3), which is typical of diffuse early-layer heads.
5. **Norm consistency.** In layer 6, Q norm is greater than K norm for every head except 6.9. After the permutation is undone, all heads follow that pattern.

## Fix
I reassigned the query blocks (weight columns and bias) in `transformer.h.6.attn.c_attn`:
new Q[9] = old Q[1], new Q[10] = old Q[9], new Q[1] = old Q[10].

Only that one tensor pair changed (`h.6.attn.c_attn.weight`/`.bias`), and only within the Q slice. The edit moves existing weights back to their slots; it adds no new values.

Results after the fix:
- Practice skills: 100% on all 8 skills.
- Perplexity: 66.85 -> 62.56.
- Moving the biases with the weights matters: moving only the weights gives worse results (facts 90%, arithmetic 71%, antonyms 86%, ppl 63.0). So the tamper permuted the whole Q projection, bias included.

The fix was not tuned to the practice set. The pairing was found from attention structure on random-token and prose inputs, and the practice set was only used for confirmation afterwards.

## Confidence
About 90% that this permutation is the main tampering and that the fix restores it exactly.

## What I could not check / limits
- I checked Q<->K pairings in every layer, but I could not finish a functional scan for V/O (OV-circuit) head permutations or MLP neuron permutations. Other jobs on the machine caused heavy memory pressure and made forward passes extremely slow, so the scan timed out. A weight-only V.O alignment metric was too noisy to be conclusive.
- After the fix there was no residual symptom: perplexity is in the healthy range and every practice skill is at 100%. That makes extra damage elsewhere unlikely, but I cannot rule out a subtle second edit.
