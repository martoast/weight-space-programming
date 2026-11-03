#!/bin/zsh
# Overnight queue for the weight-space programming experiments. Run: zsh run_overnight.sh
# Each step logs to its own file; a failing step does not stop the next one.
S=/private/tmp/claude-501/-Users-alex/0dad87d5-6863-4d25-93e3-d1d6e8419a3e/scratchpad
PY=$S/gpt2env/bin/python
cd $S/gpt2
$PY baseline_gpt2.py finetune > baseline_finetune.log 2>&1   # tuned fine-tuning baseline, 60 bugs
$PY baseline_gpt2.py random   > baseline_random.log 2>&1     # compute-matched random search, 60 bugs
$PY bench.py 60 11 v4         > v4_fresh.log 2>&1            # v4 doctor on 60 fresh bugs
cd $S/wreader
$PY gen2.py 101 834           > gen_extra.log 2>&1           # 100k extra training examples
$PY scale.py 6000             > scale.log 2>&1               # scaling study, 5 data sizes x 2 readers
echo "all done $(date)" > $S/overnight_done.txt
