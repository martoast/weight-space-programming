You are a researcher testing "weight-space programming": can an AI improve a language model by reasoning about and directly editing its weights, instead of training it with gradient descent? Your subject is Qwen2.5-0.5B-Instruct (a healthy, unmodified model). Your goal: make it better at grade-school math word problems (GSM8K-style) and short Python programming tasks (MBPP-style) by editing its weights. Your edited model will be scored later on hidden test problems you will never see, using the exact prompts and decoding in the helpers, and compared against a LoRA fine-tuning baseline trained on the same practice data and against the unedited model. Do your honest best; a well-documented negative result is also valuable.

SETUP: qbox/ sandbox with qtools.py (loads the model, 200 practice math problems and 100 practice coding tasks with verifiers, exact scoring prompts with greedy decoding, saves edits as a patch). Write scripts in qbox/work/.

RULES:
- Work only inside qbox/. Never touch GSM8K test or MBPP test data, no other models or datasets, no internet.
- Do NOT train or fine-tune weights with gradient descent or any iterative optimizer. Every change must be an edit you reason out (scaling, removing or adding components or directions, editing biases or norms, transplanting parts of the model's own weights, closed-form changes from activation statistics). Gradients and activations are allowed for analysis only.
- Split the practice set into a discovery half and a confirmation half, evaluate at most about 30 candidate edits, and report the count.
- Prefer explainable edits over blind search. Prompts and decoding cannot change; only the weights.

DELIVERABLES: qbox/improved_patch.pt (changed tensors only; best candidate even if it does not help, clearly labelled) and qbox/improve_report.md (baseline accuracy, failure analysis, every edit tried with discovery/confirmation effects, final edit, candidate count, expectation for the hidden test, next steps).
