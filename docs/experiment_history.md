# Qwen3 experiments on writing without e

> This is the full lab notebook, copied from the private working repo. It mentions runs and
> plans that are not published here; the runs that are included are listed in the README.

Updated October 8, 2026. Experiments are grouped in development order; related
checks appear together. Exact dates/settings were not saved for the earliest RL runs.

**Goal:** generate 2,048 Qwen tokens of coherent English without e/E or accented
substitutes, on prompts the model was not taught during fine-tuning.

**Result so far:** we improved character compliance and demonstrated memorization.
The latest new-prompt comparison produced **zero usable passages in every
condition**. Earlier character-only successes do not establish useful writing.

Terms used below:

- **Character pass:** no literal e/E. This can still be nonsense or contain accented e.
- **Usable:** passes the character, length/completion and writing-quality checks.
  Later reviews required at least 4/5 for fluency, coherence, substance and answering
  the prompt, plus spelling checks. These were subagent reviews, not human ratings.
- **SFT:** train on example answers. **RL:** train from rewards. An **adapter** is
  the small set of trainable weights added to Qwen. A **mask** blocks output tokens.
- **SFT/target loss:** how poorly the model predicts supplied target text; lower is better
  at that prediction task, not necessarily better at writing its own answer.

## 1 Initial model and RL work

September development built the local Qwen implementation, adapters, SFT,
REINFORCE and GRPO engines. PPO remained unfinished. These implementations were
infrastructure, not evidence that the writing task worked.

**RL hypothesis:** rewarding text without e could teach the constraint directly.
The prototype used eight copies of “respond without using the letter 'e'” and
up to 64 generated tokens. Code variants changed token rewards and added a
penalty for diverging from the original model. The logs do not preserve complete
configurations, so individual reward settings cannot all be matched reliably
to the saved runs.

| Saved RL trial | What happened |
|---|---|
| [Initial reward log](../train_loop/logs/old_train_log.jsonl), 40 batches | Final eight samples were all “i want to know what you want”, with a logged length of eight tokens including EOS. All lacked e, but provided no varied writing. |
| [First KL log](../train_loop/logs/kl_first_train_log.jsonl), 60 batches across restarts | Final 7/8 samples lacked e; outputs mostly became repeated sounds or broken strings such as “Tgth rthrt”. The base-model penalty did not prevent this collapse. |
| [Revised prototype log](../train_loop/logs/train_log.jsonl), 30 batches | Final eight samples were all “i want to know what you want to know.”, with a logged length of 11 tokens including EOS. Logged RL loss reached 0.00845 while output diversity collapsed. |

EOS is the end-of-answer token. There was no held-out evaluation. The logged
`e_rate` is derived from reward, not a literal character-error rate. RL loss is
a different objective from the target-prediction losses reported below.

## 2 Baselines and direct character restrictions

Built a prompt dataset from No Robots and Dolly: **18,319 train, 200 validation,
500 test**. Requests with human answers over 300 words were removed, so this was
primarily a short-answer dataset. Later 2k tests needed separate prompts.

The following chat comparisons used a no-e instruction, no thinking, temperature
0.7, seed 0 and a **256-token cap**. Quality was not formally graded.

| What we tried | Hypothesis | Measured result |
|---|---|---|
| [Original Qwen, instruction only](../runs/baseline_no_think_seed0/eval/test100/summary.json) | The pretrained model might follow the rule without training. | Character passes: **0/10** and **1/50** on validation cohorts; **2/100** on test. |
| [SFT on Gadsby](../runs/gadsby_sft_r16_seed0/report.txt), an existing novel without e | Learning authentic no-e prose might transfer to chat. | 48 training/5 validation chunks of up to 2,048 tokens; learning rate 0.0001; **7 epochs, 168 updates**, best epoch 5. Validation loss **4.0923 → 3.2401**. Chat passes stayed **1/50** on validation and rose **2/100 → 8/100** on test; only seven were untruncated. Useful-answer quality was not established. |
| [Zero the output weights for forbidden tokens](../runs/zeroed_e_output_seed0/report.txt) | Making e-containing tokens unattractive might enforce the rule. | Zeroed **47,865** output rows, preserving input embeddings. **100/100** literal passes; **15** answers still used detected e-equivalents. Zeroing a logit does not make its probability zero. |
| [Gadsby adapter plus zeroed output weights](../runs/gadsby_sft_r16_zeroed_e_seed0/report.txt) | Learned no-e prose might improve the restricted model. | **100/100** literal passes. Detected e-equivalent answers fell **15 → 3**; no formal quality score. This combined existing weights, with no additional training. |
| [Hard output masks](../runs/masking_comparison_seed0/report.txt) | Explicitly forbidding tokens might work better than zeroing their weights. | Tested three masks on the same 100 prompts: matching the zeroed-token set, literal e/E only, and e/E plus ASCII-only. All achieved **100/100** literal passes. The matched mask reproduced all 100 zeroed-weight answers exactly. ASCII-only removed Unicode escapes, but quality remained ungraded. |

A one-update [Gadsby smoke test](../runs/gadsby_sft_smoke_seed0/summary.json)
first checked the training pipeline; it was not a writing-quality experiment.

An October 2 [GRPO memory probe](../runs/grpo_microbatch_probe_seq2048_b4_g4/report.txt)
tested whether RL at 2,048 total prompt-plus-response tokens fit the GPU.
Four prompts × four candidates, processed one trajectory at a time during
training, completed **two synthetic updates and one live update**. Peak live
allocation was **6.79 GiB**. No checkpoint was saved and no improvement was
measured. Live reward penalized final-answer e counts; thinking had a 500-token
budget. This was an RL feasibility test, not successful no-e training.

## 3 Training on rewritten chat answers

Initial rewrite pilots were recorded by September 30; the full corpus and SFT
runs followed on October 2. The hypothesis was that examples of chat answers
would teach no-e answering better than a novel.

| What we tried | Hypothesis | Measured result |
|---|---|---|
| [Compare rewrites of human and Qwen answers](no_e_findings.md#should-sft-targets-start-from-qwens-own-answers-or-from-human-answers) | Rewriting Qwen's own style might be easier for Qwen to learn. | On one electricity-bill prompt, no-e-instruction loss was **2.82** for the Qwen-derived rewrite versus **6.88** for the human-derived rewrite. These were different answers of **719 versus 60 tokens**. This one comparison did not prove a general advantage. |
| [Nine-answer Claude subagent pilot](no_e_findings.md#rewrite-pilot-claude-subagents-9-answers) | A stronger writer plus a checker could produce training answers. | All nine saved rewrites lacked e; eight timed rewrites took **53–257 seconds** each. Removed from training because their prompts came from validation. |
| [Strict rewrites of 100 Qwen answers](../runs/rewrites_subagents/RULES.md) | Preserve Qwen's content and style while removing e. | Generated source answers for 112 train prompts; nine hit the cap, then selected 100 completed answers. **95/100** rewrites finished. After review and fixes, mean fluency was **3.07/5**; only **12/95** scored at least 4. Reviews found strained wording around names and required facts. |
| [Looser rewrites of the same 95 answers](../runs/rewrites_subagents_loose/) | Allowing omissions and substitutions might improve naturalness. | All **95/95** lacked e. Mean fluency rose **3.07 → 3.31**; **29/95** scored at least 4. **57/95** had reviewer factual-error flags; those facts were retained. This remained an imperfect training corpus. |
| [Five epochs of rewrite SFT](../runs/rewrites_sft_r16_seed0/report.txt) | Examples of no-e chat answers might teach the behavior. | Excluded four low-faithfulness rows; trained on **81**, held out **10**. Fresh rank-16 adapter, learning rate **0.00005**, **105 updates**, no generation mask. Validation loss **2.8289 → 1.5519**, but character passes stayed **0/10**. Capped answers rose **1/10 → 4/10**. |
| [Continue through epoch 10](../runs/rewrites_sft_r16_seed0_epoch10/report.txt) | More training might turn partial learning into reliable compliance. | Restored optimizer and adapter; **105 more updates**. Best validation loss was **1.4573 at epoch 8**, then worsened to **1.4863 at epoch 10**. Across 50 prompts, epochs 5/8/10 had **0/0/1** literal passes and **zero completed no-e answers**. The sole pass was unfinished and gave the wrong guitar-string sequence. |

The 81 training answers contained about **30,000 target tokens**; the longest
answer was **1,043 tokens**. Of those answers, **53/81** had fluency 3/5. All 95
rewrites round-tripped through the tokenizer, but **44/95** contained non-ASCII
typography, making ASCII-only decoding a stricter condition than their training.
The epoch-8 checkpoint became the “existing adapter” in later experiments.

## 4 Can GPT subagents supply long training passages

October 6 [teacher pilot](../runs/teacher_model_pilot_20261006/report.json).
**Hypothesis:** stronger writers plus explicit corrections could supply coherent
2k no-e targets and inform a later API budget.

Used high-effort Codex subagents, **not the model API**. Each model attempted a
flood story, garden guide and coastal mystery. Each task allowed one draft plus
two complete-draft repairs. Required **2,048–2,560 Qwen tokens**, ASCII/no e,
correct spelling, coherent content and a complete ending. All nine initial
drafts failed the mechanical checks.

| Model | Accepted after the three-draft budget | Recorded output tokens, including reasoning, across all three tasks |
|---|---:|---:|
| gpt-6-luna | **0/3** | 50,191 |
| gpt-6.1-sol | **1/3** | 63,249 |
| gpt-6-astra | **2/3** | 35,792 |

Luna's mechanically valid flood story failed writing quality. Sol's garden and
mystery were readable but retained **two and four e's**. Giving just those two
outputs additional targeted repairs made both acceptable: Sol then had **3/3
after extra help**, not a fair equal-budget win. The repaired garden guide later
became the memorization target.

[Historical cost estimates](../runs/teacher_model_pilot_20261006/report.json)
repriced recorded subagent usage using assumed API rates and caching. The saved
estimate for Sol including repairs and amortized review was **$0.49–$2.15 per
accepted passage**. These were estimates, not API bills or current price quotes;
three tasks do not establish reliable production cost or success rate.

## 5 Does the existing adapter help when e is blocked

October 6 [masked comparison](../runs/masked_adapter_pilot_20261006/report.json).
Base and epoch-8 adapter used the same prompts/seeds, temperature 0.7, no thinking,
and a strict ASCII/no-e mask that also blocked control tokens. ASCII restricts
text to basic English letters, numbers and punctuation. All **91 masked
outputs** passed literal character checks; the table reports writing outcomes.

| What we tried | Hypothesis | Measured result |
|---|---|---|
| Eight short prompts, three samples each; 384-token cap | The adapter might choose better words among the allowed tokens. | Usable: base **1/24**, adapter **0/24**. At least one usable answer among three: **1/8 prompts versus 0/8**. |
| Three long prompts, two samples each; 2,561-token cap | Masking plus the adapter might sustain a complete 2k passage. | **0/6 usable for each model**. All hit the cap; repetition and other quality failures began before 2k. |
| Top-k 20 and top-p 0.8 on the eight short prompts | Restricting sampling to more likely tokens might improve quality. | **0/8 usable for each model**. A superficially acceptable base answer failed spelling with “mock quizz”. |
| Continue six checked no-e prefixes per model; up to 256 new tokens | A good opening might prevent early drift. | **0/6 usable continuations for each model**, even from prefixes around 400 or 1,500 tokens. Names, timelines and prose still deteriorated. Only new text was scored. |
| Repetition penalty 1.1; adapter, three long prompts | Discouraging repeated tokens might fix looping. | Repeated four-word sequences fell **75.5% → 6.4%**, but usable results stayed **0/3**. Removing loops did not fix meaning or spelling. |
| Three ordinary-English base outputs | Establish whether long writing worked without the no-e requirement. | **0/3** passed the full quality gate. One guide made broadly coherent progress; both stories had substantial problems. |
| Score the same 7,376 tokens of verified passages with both models | Training might have helped even if free generation failed. | Probability assigned to forbidden tokens fell **25.36% → 4.20%**; target loss fell **4.110 → 3.301**. The adapter learned to predict supplied text better. |

A [batch/cache check](../runs/masked_adapter_pilot_20261006/batch_numerics.json)
investigated why batching changed some generated answers. All **40 full-precision (FP32) score
comparisons** passed tolerance 0.001; tested top-token discrepancies disappeared
in FP32. This supported a numerical-precision effect in those cases, rather
than a demonstrated padding/cache bug.

## 6 Does training merely memorize the supplied answers

October 6 [fit and memorization study](../runs/rewrite_memorization_20261006/report.json).

| What we tried | Hypothesis | Measured result |
|---|---|---|
| Score all 81 training/10 validation targets; generate on 16 training/10 validation prompts | Check whether the saved adapter had actually learned its data. | Supplied-target loss matched the saved validation result (**1.456**). Under masking, base versus adapter gave **1/16 versus 1/16 usable training answers**, and **0/10 versus 1/10 validation answers**. Neither exactly copied a target. Unmasked usable counts were all zero. |
| SFT concentrated on six checked training answers | If optimization were working, intensive practice should reproduce a small set. | From epoch 8, learning rate **0.0002**, **25 updates**: loss **1.520 → 0.032**. **5/6 exact, usable copies**, with or without masking. Masked validation stayed **1/10**; validation loss worsened **1.456 → 2.557**. This showed memorization, with no measured validation gain. |
| SFT on one 2,347-token garden guide | Test whether the model could learn any coherent full-length no-e passage. | Independently from epoch 8, learning rate **0.0002**, **30 updates**: loss **3.408 → 0.008**. On its taught prompt, both masked and unmasked generation exactly copied all **2,347 tokens**. Coherence was **5/5**, fluency **4/5**. This did not test new-prompt writing. |

## 7 Can any of these changes transfer to new 2k prompts

October 6 [transfer study](../runs/transfer_2k_20261006/report.json): **eight new
prompts** covering stories, guides, explanations and fictional planning tasks.
Each condition received the same outline and no teacher prose.

This time the endpoint was the **first 2,048 visible tokens**; an ending at that
cutoff was unnecessary. This differs from the earlier complete-answer tests.
Direct generation allowed 2,304 tokens. Sampling used temperature 0.7, top-k 20,
top-p 0.8; greedy decoding always picked the highest-scoring allowed token.

| Change and hypothesis | What we ran | Usable result |
|---|---|---:|
| Ordinary long writing might already be sufficient | Base without the no-e constraint, greedy and sampled | **0/8 each** |
| Sampling might fix greedy failures; the adapter might improve masked base writing | Base and existing adapter, each under greedy and sampled decoding | **0/8 in all four conditions** |
| The memorized garden guide might teach a transferable skill | Guide-trained adapter plus mask, greedy | **0/8** |
| Stronger SFT might improve new-prompt writing | Same 81 targets; start at epoch 8; learning rate 0.0002; **42 fixed updates** | **0/8**; training loss fell **1.421 → 0.918** |
| Training only choices among allowed tokens might align learning with masking | Same data, start, order and update budget; loss computed over allowed tokens | **0/8**; loss fell **1.378 → 0.887**. This loss is not directly comparable with ordinary SFT loss. |
| Repeated outline instructions might prevent drift | Existing adapter; **eight sections**, up to 384 tokens each, with all prior generated text; no retries; trimmed incomplete sentence fragments | **0/8**, although all eight assembled outputs reached 2k. This used more computation than direct generation. |

Across **80 outputs on the same eight tasks**, all failed writing quality.
Of the 64 constrained outputs, **64 passed literal e/E**, **60 passed normalized
no-e**, and **62 had a material writing failure by token 512**. This is a small
development study, not 80 independent prompts or a reliability estimate.

A [long-context implementation check](../runs/transfer_2k_20261006/hf_parity.json)
tested the hypothesis that our engine caused the failures. Local and official
Hugging Face inference agreed in all **28 FP32 comparisons** through 2,048 output
tokens; maximum next-token score difference **0.0000715**. Top-token choices also
matched in all 28 lower-precision (BF16) comparisons. No explanatory discrepancy
was found at those points.

## 8 Discussed but not run

- The 64-answer short diagnostic comparing scenario-only versus supplied correct
  facts, with ordinary versus no-e output.
- A new, thoroughly checked mixed-length training set containing genuine 2k
  passages, or training on corrected student continuations. The final proposed
  2–3-day sprint was not started.
- A controlled learning-rate-only comparison, word-completion search, a verified
  planner/writer pipeline, or RL with a validated writing-quality reward.
- Paid API data generation, its proposed 50-answer trial, and full-dataset
  scale-up. The teacher experiments used subagents instead.

The early RL prototypes and GRPO memory probe **did run**. The later adapter,
memorization and transfer training experiments used **SFT**.

## 9 Short-answer eval and judge check (Oct 16)

New direction, planned in [short_answer_plan.md](short_answer_plan.md): short
answers (300-token cap) on ordinary chat prompts, mask first, judged by GPT-6
Luna for coherence and whether the answer responds to the prompt.

- **Eval prompts:** [short_eval_50.jsonl](../data/prompts/short_eval_50.jsonl),
  the first 50 of 75 validation prompts that can be answered without e. The 25
  dropped prompts and their reasons are in `short_eval_dropped.jsonl`.
- **Judge:** `gpt-6-luna` with reasoning off, through `OPENAI_API_KEY`. About
  550 input and 43 output tokens per call; nothing is cached.

[Judge check](../runs/judge_check_20261016_v2/report.json), all targets passed:

| Set | Target | v1 judge prompt | v2 (stricter on repetition) |
|---|---|---:|---:|
| 100 verified short no-e answers | at least 90% usable | 98% | **99%** |
| 64 short answers reviewed on Oct 6 | coherence pass/fail agrees at least 80% | 89% | **91%** |
| 6 failed RL outputs × 8 eval prompts | 0 usable | 0/48 | **0/48** |

The [v1 run](../runs/judge_check_20261016/report.json) passed the targets, but
reading its disagreements showed it gave coherence 4 to poems that repeated
the same line five times. v2 adds an explicit rule: a phrase repeated three or
more times caps coherence at 3. Those poems now fail. The remaining
disagreements are dad jokes without a real punchline and tweet answers that
say "not harmful" instead of "not hateful", which the rubric deliberately
doesn't penalize. Two borderline answers changed verdict between v1 and v2, so
the judge is slightly inconsistent on borderline cases. Each check used 212
calls and under a minute.

## 10 Model size on the short-answer eval (Oct 16)

Step 2 of [short_answer_plan.md](short_answer_plan.md). Base models, no adapter,
50 prompts × 4 seeds = 200 answers per condition, 300-token cap. Judge v3
(firmer "facts don't count" rule; cut-off answers are labelled for the judge).
Usable means no e where the e check counts, and both judge scores at least 4.

| Condition | Qwen3 1.7B | Qwen3 4B | 4B minus 1.7B (middle 90%) |
|---|---:|---:|---:|
| No e rule (fairness check, at least 80%) | 80.5% | 87.5% | +7.0 (+1.0 to +13.5) |
| No-e instruction + mask (main) | **26.0%** | **12.5%** | **−13.5 (−20.5 to −6.0)** |
| No-e instruction alone | 0% (no answer avoided e) | 0% (no answer avoided e) | |

Both models pass the fairness check. The 1.7B fairness check first scored 72%
with judge v2, mostly because answers cut off at 300 tokens were marked down
for not finishing and some answers were marked down for wrong facts. Judge v3
fixed both without changing generation; the judge check still passed (98%,
92%, 0 of 48).

The bigger model is better without the e rule but **worse under the mask**.
4B's masked answers responded to the prompt more often, but were misspelled
far more often (judge notes mention misspellings in 130/200 answers vs 68/200
for 1.7B). 4B keeps writing the words it planned and drops the e's ("you manag
an amusrmt park", "Th ggm rtsins th isomtric vist"): it writes "th" for "the"
in 51/200 answers, against 1/200 for 1.7B. 1.7B more often switches to a
different word. Per the plan, steps 2.5 and 3 stay on 1.7B.

The parity check for 4B ([qwen3_4b_hf_parity.json](../runs/short_eval_20261016/qwen3_4b_hf_parity.json))
ran in BF16 on the GPU; FP32 for 4B does not fit in this machine's memory. Our
code and Hugging Face chose the same top token in all 28 comparisons, with
differences no larger than Hugging Face's own cached-versus-uncached
differences. The 4B checkpoint has no separate `lm_head.weight` (it is tied to
the embedding); the parity script now accepts that only when the model really
shares the two.

## 11 Amount of SFT data (Oct 16)

Step 2.5 of [short_answer_plan.md](short_answer_plan.md). One fresh rank-16
adapter on Qwen3 1.7B per nested training set from
`data/rewrites/splits/` (about half short, half medium and
long answers), fixed 20-example validation set, learning rate 1e-4, batch size
4, 10 passes over the data, best validation-loss checkpoint kept, full chats up
to 3,072 tokens. Each adapter then ran the short eval (200 answers per
condition, judge v3). Runner: [run_all.sh](../runs/sft_sizes_20261016/run_all.sh).

| Training examples | Best validation loss (pass) | Masked usable | vs base (middle 90%) | Instruction alone: no e / usable |
|---:|---:|---:|---:|---:|
| 0 (base) | 3.894 | 26.0% | | 0% / 0% |
| 15 | 2.905 (10) | 10.0% | −16.0 (−23.5 to −9.0) | 0% / 0% |
| 30 | 2.762 (8) | 20.5% | −5.5 (−14.0 to +2.5) | 2.5% / 1.5% |
| 45 | 2.727 (7) | 22.5% | −3.5 (−10.5 to +3.5) | 3.5% / 0.5% |
| 90 | 2.643 (5) | 31.5% | +5.5 (−2.5 to +14.5) | 6.0% / 2.0% |
| 135 | 2.594 (5) | 34.0% | +8.0 (−1.5 to +17.5) | 5.5% / 2.0% |
| 180 | 2.557 (4) | **35.0%** | **+9.0 (+0.5 to +17.5)** | 9.0% / 4.5% |

- **Gain bar not quite met.** The best adapter (180 examples) is 9 points above
  base under the mask, with the resampled range above zero, one point short of
  the 10-point bar.
- **The curve is leveling off:** doubling from 90 to 180 examples added 3.5
  points. More SFT data of this kind is unlikely to be worth writing.
- **Too little data hurts:** 15 examples made masked answers longer (153
  tokens on average, 61 of 200 cut off) and worse.
- **The gain comes from short answers.** More data made answers shorter (92
  tokens for base, 70 at 180). Under 40 tokens, usable rose from 29% to 50%;
  at 40–79 tokens it was 44% for base and 38% at 180; longer answers were the
  same.
- **Coherence rose, responds fell.** Mean coherence went from 3.29 (base) to
  3.73; mean responds from 3.61 to 3.35. Trained adapters substitute a
  different word when the key word needs an e and then answer about that word
  ("clik" for "cliché", "Athos" for "Athens").
- **SFT teaches the rule a little.** Without the mask, the 180 adapter avoided
  e in 9% of answers (base 0%).
- **Overfitting:** with 90+ examples, validation loss was lowest after 4–5
  passes and rose afterwards; the best checkpoint was kept.

## 12 Do good answers appear often enough for RL? (Oct 16)

Step 3a of [short_answer_plan.md](short_answer_plan.md). 50 new prompts from
`data/prompts/train.jsonl` ([rl_check_50.jsonl](../data/prompts/rl_check_50.jsonl)),
chosen with the eval-set rules and excluding every SFT and eval prompt. Mask on,
300-token cap, judge v3. Runs in `runs/rl_check_20261016/`.

| Starting point | Temperature | Answers | Usable | Prompts with a usable answer |
|---|---:|---:|---:|---:|
| Base | 1.0 | 16 per prompt | 27.0% | 78% of prompts (68% expected in a group of 8) |
| 180-example SFT adapter | 1.0 | 16 per prompt | 16.8% | 64% (54% in a group of 8) |
| Base | 0.7 | 4 per prompt | 26.5% | |
| 180-example SFT adapter | 0.7 | 4 per prompt | **40.0%** | adapter − base: +13.5 (+4.0 to +23.5) |

- Both starting points pass the 20% gate easily.
- **Temperature matters for the SFT adapter, not for base.** At 0.7 the adapter
  is clearly better than base on these new prompts, as on the eval set. At 1.0
  its answers fall apart (mean coherence 3.81 at 0.7, 2.95 at 1.0; invented
  words such as "Ranak" and "palacco"). Base is the same at both temperatures.
  RL should therefore sample at 0.7, not the 1.0 first planned.
- Almost every prompt (96–100%) had answers with different rewards, so RL has
  something to compare within each group.

Step 3b code (also Oct 16): `GRPOEngine` takes a `forbidden_token_mask` used
both when sampling and when scoring for training; batches can carry prompt
texts for the reward; a missing reward (failed judge call) leaves that answer
out of the update; `recompute_old_log_probs` rescores answers with the training
pass before updating (in bf16 the cached sampling pass and the full pass
differed by 0.01–0.02 on average, up to about 2 for single tokens); and
`parallel_groups` samples all groups in one batch, which cut a step from about
70 to about 28 seconds. Runner: [scripts/rl_judge.py](../scripts/rl_judge.py).
A 3-step smoke run from base worked end to end.

### RL training prompt pool (Oct 16)

[rl_pool.jsonl](../data/prompts/rl_pool.jsonl): **1,235 prompts** kept from 2,400
candidates in `train.jsonl` (excluding all SFT, eval, validation, test and
step-3a prompts), sorted by ten Haiku 5.5 subagents with the eval-set rules.
Every decision, with the words that decided it, is in `rl_pool_decisions.jsonl`.
Drops: 993 need a name or term containing e, 58 need exact text or code, 23
are too long, 91 other.

Haiku was first checked against the 161 prompts already sorted by hand. Asked
plainly, it agreed on 83% but was too lenient: 19% of its keeps had been
dropped by hand, and nearly every reason was the same sentence. Asked to list
the words a good answer cannot avoid and to mark those containing e, it kept
fewer prompts and only 11% of its keeps had been dropped by hand (several of
those debatable). That second version sorted the pool. A random sample of 40
kept prompts read afterwards had 2 borderline cases and no clear errors.

## 13 RL with the judge as the reward (Oct 16)

Step 3c of [short_answer_plan.md](short_answer_plan.md). GRPO on Qwen3 1.7B with
the mask on, from two starting points: base (fresh rank-16 adapter) and the
180-example SFT adapter from section 11. Reward = (coherence − 1 + responds − 1) / 8
from judge v3; answers are judged in parallel each step. 300 steps × 4 prompts
from [rl_pool.jsonl](../data/prompts/rl_pool.jsonl) × 8 answers each,
temperature 0.7, 300-token cap, learning rate 2e-5, drift penalty (kl_beta) 0.02
toward the starting model, short eval every 25 steps. Runner:
[scripts/rl_judge.py](../scripts/rl_judge.py); runs in `runs/rl_20261016/`.

| Step | Base: usable | SFT 180: usable |
|---:|---:|---:|
| 0 | 26.0% | 35.0% |
| 25 | 28.0% | 43.0% |
| 50 | 34.0% | 53.0% |
| 75 | 39.5% | 55.5% |
| 100 | 44.0% | 51.0% |
| 125 | 44.0% | 58.5% |
| 150 | 35.0% | 60.0% |
| 175 | 38.0% | 56.5% |
| 200 | 39.0% | 60.5% |
| 225 | 36.5% | 61.5% |
| 250 | 40.0% | 61.5% |
| 275 | 40.0% | 59.0% |
| 300 | 43.0% | 60.5% |

Paired differences (middle 90% of 1,000 resamples of the 50 prompts):

| Comparison | Difference |
|---|---:|
| Base RL step 300 vs base | +17.0 (+8.0 to +26.0) |
| SFT 180 RL step 300 vs the SFT 180 adapter | **+25.5 (+17.5 to +33.5)** |
| SFT 180 RL step 300 vs base RL step 300 | +17.5 (+7.5 to +28.0) |
| SFT 180 RL step 300 vs base (no training) | +34.5 (+25.0 to +44.0) |

- **Both runs clear the gain bar.** RL from the SFT adapter ends highest, at
  60.5% usable (85% readable), so the SFT step was worth keeping as a starting
  point: it starts 9 points higher and gains more.
- **Both level off by about step 125.** After that the eval moves within a few
  points (base 35–44%, SFT 56.5–61.5%) and the training reward stays flat
  (base about 0.70–0.76, SFT about 0.78–0.81), while drift from the starting
  model keeps growing (0.04 or less in the first 25 steps; 0.5–1.0 for base
  and 0.2–0.7 for SFT afterwards).
- **Answers got much shorter in both runs.**

  | Eval answers (200) | Mean tokens | Median | Under 25 tokens | Cut off | Coherence | Responds | Coherence 4+ but responds 3 or less |
  |---|---:|---:|---:|---:|---:|---:|---:|
  | Base | 92 | 52 | 51 | 27 | 3.29 | 3.61 | 24 |
  | Base RL step 300 | 42 | 18 | 124 | 5 | 4.20 | 3.56 | 66 |
  | SFT 180 | 70 | 53 | 17 | 7 | 3.73 | 3.35 | 54 |
  | SFT 180 RL step 300 | 40 | 24 | 102 | 3 | 4.39 | 3.75 | 49 |

- **From base, the gain is all coherence.** Responds did not rise, and answers
  that are clean but only partly answer the request went from 24 to 66 of 200
  ("Strong" for "how do you rate strength of tornado", "It's a classic." for
  why In-N-Out burgers are good). The judge already marks these down on
  responds, but the reward adds the two scores, so coherence 5 + responds 3
  earns the same 0.75 as a usable 4 + 4. Under the mask, a short answer is the
  easiest way to a coherence of 5.
- **From the SFT adapter, both scores rose** and the partial-answer count did
  not grow (54 → 49). Usable rose in every length band (under 25 tokens 47% →
  66%; 25–49: 48% → 64%; 50–99: 30% → 54%; 100+: 7% → 25%), so its gain is not
  only from shortening. Many short answers are right for the prompt ("John
  Jay", "SNCF"); multi-part requests now usually get all parts ("Funny. It
  starts with a scary warning, turns into a silly mix-up...").
- **Shrinking is partly a training effect, not only the judge.** Within a group
  of 8, the shorter half earned more reward than the longer half in about two
  groups for every one the other way in the first 25 steps (54 to 28 for base,
  53 to 30 for SFT). The loss also divides each
  answer by its own length (`training/rl/grpo.py`), which gives short answers
  with the same reward a bigger push per word, a known GRPO length bias.
- Cost and speed: no judge errors in 19,200 training calls; peak GPU memory
  10–13 GB under a 14 GB cap; 2.1 h (base) and 1.7 h (SFT) for 300 steps and
  12 evals each. Two earlier attempts were set aside: base ran past GPU memory
  on WSL, which slowed steps to about 170 s instead of failing; SFT ran out of
  memory from a second full copy of the model kept as the drift anchor (fixed
  by swapping the starting adapter values in and out instead).

## 14 Two fixes for shrinking answers (Oct 16–17)

Each run changes one thing from the SFT 180 RL run in section 13 (same start,
seed, prompt order and settings) and stops at 150 steps. Runner:
[run_both.sh](../runs/rl_fixes_20261016/run_both.sh).

- **Lower score as the reward:** reward = (lower of coherence and responds − 1) / 4,
  so a clean answer that skips part of the request (5 and 3) earns less than a
  usable 4 and 4. `--reward_combine=lowest`.
- **Fixed-length loss:** each answer's loss is divided by 300 tokens instead of
  by its own length, which removes the extra push per word that short answers
  get (Liu et al. 2025). `--fixed_length_loss=True`.

| Step | Original (sum, own length) | Lower score | Fixed-length loss |
|---:|---:|---:|---:|
| 0 | 35.0% | 35.0% | 35.0% |
| 25 | 43.0% | 46.0% | 48.5% |
| 50 | 53.0% | 49.5% | 47.5% |
| 75 | 55.5% | 53.0% | 48.5% |
| 100 | 51.0% | 53.5% | 50.5% |
| 125 | 58.5% | 55.5% | 57.5% |
| 150 | 60.0% | 56.0% | 50.5% |

Paired differences against the original at the same step (middle 90%):

| Step | Lower score | Fixed-length loss |
|---:|---:|---:|
| 100 | +2.5 (−3.5 to +9.0) | −0.5 (−8.0 to +7.0) |
| 125 | −3.0 (−10.0 to +3.5) | −1.0 (−7.0 to +5.5) |
| 150 | −4.0 (−10.5 to +2.5) | −9.5 (−15.0 to −4.0) |

- **Neither fix raised the usable rate.** At steps 100 and 125 both are within
  noise of the original. The fixed-length run is lower at step 150, but it
  fell 7 points between its own step 125 and 150 evals, and evals of the same
  run move that much (section 13), so one eval is weak evidence that it is worse.
- **Both slowed the shrinking a little, not much.** Answers under 25 tokens at
  steps 100 / 125 / 150: original 32 / 68 / 67; lower score 19 / 41 / 75;
  fixed-length 27 / 46 / 36. Median length at step 150: 33, 30 and 38 tokens
  (53 before RL).
- **Neither reduced answers that are clean but skip part of the request**
  (coherence 4+, responds 3 or less) at step 150: original 45, lower score 56,
  fixed-length 61.
- **Both drifted about half as far from the starting adapter** (0.15–0.17 in
  steps 126–150, against 0.32).
- **Reading:** shrinking does not come mainly from how the scores are added or
  how the loss is averaged. Under the mask, every extra word is another chance
  to be forced into an odd word, so shorter answers score better on their own
  merits. The original setup remains the best of the three. Each setup has one
  run, so differences of a few points between them are not meaningful.
