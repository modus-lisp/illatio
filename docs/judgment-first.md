# Judgment-first pretraining: an experiment on one Mac

Does training a small decoder first on a one-token coherence verdict make it
absorb story structure faster than next-token prediction, or than
distillation from the same teacher?

**Hypothesis.** Most of next-token loss is surface entropy (which name, which
adjective).  A verdict loss spends the whole signal on whether a story hangs
together.  So a judgment phase followed by next-token training should reach a
target held-out loss with fewer tokens processed in total.

**Not tested:** whether the student beats the teacher.  The question is how
fast it absorbs structure, not how high its ceiling is.

**Format.** No new architecture and no new head.  The verdict is ordinary
next-token prediction: the story, then ` Coherent?`, with the loss on one
answer token, `yes` or `no`.

Everything runs in illatio, on one 48 GB M4 Pro: the teacher, the trainer,
the judge, and the weight-space measurements.

## What illatio provides

| Job | illatio | To build |
| --- | --- | --- |
| Train the student | `nanogpt-train`: the speedrun architecture with Muon, about 21K tokens/s at 6x384 | Tied embeddings, a small vocabulary, a loss weight per position, packed story data |
| Write corruptions | `--serve` with Qwen3-4B-Instruct | Generation is serial |
| Teacher logprobs (arm B) | Teacher-forced passes (`vectors-test`) | A dump of top-k logprobs per position |
| Judge samples | The 27B on Metal | None |
| Minimal pairs, probes, ablations | `ngpt-ablate.lisp`: forward passes, activation means, weight edits | Sum of log-likelihood per story |

## Setup

| Role | Choice | Why |
| --- | --- | --- |
| Corpus | TinyStories V2, the GPT-4 split (CDLA-Sharing-1.0) | Skips generating volume serially |
| Corruption writer | Qwen3-4B-Instruct | Strong enough to make one precise edit |
| Distillation teacher | Qwen2.5-0.5B | Fastest prefill, far larger than the student |
| Final judge | The 27B on Metal | Independent of the training teachers |
| Student | The speedrun model, 8 layers, width 384, context 512 | TinyStories models this size write coherent stories |

**Tokenizer.** The Qwen tokenizer, pruned to the ids the corpus uses (likely
10K–15K).  A 151K vocabulary would dominate a 20M model.  The vocabulary
stays shared with the teachers, so arm B renormalizes their top-k over the
kept ids.

**Tied embeddings** in every arm.  Verdict-only training touches the head only
at `yes` and `no`; tying lets reading train every word's vector anyway.  This
departs from the speedrun's untied head, so it is a flag.

**Packing.** Examples are packed into rows of 512, separated by
`<|endoftext|>`, as pretraining usually does.  Verdict examples carry a loss
weight only at their answer position, so the loss weights become a buffer
rather than one scalar.

## Data

The verdict data is minimal pairs: the same story intact, or with exactly one
break.  The only thing separating the labels is coherence.

| Source | Edits | Volume |
| --- | --- | --- |
| Programmatic (`corrupt.lisp`) | Swap two sentences; call a character by another character's name from some sentence on (a wrong referent); splice in another story's ending | As many as needed, free |
| Teacher (Qwen3-4B) | One edit of one type: a contradiction of an earlier fact, a cause without effect, an out-of-character act, a wrong referent, a timeline error | About 20K, with the sentence changed |

- **One type held out.** Timeline errors stay out of training and are used
  only in evaluation, to test whether structure learning generalizes.
- **Verdict format.** Cut each story after the break, append ` Coherent?`,
  and train on the answer token.  Intact stories are cut at the same
  distribution of points.  Labels are 50/50.
- **Distance curriculum.** The difficulty is the token distance between the
  fact a break violates and the break.  Start under about 50 tokens, then
  widen as the verdict loss falls.
- **Shortcut audit, before any training.** Length-match each pair, then fit
  a bag-of-bigrams logistic regression on intact against corrupted.  Above
  about 60%, the corruptions leave surface traces; remove them first.

  The audit, on 4,000 pairs of each kind from the first 20,000 training
  stories, cut as verdict examples (held-out accuracy, chance 0.5):

  | Break | Bigram accuracy |
  | --- | --- |
  | Swap | 0.561 |
  | Wrong referent | 0.561 |
  | Splice | 0.591 |
  | Rename (not used) | 0.707 |

  What the audit caught on the way:
  - Sentences split inside quotations, so a swap could carry half a quote
    out of its marks.  A quotation and its attribution are now one piece.
  - A short donor's whole story spliced in, opening and all ("Once upon a
    time").  The ending now comes from the donor's second half.
  - Renaming, even with the names balanced (drawn alike for both sides, by
    gender), leaves the corrupted story one name more than its twin, and a
    bag of words can count names.  The wrong referent replaces it: both
    names are already in the story.

## Training arms

Every arm gets 50M tokens processed, drawn from one fixed 25M-token subset
(two passes).  The budget counts tokens, not loss positions: a verdict
example runs forward and backward over the whole story.

| Arm | Training | Tests |
| --- | --- | --- |
| A | Next-token on the stories | Baseline |
| B | Distillation: top-20 teacher logprobs per position, mixed with the true-token loss | The strong baseline to beat |
| C | Verdict-only for the first J% of the budget, then next-token; J = 10, 25, 50 | Judgment first, in stages |
| D | Verdict and next-token in the same batches, the verdict share annealed from 50% to 0 | Judgment first, blended |
| E | Arm C with shuffled labels | Whether gains come from the verdict or from reading |

Three seeds per arm, with the same optimizer, schedule and batch size in
tokens.  Verdict examples have their own loss weight, tuned once on a short
run.  At the speedrun's throughput an arm is about 40 minutes, so all 21 runs
take one long night.

## Measurements

1. **Tokens to target.** The target is arm A's final held-out loss; the
   judgment phase counts in the total.  Averaged over seeds.
2. **Minimal-pair accuracy.** On held-out pairs, does the model give the
   intact story the higher log-likelihood?  No verdict prompt is needed, so
   every arm is scored the same way.  The held-out timeline type is reported
   separately.
3. **Sample quality.** 200 stories per arm from the same openings, judged
   blind and pairwise by the 27B for coherence.
4. **Weight space** (`ngpt-ablate.lisp`).
   - **The coherence direction.** The difference of the residual stream's
     means over intact and corrupted stories, per layer, at the positions
     after the break.  Minimal pairs are an unusually clean contrast.
   - **Its causal role.** Remove the direction from the writers
     (W <- (I - D D^T) W), then measure next-token loss and minimal-pair
     accuracy again.  If removing it hurts next-token prediction in arm C but
     not in arm A, the judgment phase built structure the writer uses.  If it
     hurts only the verdict, C trained a critic.
   - **The ablation table** per arm: which heads and layers carry the
     verdict, and whether arm C's model is organized differently from A's.

| Result | Reading |
| --- | --- |
| C or D reaches the target with at least 20% fewer tokens than A, and no more than B | Judgment-first speeds structure learning; go to the RL stage |
| C or D wins only on minimal pairs | It trained a critic, not a writer |
| E matches C | The gain came from reading, not the verdict |
| C and D trail A | One token of signal per story is too thin at this scale |
| The direction matters to C's next-token loss, not A's | The mechanism, not only the outcome |

## Plan

1. **Data and tokenizer.** TinyStories, the 25M-token subset, the held-out
   set, the pruned vocabulary.
2. **Trainer changes.** Tied embeddings, the small vocabulary, loss weights
   per position, packed rows of 512.  Arm A as the first run.
3. **Programmatic corruptions and the shortcut audit.**
4. **Teacher corruptions** (overnight): about 20K from Qwen3-4B.
5. **Pilot.** Arms A and C, one seed, 10M tokens; tune the verdict weight.
6. **Distillation logprobs** (overnight), after the top-k dump exists.
7. **Full runs** (one night), then the evaluation.

## Pilot (2026-11-06)

8x384 with tied embeddings (embedding lr 0.01, the best of a sweep of
0.003, 0.01, 0.03 and untied), context 512, one seed, 10M tokens an arm
(150 steps of 64K).  Minimal-pair accuracy is on 300 held-out pairs per
kind.

| Arm | Val loss | Swap | Referent | Splice |
| --- | --- | --- | --- | --- |
| A, next-token | **1.883** | 0.982 | 0.892 | 0.780 |
| C, verdict for the first 25% | 2.264 | 0.968 | 0.755 | 0.677 |
| D, verdict weight 32 | 1.919 | 0.982 | 0.902 | 0.787 |
| D, verdict weight 128 | 1.937 | 0.985 | 0.915 | 0.767 |

- **No arm learned the verdict.**  Every arm's held-out verdict accuracy
  is 0.50; C's verdict loss stayed at ln 2 for its whole phase, and D
  settled on one answer.
- **Not from a trained model either.**  Arm A's model, trained on verdicts
  alone for 4M tokens, stays at ln 2.  (And with the loss only on the
  answer, it learns to predict "yes"/"no" everywhere: its next-token loss
  goes from 1.92 to 19.6.)
- **The pipeline is sound.**  A learnable label through the same batches
  (does the story name Tim?) trains from 0.88 to 0.007 in 30 steps.
- **Why: the model has no single-story judgment to read.**  A linear probe
  on the stream at a held-out story's last token (200 pairs a kind, chance
  0.5, about +-0.025) finds little: best layer for swaps 0.548, referents
  0.527, splices 0.587 (arm A); 0.555, 0.540, 0.645 after D's verdict
  training.  Arm A's 98% on swapped pairs is a comparison of summed
  likelihoods, not a state it holds.

So at this scale the pre-registered reading is the fourth: one token of
signal per story is too thin.

## Risks

| Risk | Mitigation |
| --- | --- |
| Corruptions leave surface traces | Length-matched pairs; the bigram audit; the held-out type |
| Verdict training stalls at chance from random init | A very short distance curriculum; if it still stalls, a variant of C with 2% next-token warmup, reported separately.  A probe for the coherence direction every N steps shows a stall early |
| One loss position per story is a thin, noisy gradient | Its own loss weight, tuned in the pilot; larger batches in the verdict phase |
| The pruned vocabulary breaks the teacher's distribution | Renormalize top-k over the kept ids; report the mass lost |
| One corpus | TinyStories is a first test; repeat on a second simple corpus before concluding |

## If positive: the RL stage

The same model writes endings and a frozen copy of itself scores them.

- **Policy:** the best C or D checkpoint, continuing from where next-token
  training ended.
- **Reward:** the probability of `yes` from a copy frozen at the end of the
  judgment phase.  Self-scoring would drift toward leniency.
- **Guardrails:** a KL penalty toward arm A's model, plus length and
  diversity checks.
- **A weight-space baseline first:** add the coherence direction to the
  writers, scaled, and see whether samples become more coherent with no RL
  at all.  RL has to beat that.
- **Comparison:** the same RL from arm A with a separately trained verdict
  model.
