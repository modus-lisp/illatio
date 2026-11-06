# The Mac speedrun

modded-nanogpt's game, sized for a laptop: train a language model on
FineWeb (the speedrun's own GPT-2 tokens, `kjj0/fineweb10B-gpt2`, read in
order) to a validation loss in as little wall time as possible, on one
MacBook Pro (M4 Pro, 48 GB).  Validation is the first 16 batches of 8 x 1024
tokens of the validation shard.

The model is small, 6 layers x 384 channels, so a run is an hour, not a
night.  Times are training time (validation excluded), as the speedrun counts them.

| # | time to 5.37 | time to 4.30 | val loss after 1070 steps of 64K tokens (70M) | what | date |
|---|---|---|---|---|---|
| 0 | 61 min | -- | 5.37 | llm.c's GPT-2 recipe: AdamW, lr 1e-3, warmup then cosine (`gpt2-pretrain`) | 2026-10-30 |
| 1 | 12 min | 62 min | 4.297 | modded-nanogpt's records 5-11: RMSNorm, QK-norm + RoPE, ReLU^2, zero-init projections, untied head, softcap 30; Muon (lr 0.04) for the hidden matrices, Adam for the embedding and the head (`nanogpt-train`) | 2026-10-30 |
| 2 | -- | **52.8 min** | 4.257 | record 11's shortcuts: embedding norm into x0, x0 mixing, value residual, U-net skips; causal attention tiles and a fused softcapped loss (3.17 s a step, from 3.53) | 2026-10-30 |
| 3 | -- | **52.1 min** | 4.222 | warmdown over the last 40% (from 30%), Adam beta1 0.8 (from 0.9) | 2026-10-30 |

The target is now 4.30.

Notes:
- Short runs mislead on batch size.  At 20M tokens, 16K-token steps beat 64K
  ones (4.83 vs 4.90 with the shortcuts), but over the full hour the 16K run
  stalls in the constant-lr phase and ends at 4.346, behind record 1.
  Sweep at the full horizon.
- Doubling Muon's or Adam's learning rates is worse at 20M tokens; halving
  them is no different.
- Softcap 15 instead of 30 is worse: 4.30 at 55.8 min, 4.272 at the end.
- Record 3's steps ran about 4% slower than record 2's (3.30 s against
  3.17 s; same code), so its gain in steps (950 against 1000) is larger than
  its gain in time.  Step times drift by a few percent between runs.

## Reading record 3 by ablation

`ngpt-ablate.lisp` switches components off in record 3's trained weights and
measures the validation loss (8 batches, 4.215 with everything on).  All 62
components take under two minutes; retraining without one takes an hour.
Switching a part off in a trained model is not the same as training without
it, and zeroing overstates a part whose output has a large mean, so this
ranks candidates for retraining rather than replacing it.

| Component | Loss without it |
| --- | --- |
| Layer 0's MLP | +1.353 |
| Layer 2's attention | +1.352 |
| Layer 1's attention | +1.325 |
| Layer 2's value residual | +0.847 |
| Layer 0's attention | +0.580 |
| The U-net skip from layer 2 into 3 | +0.385 |
| Layer 5's MLP | +0.238 |
| Layer 4's attention | +0.074 |
| Each head of layers 3–5 | +0.005 to +0.044 |
| The U-net skip from layer 0 into 5 | +0.010 |
| x0's mixing into layers 2 and 5 | +0.003, +0.009 |
| Layer 1's value residual | +0.002 |

- The early layers carry the model.  Late attention is nearly idle, which
  suggests cheaper attention there (short windows, as later modded-nanogpt
  records use): a retraining worth its hour.
- Some shortcuts barely register in some layers: layer 1's value residual,
  the longest skip, x0's mixing into layers 2 and 5.
