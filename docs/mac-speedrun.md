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
