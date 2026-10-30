# The Mac speedrun

modded-nanogpt's game, sized for a laptop: train a language model on
FineWeb (the speedrun's own GPT-2 tokens, `kjj0/fineweb10B-gpt2`, read in
order) to a validation loss in as little wall time as possible, on one
MacBook Pro (M4 Pro, 48 GB).  Validation is the first 16 batches of 8 x 1024
tokens of the validation shard.

The model is small, 6 layers x 384 channels, so a run is an hour, not a
night.  The first target: the llm.c recipe's loss after an hour, 5.37.

| # | time to 5.37 | val loss at 1 h (70M tokens) | what | date |
|---|---|---|---|---|
| 0 | 61 min | 5.37 | llm.c's GPT-2 recipe: AdamW, lr 1e-3, warmup then cosine (`gpt2-pretrain`) | 2026-10-30 |
| 1 | 12 min | 4.30 | modded-nanogpt's records 5-11: RMSNorm, QK-norm + RoPE, ReLU^2, zero-init projections, untied head, softcap 30; Muon (lr 0.04) for the hidden matrices, Adam for the embedding and the head (`nanogpt-train`) | 2026-10-30 |

Record 1 reaches record 0's final loss at step 200 of 1070 (13M tokens);
both run at about 19K tokens/s.  The next target is record 1's 4.30.
