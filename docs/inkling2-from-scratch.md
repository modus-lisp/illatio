# Porting Inkling 2 Mini from its documentation: the gap

Inkling 2 Mini (Thinking Machines, released 2026-11-20) was ported to
illatio the day it came out. It was written first from what was published,
with no implementation read. When that stalled, the code in Hugging Face's
Transformers finished the job. These notes are about the distance between
the two: what a from-scratch port reached, how, where it stopped, and why.
docs/inkling2.md has the step-by-step log.

## What there was to work from

- **The model card and blog post.** These give the shape of the thing: 24
  layers in four blocks of six, the middle twelve run three times, 128
  experts with 8 chosen, local and global attention, NVFP4 experts.
- **The 4-bit training post.** It gives the numeric formats: FP4 values × an
  E4M3 scale per 16 × an FP32 outer scale per expert, and MXFP8 for the
  last four layers.
- **config.json.** About sixty named numbers. Many name mechanisms the prose
  never mentions (`qk_l2_clip_threshold`, `log_scaling_n_floor`,
  `mlp_gate_softcap`, `use_value_norm_no_gamma`, `sconv_kernel_size`), with
  values but no formulas.
- **The checkpoint's tensor names, shapes and dtypes.** These were the most
  informative source after the equations. A `(24, 2304)` sink projection, a
  `(16, 1024)` relative table and a `(32, 2304)` per-token tap projection
  each come close to describing their mechanism by shape alone.
- **Goodfire's interpretability write-up of this model.** It was the one
  document with equations: the depth-wise attention residuals, the
  cross-loop message reads, and the gates that mix them in. Its reader
  counts (31 attention, 24 MoE, 1 output) matched the checkpoint exactly.
  Every equation in it was right as written.
- **The tokenizer files and the chat template.** These are complete
  specifications, and the port matches both exactly.

The weights themselves were the oracle. A wrong reading scores worse
perplexity on ordinary text, so every choice the documents left open was
tried every way it could be read and kept or dropped by score.

## How far that got

**Everything structural came out right:**

- the block and loop data flow;
- which sources each sublayer reads, and how it mixes them;
- the four gated message inputs per looped layer, and which query row feeds
  which;
- the NVFP4 and MXFP8 decoding (checked exactly: a dequantized expert's
  maximum equals its stored `amax`);
- the interleaved gate/up rows;
- the sigmoid router with a selection-only bias, normalized and × 8;
- the per-token sinks, the sigmoid output gate, the relative-position bias;
- grouped-query heads;
- the embedding norm;
- the tokenizer (o200k's split by hand) and the chat template, byte for
  byte.

**The first all-guesses run scored perplexity 575,000**, worse than
choosing uniformly over the vocabulary. One error hid all the others: the
fused expert matrix interleaves its gate and up rows. With that wrong,
turning the experts off scored *better* than leaving them on, so every
other comparison was noise. Once found (760 to 7.6), the rest of the
search worked. **Finding it took a sweep of every layout the tensor's
shape allowed.**

**The port then stood at perplexity 2.7 on a short paragraph.** That looked
like a working model, and nothing local could say otherwise. It answered
questions correctly in its own chat format. The CPU reference, the GPU path
and the NumPy prototype all agreed to 1e-5.

**The first outside comparison changed that.** llama.cpp, used as a black
box (its code not read), scored the same 9586 tokens at 6.29 where the port
scored 11.5. The ratio was 1.33 even with 16-token chunks, so the error was
in every token, not only at long range.

## The six details

What Transformers' code showed the port had wrong. None of them appears in
any of the documents.

| detail | from-scratch reading | actual |
|---|---|---|
| a short conv's output | x + conv(x) | (x + conv(x)) / √5, or / 3 for the dynamic ones |
| the dynamic convs' taps | oldest first, like the static ones | tap k is the token k back |
| K | RMS-normalized (by analogy with `use_value_norm_no_gamma` for V) | L2-clipped to the same threshold as Q, no learned scale |
| Q and the relative query | clipped, then times the learned scale | times the learned scale, then clipped |
| the experts' SwiGLU | silu of the softcapped gate | the softcapped gate × the sigmoid of the *raw* gate |
| a block's first attention read | its empty running sum among the sources | no empty source |

**Where the black-box search got to on each:**

- **The clip order.** The search found it (26.96 → 26.02 at 16 tokens).
- **The first-read source.** The search found it, but by too little to
  trust.
- **K.** RMS-norm and L2-clip tied on short text and stayed within noise of
  each other at length. Nothing could separate them.
- **The SwiGLU form.** The variants tried (softcap on gate, on up, on both,
  on the output, plain silu) did not include the true one. The space of
  plausible forms is larger than the variants anyone thinks to write.
- **The dynamic taps.** The search noticed something wrong, since static
  taps alone scored better, but never found the tap order.
- **The conv rescaling.** It was never suspected. It touches every layer,
  about 100 convs, and a residual that is 1/√5 too large on all of them is
  exactly the kind of error a model half-compensates for: it still works,
  only worse.

Two of these hid a third. The logits wanted to be about 1.6× sharper, and
a fitted constant (0.14 instead of 1/√128), then a principled-looking
guess, `1 + 0.1 ln n` from the config's `log_scaling_alpha`, both
delivered most of it. Both were wrong. The real log scale is exactly 1
below 8192 tokens. The sharpness was standing in for the clip order and
the unscaled residuals, which shrink activations differently. **A fitted
constant that improves perplexity is evidence that something is wrong, not
of what.**

## What the gap says

- **The equations were enough.** Where Goodfire wrote the math down, the
  port matched it on the first try. Every component that came out right
  either had an equation, or had a shape that left only one or two
  readings. **The six that did not were all scalars and orders**: a
  normalization constant, a sign of iteration, a choice of which side of a
  nonlinearity a cap goes on. Documentation of a model tends to describe
  its structure; implementations carry these.
- **Perplexity is a good oracle with a short reach.** It found the
  catastrophic errors immediately. It ranked discrete choices reliably when
  they mattered by more than about 1%. **It could not find a correction
  nobody proposed**, and it rewarded compensating fudges. Without an
  external number to aim at, the port would have shipped at 11.5 believing
  it was fine: nothing internal disagreed.
- **The decisive instrument was another implementation used as a black
  box.** Not its code: its perplexity on the same tokens. That turned "it
  works" into "it is 19% off at every length", and made each remaining
  hypothesis a 40-second test.
- **Cost.** The from-scratch port, including its CPU reference, GPU path,
  tokenizer and chat format, took about a day and reached ~19% above the
  target perplexity. The sweeps of about thirty hypotheses closed some of
  that and plateaued. Reading the reference closed the rest in under an
  hour. Every one of its corrections was a one- or two-line change to code
  that was otherwise right.

## Numbers

Perplexity, llama-perplexity's protocol (the second half of each chunk
scored), on illatio's README (9586 tokens, tokenized identically by
both):

| chunk | from scratch, as settled | best black-box variant | with the reference | llama.cpp |
|---|---|---|---|---|
| 16 | — | 33.5 | — | 25.1 ± 4.6 |
| 32 | 27.0 | 23.9 | 19.99 | 20.01 |
| 128 | 14.6 | 12.5 | 9.17 | 9.17 |
| 512 (18 chunks) | 11.53 | — | 6.297 | 6.286 ± 0.236 |

Speed on the M4 Pro after the fixes (llama-bench's tests, the official
checkpoint vs unsloth's UD-Q4_K_M GGUF; the same NVFP4 experts, Q8_0
projections in both):

| t/s | illatio | llama.cpp |
|---|---|---|
| pp512 | 162 | 73 |
| tg128 | 31.5 | 17.1 |
| pp512 @ d4096 | 116 | 68 |
| tg128 @ d4096 | 22.6 | 17.0 |
