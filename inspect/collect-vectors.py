#!/usr/bin/env python3
"""Record reference vectors from a CPU llama-server.

For each prompt: the prompt's token ids (as llama.cpp tokenizes them), then
N greedy steps, each with the chosen token and the top-K logprobs before
sampling.  Written as s-expressions so the Lisp test needs no JSON reader.

Run the server CPU-only with an f32 KV cache, so the reference is the
plainest ggml path (no Metal, no f16 rounding of K/V):

  llama-server -m MODEL --port 18931 -ngl 0 -fa off -ctk f32 -ctv f32 -c 4096
  inspect/collect-vectors.py http://localhost:18931 inspect/vectors/qwen2.5-0.5b-q8_0.sexp
"""
import json, sys, urllib.request

PROMPTS = [
    ("plain", "The capital of France is"),
    ("chat", "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
             "<|im_start|>user\nWrite a haiku about Lisp.<|im_end|>\n<|im_start|>assistant\n"),
    ("code", "(defun fib (n)\n  \"Return the Nth Fibonacci number.\"\n"),
    ("unicode", "Le café naïve — 東京は日本の首都です。 It's 2026, and"),
    ("numbers", "1, 1, 2, 3, 5, 8, 13, 21,"),
]
STEPS, TOPK = 48, 10

def post(base, path, body):
    req = urllib.request.Request(base + path, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req))

def lisp_string(s):
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'

def main(base, out):
    with open(out, "w") as f:
        f.write(";; reference vectors from CPU llama.cpp (collect-vectors.py)\n")
        for name, text in PROMPTS:
            ids = post(base, "/tokenize", {"content": text, "add_special": False,
                                           "parse_special": True})["tokens"]
            r = post(base, "/completion", {"prompt": ids, "n_predict": STEPS,
                                           "temperature": 0, "top_k": 1, "n_probs": TOPK,
                                           "post_sampling_probs": False,
                                           "cache_prompt": False, "return_tokens": True})
            f.write(f"(:name {lisp_string(name)}\n :text {lisp_string(text)}\n")
            f.write(f" :prompt ({' '.join(map(str, ids))})\n :steps (\n")
            for step in r["completion_probabilities"]:
                top = " ".join(f"({p['id']} {p['logprob']:.6f})" for p in step["top_logprobs"])
                f.write(f"  ({step['id']} {top})\n")
            f.write(" ))\n")
            print(name, len(ids), "prompt tokens,", len(r["completion_probabilities"]), "steps")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
