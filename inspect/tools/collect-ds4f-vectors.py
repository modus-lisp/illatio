#!/usr/bin/env python3
"""Reference vectors for DeepSeek V4 Flash from ds4 itself.

For each prompt (raw text, no template): ds4 --dump-tokens gives its token
ids, ds4 --dump-logprobs its greedy continuation with top logprobs.  Written
in inspect/vectors' format (see collect-vectors.py), so ds4f-vectors-test.lisp
teacher-forces illatio through ds4's tokens.

usage: collect-ds4f-vectors.py DS4_DIR MODEL OUT.sexp [STEPS] [BACKEND]

BACKEND is cpu (default: ds4's CPU reference path, the code illatio ports, so
the tight oracle) or metal (its GPU path, which differs from its own CPU
path by about as much as illatio does from it).
"""
import json, os, subprocess, sys, tempfile

CASES = [
    ("plain", "The capital of France is"),
    ("press", "Here is a short history of the printing press. Johannes Gutenberg, a goldsmith from Mainz, developed a system of movable metal type around 1440. Before that, books in Europe were copied by hand, usually by monks working in scriptoria, and a single Bible could take years to produce. Gutenberg combined several existing technologies: the screw press used for wine and olive oil, oil-based inks, and a hand mould for casting type quickly and precisely. His most famous work, the Gutenberg Bible, was printed in about 1455, and fewer than fifty copies survive today. Within fifty years, printing presses had spread to more than two hundred cities across Europe, and millions of books had been printed. The consequences were enormous. Literacy rose, scientific ideas travelled faster, and the Reformation spread its pamphlets far and wide. In short, the printing press"),
    ("code", "def merge_sorted(a, b):\n    \"\"\"Merge two sorted lists into one sorted list.\"\"\"\n    result = []\n    i = j = 0\n    while i < len(a) and j < len(b):\n        if a[i] <= b[j]:\n            result.append(a[i])\n            i += 1\n        else:\n            result.append(b[j])\n            j += 1\n    result.extend(a[i:])\n    result.extend(b[j:])\n    return result\n\n\ndef merge_sort(xs):\n    \"\"\"Sort a list with merge sort, returning a new list.\"\"\"\n    if len(xs) <= 1:\n        return list(xs)\n    mid = len(xs) // 2\n    left = merge_sort(xs[:mid])\n    right = merge_sort(xs[mid:])\n    return merge_sorted(left, right)\n\n\ndef test_merge_sort():\n    assert merge_sort([]) == []\n    assert merge_sort([3, 1, 2]) =="),
]

def lisp_string(text):
    """TEXT as a Lisp string literal: only \\ and " are escaped (a Lisp reader
    takes \\n as n, so JSON's escapes would change the prompt)."""
    return '"' + text.replace('\\', '\\\\').replace('"', '\\"') + '"'

def main():
    ds4_dir, model, out = sys.argv[1], sys.argv[2], sys.argv[3]
    steps = int(sys.argv[4]) if len(sys.argv) > 4 else 24
    backend = sys.argv[5] if len(sys.argv) > 5 else "cpu"
    ds4 = os.path.join(ds4_dir, "ds4")
    with open(out, "w") as f:
        f.write(";; reference vectors from ds4 (%s), collect-ds4f-vectors.py\n" % backend)
        for name, text in CASES:
            toks = subprocess.run([ds4, "-m", model, "--raw", "-p", text, "--dump-tokens"],
                                  capture_output=True, text=True, cwd=ds4_dir).stdout
            ids = [int(line.split()[0]) for line in toks.splitlines() if line.strip() and line.split()[0].isdigit()]
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as t:
                path = t.name
            subprocess.run([ds4, "-m", model] + (["--ssd-streaming"] if backend == "metal" else ["--cpu"])
                           + ["--ctx", "4096", "--raw", "-p", text,
                            "-n", str(steps), "--temp", "0", "--dump-logprobs", path],
                           capture_output=True, cwd=ds4_dir, check=True)
            d = json.load(open(path)); os.unlink(path)
            assert d["prompt_tokens"] == len(ids), (name, d["prompt_tokens"], len(ids))
            f.write('(:name "%s"\n :text %s\n :prompt (%s)\n :steps (\n' % (name, lisp_string(text), " ".join(map(str, ids))))
            for s in d["steps"]:
                top = " ".join("(%d %.6f)" % (e["token"]["id"], e["logprob"]) for e in s["top_logprobs"][:10])
                f.write("  (%d %s)\n" % (s["selected"]["id"], top))
            f.write("))\n")
            print(name, len(ids), "prompt tokens,", len(d["steps"]), "steps", file=sys.stderr)

main()
