#!/usr/bin/env python3
"""Token ids from ds4 itself for DeepSeek V4 Flash, for ds4f-tokenizer-test.lisp.

usage: collect-ds4f-tokens.py DS4_DIR MODEL OUT.sexp
"""
import os, subprocess, sys

TEXTS = [
    "Hello, world!",
    "The year 2026 had 365 days; 1234567 is a big number, 3.14159 a small one.",
    "Caffè, perché sì: l'Italia è bella. Ünïcödé façade naïve résumé.",
    "日本語のテキストと漢字、カタカナ。中文也可以。한국어 텍스트",
    "emoji 🚀🔥 and symbols ← → ∑ ∞ © ®",
    "def f(x):\n    if x > 0:\n        return x * 2\n\treturn -x\n",
    "int main() {\n    printf(\"%d\\n\", 42);\n    return 0;\n}\n",
    "a  b   c    d\n\n\nnew paragraph\r\nwindows line\n  indented",
    "<｜begin▁of▁sentence｜>You are helpful.<｜User｜>Hi there<｜Assistant｜></think>Hello!<｜end▁of▁sentence｜>",
    "x=1;y=2;\nz = x+y;\n// comment\n/* block */\n",
    "<think>\nreasoning here\n</think>\n\nanswer",
    "   leading spaces and trailing   ",
    "URL https://example.com/path?q=1&r=two#frag and email a.b@c.org",
    "MixedCASE camelCase snake_case kebab-case ALLCAPS",
]

def lisp_string(text):
    return '"' + text.replace('\\', '\\\\').replace('"', '\\"') + '"'

def main():
    ds4_dir, model, out = sys.argv[1:4]
    ds4 = os.path.join(ds4_dir, "ds4")
    with open(out, "w") as f:
        f.write(";; DeepSeek V4 Flash token ids from ds4 --raw --dump-tokens (collect-ds4f-tokens.py)\n")
        for text in TEXTS:
            r = subprocess.run([ds4, "-m", model, "--raw", "-p", text, "--dump-tokens"],
                               capture_output=True, text=True, cwd=ds4_dir)
            ids = [int(l.split()[0]) for l in r.stdout.splitlines() if l.strip() and l.split()[0].isdigit()]
            f.write("(%s\n (%s))\n" % (lisp_string(text), " ".join(map(str, ids))))
            print(len(ids), repr(text[:40]), file=sys.stderr)

main()
