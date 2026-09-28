#!/usr/bin/env python3
"""amd/longctx_test.py - a long-context speed and recall test for a running Strata (or any OpenAI-compatible server).

It builds a prompt of about --tokens tokens from real text on disk (Strata's own source files), hides one made-up
fact (the "needle") at --depth of the way through, asks the model to find it, and measures:

  * time to first token  = how long the model spends reading the prompt before it starts answering
  * prompt speed         = prompt tokens / time to first token
  * output speed         = generated tokens / time spent generating
  * recall               = whether the answer contains the hidden fact

Standard library only; run it with any Python 3:

    python3 amd/longctx_test.py --tokens 100000
    python3 amd/longctx_test.py --tokens 180000 --depth 0.1

Nothing is stored anywhere; it only prints a report.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NEEDLE = "IMPORTANT NOTE FOR THE READER: the vault passphrase is BLUE-HERON-4471, and it was set on 14 March."
QUESTION = ("Somewhere in the text above there is a note giving a vault passphrase and the date it was set. "
            "Reply with the passphrase and the date, then one sentence on what kind of text the rest of the "
            "document is.")
CHARS_PER_TOKEN = 3.3          # source code; the server reports the real count, printed below


def build_text(target_tokens: int) -> str:
    files = sorted([*ROOT.glob("src/**/*.cpp"), *ROOT.glob("src/**/*.cu"), *ROOT.glob("include/**/*.hpp"),
                    *ROOT.glob("docs/*.md"), *ROOT.glob("serve/*.py"), *ROOT.glob("tools/*.py")])
    want = int(target_tokens * CHARS_PER_TOKEN)
    parts, have = [], 0
    while have < want:                                     # repeat the files if one pass is not enough
        for f in files:
            t = f"\n\n===== {f.relative_to(ROOT)} =====\n" + f.read_text(errors="replace")
            parts.append(t)
            have += len(t)
            if have >= want:
                break
    return "".join(parts)[:want]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokens", type=int, default=100000, help="about how long the prompt is (default 100000)")
    ap.add_argument("--depth", type=float, default=0.5, help="where the hidden fact goes, 0 = start, 1 = end")
    ap.add_argument("--url", default="http://127.0.0.1:8081/v1/chat/completions")
    ap.add_argument("--api-key", default="none")
    ap.add_argument("--effort", default="low", help="thinking level: none, low, medium, high (default low)")
    ap.add_argument("--max-tokens", type=int, default=1500)
    a = ap.parse_args()

    text = build_text(a.tokens)
    cut = int(len(text) * min(max(a.depth, 0.0), 1.0))
    cut = text.rfind("\n", 0, cut) + 1 if cut else 0
    prompt = text[:cut] + "\n" + NEEDLE + "\n" + text[cut:] + "\n\n" + QUESTION
    body = {"model": "strata", "stream": True, "max_tokens": a.max_tokens, "reasoning_effort": a.effort,
            "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request(a.url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {a.api_key}"})
    print(f"Sending about {a.tokens:,} tokens ({len(prompt):,} characters), fact hidden at {a.depth:.0%} depth ...")
    print("The model reads the whole prompt before answering; at 100K+ tokens that takes minutes. Please wait.")
    t0 = time.time()
    first = None
    answer, reasoning, usage = [], [], None
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:
            for raw in r:
                line = raw.decode(errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    c = json.loads(data)
                except ValueError:
                    continue
                if c.get("usage"):
                    usage = c["usage"]
                for ch in c.get("choices") or []:
                    d = ch.get("delta") or {}
                    piece = (d.get("content") or "") + (d.get("reasoning_content") or "")
                    if piece and first is None:
                        first = time.time()
                        print(f"  first token after {first - t0:.1f} s - now generating ...")
                    answer.append(d.get("content") or "")
                    reasoning.append(d.get("reasoning_content") or "")
    except Exception as e:
        print(f"\nRequest failed: {e}")
        print("Is Strata running on that address? Is the prompt longer than the context it was started with?")
        return 1
    t1 = time.time()
    out = "".join(answer).strip()
    p_tok = (usage or {}).get("prompt_tokens")
    c_tok = (usage or {}).get("completion_tokens")
    ttft = (first or t1) - t0
    gen_s = max(1e-6, t1 - (first or t1))
    print()
    print("=== result ===")
    print(f"prompt tokens:      {p_tok:,}" if p_tok else "prompt tokens:      (not reported)")
    print(f"time to first token: {ttft:.1f} s" + (f"   -> prompt speed {p_tok / ttft:.0f} tokens/s" if p_tok else ""))
    if c_tok:
        print(f"generated tokens:   {c_tok:,} in {gen_s:.1f} s   -> output speed {c_tok / gen_s:.1f} tokens/s "
              f"(thinking included)")
    found = "BLUE-HERON-4471" in out.upper() and "14 MARCH" in out.upper().replace("MARCH 14", "14 MARCH")
    print(f"recall:             {'FOUND the hidden fact' if found else 'did NOT find the hidden fact (read the answer)'}")
    print()
    print("answer:")
    print(out or "(no answer text; the model may have spent its whole budget thinking - raise --max-tokens)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
