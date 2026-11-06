"""G5: LM Studio suggestion benchmark - JSON validity and latency over N calls.

    uv run python scripts/llm_bench.py                  # 20 calls, 4 suggestions each (LIVE)
    uv run python scripts/llm_bench.py --wifi off       # records the Wi-Fi condition in results

Start LM Studio first: Developer tab -> load a model -> Start Server (port 1234).
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from neurohack.config import load_config
from neurohack.llm import LLMClient
from neurohack.session import Session


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wifi", choices=["on", "off", "unknown"], default="unknown")
    ap.add_argument("--n", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config("llm")
    b = cfg["benchmark"]
    n_calls, n_sug = args.n or b["n_calls"], b["n_suggestions"]
    client = LLMClient(cfg)
    try:
        client._ensure()
        print(f"server {cfg['base_url']} model={client.model!r}")
    except Exception as e:
        print(f"GATE G5 - FAIL   [LIVE]  server not reachable: {type(e).__name__}: {e}")
        print("Start LM Studio -> Developer -> Start Server, load a model, and re-run.")
        return
    sess = Session("llm_bench", label="LIVE")
    client.suggest("warm up", "", n_sug)  # first call loads weights / caches; excluded
    rows = []
    for i in range(n_calls):
        ctx, partial = b["prompts"][i % len(b["prompts"])]
        s = client.suggest(ctx, partial, n_sug)
        valid = s.strict_json  # gate criterion: the reply itself is valid JSON with n suggestions
        usable = s.source == "llm" and len(s.items) == n_sug  # after parsing/repair, no fallback
        rows.append(
            {
                "context": ctx,
                "partial": partial,
                "latency_s": s.latency_s,
                "source": s.source,
                "valid_json": valid,
                "usable": usable,
                "repaired": s.repaired,
                "items": s.items,
                "error": s.error,
            }
        )
        sess.log("llm_call", **rows[-1])
        print(
            f"{i + 1:2d} {s.latency_s * 1e3:6.0f} ms  {s.source:13s} {partial!r:22s} -> {s.items}"
            + (f"   [error: {s.error}]" if s.error else "")
            + ("" if valid else "   [NOT strict JSON" + (", repaired]" if s.repaired else "]"))
        )
    lat = np.array([r["latency_s"] for r in rows])
    n_valid = sum(r["valid_json"] for r in rows)
    n_usable = sum(r["usable"] for r in rows)
    p50, p90 = float(np.percentile(lat, 50)), float(np.percentile(lat, 90))
    if n_valid == n_calls and p90 < b["pass_p90_s"]:
        v = "PASS"
    elif n_valid >= 0.9 * n_calls and p90 < b["partial_p90_s"]:
        v = "PARTIAL"
    else:
        v = "FAIL"
    res = {
        "model": client.model,
        "wifi": args.wifi,
        "n_calls": n_calls,
        "n_suggestions": n_sug,
        "n_valid": n_valid,
        "n_usable_after_repair": n_usable,
        "latency_p50_s": p50,
        "latency_p90_s": p90,
        "latency_max_s": float(lat.max()),
        "structured_output": cfg["structured_output"],
        "verdict": v,
        "calls": rows,
    }
    sess.save_results(res)
    sess.close()
    print(json.dumps({k: v for k, v in res.items() if k != "calls"}, indent=1))
    print(
        f"\nstrict JSON {n_valid}/{n_calls} (gate criterion); usable after repair {n_usable}/{n_calls}"
    )
    print(f"GATE G5 - {v}   [LIVE]  wifi={args.wifi}")
    print(f"Evidence: {sess.dir / 'results.json'}")


if __name__ == "__main__":
    main()
