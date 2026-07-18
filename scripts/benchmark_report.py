#!/usr/bin/env python3
"""Generate benchmark report (JSON, Markdown, HTML) for model card and README."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ds_rag_embedder.evaluate import compare_models


def _markdown_table(results: dict[str, dict]) -> str:
    lines = [
        "| Model | Recall@1 | Recall@5 | MRR | nDCG@10 | Latency (ms/q) |",
        "|-------|----------|----------|-----|---------|----------------|",
    ]
    for name, metrics in results.items():
        if "overall" in metrics:
            metrics = metrics["overall"]
        lines.append(
            f"| {name} | {metrics.get('recall_at_1', 0):.3f} | "
            f"{metrics.get('recall_at_5', 0):.3f} | {metrics.get('mrr', 0):.3f} | "
            f"{metrics.get('ndcg_at_10', 0):.3f} | "
            f"{metrics.get('latency_ms_per_query') or '-'} |"
        )
    return "\n".join(lines)


def _html_report(results: dict, category_breakdown: dict | None) -> str:
    # simple HTML wrapper
    parts = [
        "<html><head><meta charset='utf-8'><title>DS RAG Benchmark</title>",
        "<style>body{font-family:system-ui;max-width:960px;margin:2rem auto}",
        "table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;padding:8px}",
        "th{background:#f5f5f5}</style></head><body>",
        f"<h1>DS RAG Embedder Benchmark</h1><p>Generated {datetime.now(timezone.utc).isoformat()}</p>",
        "<h2>Model comparison</h2><table>",
        "<tr><th>Model</th><th>Recall@1</th><th>Recall@5</th><th>MRR</th><th>nDCG@10</th><th>Latency</th></tr>",
    ]
    for name, metrics in results.items():
        m = metrics.get("overall", metrics)
        lat = m.get("latency_ms_per_query")
        lat_s = f"{lat:.1f}" if lat is not None else "n/a"
        parts.append(
            f"<tr><td>{name}</td><td>{m.get('recall_at_1', 0):.3f}</td>"
            f"<td>{m.get('recall_at_5', 0):.3f}</td><td>{m.get('mrr', 0):.3f}</td>"
            f"<td>{m.get('ndcg_at_10', 0):.3f}</td><td>{lat_s}</td></tr>"
        )
    parts.append("</table>")
    if category_breakdown:
        parts.append("<h2>Category breakdown (Recall@5)</h2><table>")
        parts.append("<tr><th>Category</th>")
        for model in category_breakdown:
            parts.append(f"<th>{model}</th>")
        parts.append("</tr>")
        cats = sorted({c for m in category_breakdown.values() for c in m.get("by_category", {})})
        for cat in cats:
            parts.append(f"<tr><td>{cat}</td>")
            for model, data in category_breakdown.items():
                val = data.get("by_category", {}).get(cat, {}).get("recall_at_5", 0)
                parts.append(f"<td>{val:.3f}</td>")
            parts.append("</tr>")
        parts.append("</table>")
    parts.append("</body></html>")
    return "".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate DS RAG benchmark report")
    parser.add_argument("--model", default="models/ds-rag-embedder-v1")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    models = {
        "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
        "bge-small-en-v1.5": "BAAI/bge-small-en-v1.5",
    }
    if Path(args.model).exists():
        models["ds-rag-embedder-v1"] = args.model

    results = compare_models(models, include_categories=False)
    category_results = {}
    if Path(args.model).exists():
        category_results["ds-rag-embedder-v1"] = compare_models(
            {"ds-rag-embedder-v1": args.model}, include_categories=True
        )["ds-rag-embedder-v1"]

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "models": results,
        "category_breakdown": category_results,
    }
    (out / "eval_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md = [
        "# DS RAG Benchmark Report",
        "",
        f"Generated: {payload['generated_at']}",
        "",
        _markdown_table(results),
        "",
        "Canonical JSON: `outputs/eval_results.json`. Sync table into `MODEL_CARD.md` when re-publishing.",
    ]
    (out / "benchmark_report.md").write_text("\n".join(md), encoding="utf-8")
    (out / "benchmark_report.html").write_text(
        _html_report(results, category_results or None), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))
    print(f"\nSaved → {out}/eval_results.json, benchmark_report.md, benchmark_report.html")


if __name__ == "__main__":
    main()
