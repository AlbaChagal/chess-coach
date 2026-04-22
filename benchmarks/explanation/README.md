# Explanation Benchmarks

This directory contains curated benchmark cases for the explanation module.

The benchmark focuses on deterministic explanation quality first:

- recurring idea detection,
- convergence vs divergence classification,
- best-move role wording,
- shared-plan wording,
- and conservative counterplay handling.

The seed dataset lives in [positions.jsonl](./positions.jsonl). Each line is one
JSON object with:

- `id`
- `fen`
- `top_n`
- `expected`
- `notes`

Run the deterministic benchmark with:

```bash
uv run python scripts/evaluate_explanations.py
```

The script prints a short summary and can optionally write a JSON report.
