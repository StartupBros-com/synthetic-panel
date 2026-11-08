# synthetic-panel

An open-source synthetic customer panel for testing marketing copy. No API key required.

[![CI](https://github.com/StartupBros-com/synthetic-panel/actions/workflows/ci.yml/badge.svg)](https://github.com/StartupBros-com/synthetic-panel/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

`synthetic-panel` helps you compare how different messages, offers, headlines, or product ideas resonate with a synthetic customer panel. v1 is designed for relative ranking: it is best at answering "which variant resonates more?" Absolute 1-5 scores are directional, not a promise of market truth. Niche-specific calibration is a future upgrade.

Scoring uses Semantic Similarity Rating (SSR), following the method described in arXiv:2510.08338. The default setup runs embeddings locally with `all-MiniLM-L6-v2`, so you can score reactions without an API key.

## Quickstart

On Linux, select the CPU backend to avoid downloading unneeded CUDA dependencies:

```bash
uv tool install --torch-backend cpu git+https://github.com/StartupBros-com/synthetic-panel@v0.1.1
synthetic-panel --help
```

On macOS and other platforms, omit `--torch-backend cpu` and use the default backend. Run only the install command for your platform.

For local development from a checkout:

```bash
uv venv
uv pip install -e .
uv run synthetic-panel --help
```

## Example

A real run from November 7, 2026, from a checkout of this repo: the three example personas included in the package (a SaaS founder, a skills learner and a local-services customer), two samples each, against [`examples/saas_headline_trio.json`](examples/saas_headline_trio.json).

```bash
synthetic-panel emit-prompts \
  --persona-file synthetic_panel/data/personas.example.yaml \
  --variants-file examples/saas_headline_trio.json \
  --stimulus-type headline --n-samples 2 --output prompts.json
# Claude wrote one reaction per prompt, each in its own context, into reactions.json
synthetic-panel score-reactions reactions.json
```

```text
Comparability: comparable
Comparable directional ranking. Absolute scores are directional only.
         Directional Ranking
┏━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┓
┃ Rank ┃ Variant ┃ Directional Score ┃
┡━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━┩
│ 1    │ plain   │ 3.052             │
│ 2    │ outcome │ 2.999             │
│ 3    │ proof   │ 2.913             │
└──────┴─────────┴───────────────────┘
```

The spread is small because only the founder is the audience for these headlines; the founder alone scored them 3.39, 3.20 and 3.19. The reactions and the full results are in [`examples/sample-run/`](examples/sample-run/).

## How It Works

Your AI agent generates short customer-reaction text for each persona and variant. `synthetic-panel` then scores those reactions locally by embedding the reaction and comparing it with ordered SSR anchor statements. The output is a probability distribution across a 1-5 Likert scale plus a weighted score.

The included anchors are generic. Results are most reliable as relative comparisons among variants tested in the same run, with the same personas, anchors, embedding model, and temperature settings.

## What v1 Is Good For

- Ranking copy, positioning, offer, CTA, and pricing-message variants.
- Finding variants that are clearly stronger or weaker for a target persona.
- Running zero-key local scoring in an agent workflow.
- Keeping the scoring instrument inspectable through YAML personas and anchors.

## What v1 Is Not

- A replacement for customer interviews, analytics, or paid research panels.
- A source of calibrated market-size, conversion-rate, or revenue forecasts.
- A guarantee that a 4.2 means the same thing across categories, niches, or anchor revisions.
- A direct-API generation product. In v1, generation is handled by your AI agent; this package handles prompts, scoring, and packaging utilities.

## Configuration

Default `.env`:

```bash
SSR_EMBEDDING_PROVIDER=local
```

No API key is needed for the default. The first local scoring run downloads the sentence-transformers model once.

Optional API embedding providers can be configured with `SSR_EMBEDDING_PROVIDER=openai` or `SSR_EMBEDDING_PROVIDER=gemini` plus the relevant API key and provider model. Local embeddings remain the documented default.

## Limitations

- v1 reports relative preference signals. Absolute scores are directional.
- Default anchors are generic and may not reflect the language of your niche.
- Synthetic responses inherit the strengths and weaknesses of the AI agent that generated them.
- Local embedding quality depends on the selected sentence-transformers model.
- Calibration against real customers, domain-specific anchors, and benchmark datasets is planned future work.

## License

MIT. See `LICENSE`.

## Contributions

Contributions are welcome. Please keep changes generic, avoid project-specific fixtures, include tests for behavior changes, and document any scoring or anchor changes clearly. For anchor updates, include validation showing that the ordered Likert ladder still ranks negative, neutral, and positive reactions in the expected order.

---

Built by [Will Mitchell](https://willmitchell.com) at [House of Vibe](https://houseofvibe.ai/?utm_source=github&utm_medium=repo&utm_campaign=synthetic-panel).
