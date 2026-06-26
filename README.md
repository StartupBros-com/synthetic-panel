# synthetic-panel

An open-source, calibrated synthetic customer panel — no API key required.

`synthetic-panel` helps you compare how different messages, offers, headlines, or product ideas resonate with a synthetic customer panel. v1 is designed for relative ranking: it is best at answering "which variant resonates more?" Absolute 1-5 scores are directional, not a promise of market truth. Niche-specific calibration is a future upgrade.

Scoring uses Semantic Similarity Rating (SSR), following the method described in arXiv:2510.08338. The default setup runs embeddings locally with `all-MiniLM-L6-v2`, so you can score reactions without an API key.

## Quickstart

```bash
uv tool install git+https://github.com/StartupBros/synthetic-panel@v0.1.0
synthetic-panel --help
```

For local development from a checkout:

```bash
uv venv
uv pip install -e .
uv run synthetic-panel --help
```

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
