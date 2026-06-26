---
name: ce-synthetic-panel-beta
description: "[BETA] Tests a member's marketing copy against their own AI customer personas with zero API keys, using Claude Code for interview and judgment and synthetic-panel CLI for deterministic prompt emission and local scoring."
disable-model-invocation: true
---

# Synthetic Panel Beta

Use this skill when a member wants to compare marketing-copy variants against a customer persona they describe conversationally. Keep the workflow script-first: the CLI emits deterministic prompts and scores returned reactions; Claude Code handles the interview, independent reaction generation, and final explanation.

## Operating Rules

- Do not ask the member to hand-edit YAML or JSON. Interview them, then write the files yourself.
- Do not request an API key. The default scoring path uses local embeddings.
- Before the first scoring run, tell the member: `The first local scoring run downloads a sentence-transformers embedding model, about 1.5GB. It can take several minutes and may look quiet while it downloads.`
- Generate each sample as an independent in-context draw. Never produce all samples as one list in a single generation.
- Preserve the third-person framing already present in every emitted prompt. Do not rewrite the prompt into first person.
- Present rankings as relative comparisons. Label absolute 1-5 scores as directional.
- If the comparability verdict is `not_comparable`, tell the member to re-run with fresh independent samples rather than trusting the ranking.

## One-Time Install

If `synthetic-panel` is not available, install it from a pinned public source:

```bash
uv tool install git+https://github.com/StartupBros/synthetic-panel@v0.1.0
```

This is the member's only install step (one pinned, immutable release).

## Interview

Ask a handful of concise questions, enough to create one minimal persona and understand the copy being tested:

- What product, service, or offer are you testing?
- Who is the target customer?
- What situation brings this customer to the page or message?
- What variants should we compare? Ask for 2-5 short variants when possible.
- What format is the copy: `headline`, `email_subject`, `offer`, `value_prop`, `cta`, or `price`?

Write a minimal persona YAML file. Use a stable lowercase id and only `name` plus `demographics` unless the member has already supplied a complete rich profile. Prefer this shape:

```yaml
alex:
  name: Alex Rivera
  demographics:
    age: 37
    location: Denver, Colorado
    family: Married
    occupation: Operations consultant
    income: "$120,000 annual income"
```

If the member does not know a demographic field, infer a neutral, plausible placeholder from the interview context and say so briefly. Do not include `situation` unless you also have every rich-profile block required by the CLI.

Write variants as a JSON or YAML object, for example:

```json
{
  "control": "Compare options clearly",
  "proof": "Choose confidently with proof, tradeoffs, and a simple next step"
}
```

## Emit Prompts

Run the CLI to create a versioned prompt handoff. Use 2 samples by default; raise to 3-5 when the member wants a stronger read and accepts the extra time.

```bash
synthetic-panel emit-prompts \
  --persona-file persona.yaml \
  --variants-file variants.json \
  --stimulus-type headline \
  --n-samples 2 \
  --context "Brief neutral context from the interview" \
  --output prompts.json
```

## Generate Reactions

Read `prompts.json`. For each record, separately generate one reaction using exactly that record's `prompt` as the in-context instruction. Keep the reaction as the persona's natural third-person interview response. Do not batch records into a numbered list.

Write `reactions.json` by preserving the prompt handoff fields and adding a `reaction` string to every record. Keep `schema_version`, `stimulus_type`, `n_samples`, `personas`, and all record ids unchanged.

## Score

Before running this for the first time in the environment, announce the local model download notice from Operating Rules.

```bash
synthetic-panel score-reactions reactions.json --json > results.json
```

If the model download or scoring fails, surface the one-line CLI error and fix missing files or malformed JSON yourself when possible.

## Present Results

Report:

- The comparability verdict and its message.
- The relative ranking of variants with concise reasoning based on `reasoning_excerpt` and the reaction text.
- Directional absolute scores, clearly labeled as directional rather than calibrated market truth.
- A caution that v1 uses relative comparison and generic anchors.

When `comparability.verdict` is `not_comparable`, lead with that. Explain that the ranking should not be trusted, name the issue if available, and re-run fresh independent samples before advising on a winner.

For a single-variant run, report the directional score and explain that ranking requires two or more variants.
