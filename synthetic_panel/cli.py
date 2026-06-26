"""Agent-friendly CLI for synthetic-panel's zero-key prompt/reaction seam."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .config import get_persona_names, load_default_anchors, load_personas
from .testing import (
    DEFAULT_N_SAMPLES,
    DEFAULT_STIMULUS_TYPE,
    SeamInputError,
    emit_prompt_handoff,
    score_reaction_handoff,
    write_prompt_handoff,
)

__all__ = ["main", "display_results"]

console = Console(stderr=True)
stdout_console = Console()


class CLIError(Exception):
    """Raised for one-line actionable CLI errors."""


def display_results(persona_name: str, persona_full_name: str, results: list):
    """Display legacy test results in a formatted table."""
    table = Table(title=f"Results for {persona_full_name}")
    table.add_column("Rank", style="cyan", width=4)
    table.add_column("Variation", style="white", max_width=50)
    table.add_column("Score", style="green", width=6)
    table.add_column("Confidence", style="yellow", width=12)

    for i, result in enumerate(results, 1):
        score = result["score"]
        filled = int((score - 1) / 4 * 10)
        bar = "█" * filled + "░" * (10 - filled)
        variation = result["content"]
        if len(variation) > 47:
            variation = variation[:47] + "..."
        table.add_row(str(i), variation, f"{score:.2f}", bar)

    stdout_console.print(table)
    if results:
        top = results[0]
        stdout_console.print(
            Panel(
                top["response"][:500] + ("..." if len(top["response"]) > 500 else ""),
                title="[bold green]Top Performer Rationale[/bold green]",
                subtitle=f"Score: {top['score']:.2f}",
            )
        )


def _load_variants(args: argparse.Namespace) -> dict[str, str]:
    variants: dict[str, str] = {}
    if args.variants_file:
        path = Path(args.variants_file)
        if not path.exists():
            raise CLIError(f"missing variants file: {path}")
        if path.stat().st_size == 0:
            raise CLIError(f"empty variants file: {path}")
        try:
            if path.suffix.lower() in {".yaml", ".yml"}:
                loaded = yaml.safe_load(path.read_text())
            else:
                loaded = json.loads(path.read_text())
        except (json.JSONDecodeError, yaml.YAMLError) as exc:
            raise CLIError(f"invalid variants file: {path}") from exc
        if not isinstance(loaded, dict):
            raise CLIError(f"invalid variants file: {path} must contain an object")
        variants.update({str(k): str(v) for k, v in loaded.items()})

    for raw_variant in args.variant or []:
        if "=" not in raw_variant:
            raise CLIError("invalid variant: use --variant id=copy")
        variant_id, copy_text = raw_variant.split("=", 1)
        variants[variant_id.strip()] = copy_text.strip()

    if not variants:
        raise CLIError("missing variants: provide --variant id=copy or --variants-file")
    return variants


def _load_personas_for_cli(path: str | None) -> dict:
    persona_path = Path(path) if path else None
    if persona_path and not persona_path.exists():
        raise CLIError(f"missing persona file: {persona_path}")
    try:
        personas = load_personas(persona_path)
    except FileNotFoundError as exc:
        raise CLIError(
            f"missing persona file: {persona_path or 'default personas'}"
        ) from exc
    if not personas or not get_persona_names(personas):
        raise CLIError(f"empty persona file: {persona_path or 'default personas'}")
    return personas


def _emit_prompts(args: argparse.Namespace) -> int:
    personas = _load_personas_for_cli(args.persona_file)
    persona_ids = args.persona or None
    variants = _load_variants(args)
    try:
        handoff = emit_prompt_handoff(
            personas,
            variants,
            stimulus_type=args.stimulus_type,
            n_samples=args.n_samples,
            persona_ids=persona_ids,
            context=args.context or "",
        )
    except SeamInputError as exc:
        raise CLIError(str(exc)) from exc

    if args.output:
        write_prompt_handoff(handoff, args.output)

    if args.json or not args.output:
        print(json.dumps(handoff, indent=2, sort_keys=True))
    else:
        record_count = len(handoff["records"])
        stdout_console.print(f"Wrote {record_count} prompts to {args.output}")
    return 0


def _score_reactions(args: argparse.Namespace) -> int:
    path = Path(args.reactions_file)
    if not path.exists():
        raise CLIError(f"missing reactions file: {path}")
    if path.stat().st_size == 0:
        raise CLIError(f"empty reactions file: {path}")
    try:
        result = score_reaction_handoff(
            path,
            reference_statements=load_default_anchors(args.anchors_file),
            dist_temperature=args.dist_temperature,
        )
    except SeamInputError as exc:
        raise CLIError(str(exc)) from exc
    except (OSError, ValueError) as exc:
        message = str(exc).splitlines()[0]
        if "SentenceTransformer" in message or "does not appear" in message:
            message = "missing local model: install/download the configured sentence-transformers model"
        raise CLIError(message) from exc

    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True))

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        verdict = result["comparability"]["verdict"]
        stdout_console.print(f"Comparability: {verdict}")
        stdout_console.print(result["comparability"]["message"])
        if result["ranking"]:
            table = Table(title="Directional Ranking")
            table.add_column("Rank")
            table.add_column("Variant")
            table.add_column("Directional Score")
            for item in result["ranking"]:
                table.add_row(
                    str(item["rank"]),
                    item["variant_id"],
                    f"{item['score']:.3f}",
                )
            stdout_console.print(table)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Emit prompts and score returned reactions for synthetic-panel.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    emit = subparsers.add_parser(
        "emit-prompts", help="Write prompts.json for external generation"
    )
    emit.add_argument("--persona-file", required=True, help="Path to persona YAML file")
    emit.add_argument(
        "--persona", action="append", help="Persona id to include; repeatable"
    )
    emit.add_argument("--stimulus-type", default=DEFAULT_STIMULUS_TYPE)
    emit.add_argument(
        "--variant", action="append", help="Copy variant as id=text; repeatable"
    )
    emit.add_argument(
        "--variants-file", help="JSON/YAML object mapping variant id to copy"
    )
    emit.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES)
    emit.add_argument("--context", default="")
    emit.add_argument("--output", "-o", help="Output prompts.json path")
    emit.add_argument("--json", action="store_true", help="Print prompt handoff JSON")
    emit.set_defaults(func=_emit_prompts)

    score = subparsers.add_parser(
        "score-reactions", help="Score reactions.json locally"
    )
    score.add_argument("reactions_file", help="Path to reactions JSON file")
    score.add_argument("--anchors-file", help="Optional anchor YAML file")
    score.add_argument("--dist-temperature", type=float, default=1.0)
    score.add_argument("--output", "-o", help="Output result JSON path")
    score.add_argument(
        "--json", action="store_true", help="Print stable parseable JSON"
    )
    score.set_defaults(func=_score_reactions)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except CLIError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
