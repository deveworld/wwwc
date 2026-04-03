from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

from .budgeting import BudgetCalibrationInput, evaluate_budget_calibration
from .config import CalibrationConfig
from .datasets import load_jsonl_records
from .gemma_runner import CpuEchoGemmaRunner, GemmaRequest
from .logging_utils import write_accounting
from .manifests import load_manifest, manifest_summary
from .matrix import generate_run_matrix
from .pipeline import run_cpu_calibration
from .prompting import build_final_prompt, build_gemma_messages
from .reporting import summarize_artifacts
from .repro import capture_environment_snapshot, write_environment_snapshot
from .validation import validate_manifest, validate_manifest_against_dataset


SAMPLE_DOCUMENT = """
RULER-style long-context tasks often combine retrieval, aggregation, and tracing demands.
TriStore-BMA studies whether selective writing and exact caching should share a fixed budget.
The stable scaffold stays always on and provides bounded support context for all methods.
Mixed-failure slices are preregistered to prevent cherry-picking after evaluation results arrive.
Gemma 4 E2B and E4B are the calibration ladder for low-cost bring-up before the default E4B run.
Route overhead, chunk size, shortlist K, and cache span length are all frozen after calibration.
Dependency-heavy slices should favor write gains, while exact-recall slices should favor cache gains.
The Go/No-Go decision is whether hybrid allocation shows a plausible interior optimum signal.
""".strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="CPU-first TriStore-BMA utilities.")
    subparsers = parser.add_subparsers(dest="command")

    calibrate_parser = subparsers.add_parser("calibrate", help="Run CPU-only calibration.")
    calibrate_parser.add_argument("--config", required=True, help="Path to a YAML config.")
    calibrate_parser.add_argument("--dry-run", action="store_true", help="Run without writing outputs.")

    inspect_parser = subparsers.add_parser("inspect-manifest", help="Print manifest summary.")
    inspect_parser.add_argument("--manifest", required=True, help="Path to a YAML manifest.")

    validate_parser = subparsers.add_parser("validate-manifest", help="Validate manifest structure.")
    validate_parser.add_argument("--manifest", required=True, help="Path to a YAML manifest.")

    cross_validate_parser = subparsers.add_parser(
        "cross-validate",
        help="Validate a manifest against a dataset file.",
    )
    cross_validate_parser.add_argument("--manifest", required=True, help="Path to a YAML manifest.")
    cross_validate_parser.add_argument("--dataset", required=True, help="Path to a JSONL dataset.")

    dataset_parser = subparsers.add_parser("inspect-dataset", help="Print dataset sample stats.")
    dataset_parser.add_argument("--dataset", required=True, help="Path to a JSONL dataset.")

    runner_parser = subparsers.add_parser("runner-smoke", help="Validate Gemma request formatting.")
    runner_parser.add_argument("--config", required=True, help="Path to a YAML config.")
    runner_parser.add_argument("--thinking", action="store_true", help="Enable thinking mode for the stub.")

    budget_parser = subparsers.add_parser("budget-check", help="Apply plan budget calibration rules.")
    budget_parser.add_argument("--config", required=True, help="Path to a YAML config.")
    budget_parser.add_argument(
        "--median-route-overhead-ms",
        required=True,
        type=float,
        help="Measured median route overhead in milliseconds.",
    )

    matrix_parser = subparsers.add_parser("generate-matrix", help="Generate run specs.")
    matrix_parser.add_argument("--config", required=True, help="Path to a YAML config.")
    matrix_parser.add_argument("--benchmark", required=True, help="Benchmark name.")
    matrix_parser.add_argument("--manifest", required=True, help="Manifest path label.")
    matrix_parser.add_argument("--variants", nargs="+", default=["gemma-4-E2B-it", "gemma-4-E4B-it"])
    matrix_parser.add_argument("--seeds", nargs="+", type=int, default=[13, 17, 23])
    matrix_parser.add_argument("--include-thinking", action="store_true")

    prompt_parser = subparsers.add_parser("prompt-smoke", help="Render final prompt and Gemma messages.")
    prompt_parser.add_argument("--config", required=True, help="Path to a YAML config.")
    prompt_parser.add_argument("--thinking", action="store_true")

    env_parser = subparsers.add_parser("env-snapshot", help="Capture environment metadata.")
    env_parser.add_argument("--output", required=True, help="Where to write the snapshot JSON.")

    report_parser = subparsers.add_parser("summarize-artifacts", help="Summarize accounting outputs.")
    report_parser.add_argument("--root", required=True, help="Artifact root directory.")

    raw_args = sys.argv[1:]
    if not raw_args or raw_args[0].startswith("-"):
        raw_args = ["calibrate", *raw_args]
    args = parser.parse_args(raw_args)
    command = args.command or "calibrate"

    if command == "inspect-manifest":
        summary = manifest_summary(load_manifest(args.manifest))
        print(json.dumps(summary, indent=2))
        return 0

    if command == "validate-manifest":
        result = validate_manifest(load_manifest(args.manifest))
        print(json.dumps(asdict(result), indent=2))
        return 0 if result.ok else 1

    if command == "cross-validate":
        manifest = load_manifest(args.manifest)
        records = load_jsonl_records(args.dataset)
        result = validate_manifest_against_dataset(manifest, records)
        print(json.dumps(asdict(result), indent=2))
        return 0 if result.ok else 1

    if command == "inspect-dataset":
        records = load_jsonl_records(args.dataset)
        payload = {
            "records": len(records),
            "benchmarks": sorted({record.benchmark for record in records}),
            "slices": sorted({record.slice_name for record in records}),
        }
        print(json.dumps(payload, indent=2))
        return 0

    if command == "runner-smoke":
        config = CalibrationConfig.from_file(args.config)
        runner = CpuEchoGemmaRunner()
        response = runner.run(
            GemmaRequest(
                variant=config.gemma_variant,
                enable_thinking=args.thinking,
                messages=build_gemma_messages(
                    system_instruction="You are a helpful assistant.",
                    query=config.sample_query,
                    scaffold=[],
                    cache=[],
                    enable_thinking=args.thinking,
                ),
            )
        )
        print(response.raw_text)
        return 0

    if command == "budget-check":
        config = CalibrationConfig.from_file(args.config)
        report = evaluate_budget_calibration(
            BudgetCalibrationInput(
                base_latency_ms=config.budget.base_latency_ms,
                budget_ratios=config.budget.ratios,
                median_route_overhead_ms=args.median_route_overhead_ms,
                shortlist_k=config.preselector.shortlist_k,
            )
        )
        print(json.dumps(asdict(report), indent=2))
        return 0

    if command == "generate-matrix":
        config = CalibrationConfig.from_file(args.config)
        variants = [config.gemma_variant.__class__(item) for item in args.variants]
        matrix = generate_run_matrix(
            benchmark=args.benchmark,
            manifest=args.manifest,
            variants=variants,
            budget_ratios=config.budget.ratios,
            seeds=args.seeds,
            include_thinking=args.include_thinking,
        )
        print(json.dumps([asdict(item) for item in matrix], indent=2))
        return 0

    if command == "prompt-smoke":
        config = CalibrationConfig.from_file(args.config)
        prompt = build_final_prompt(
            instruction="Answer using the bounded evidence only.",
            query=config.sample_query,
            scaffold=["scaffold span A", "scaffold span B"],
            cache=["cache span A"],
        )
        messages = build_gemma_messages(
            system_instruction="Answer using the bounded evidence only.",
            query=config.sample_query,
            scaffold=["scaffold span A", "scaffold span B"],
            cache=["cache span A"],
            enable_thinking=args.thinking,
        )
        print(json.dumps({"prompt": prompt.render_text(), "messages": messages}, indent=2))
        return 0

    if command == "env-snapshot":
        snapshot = capture_environment_snapshot()
        write_environment_snapshot(args.output, snapshot)
        print(json.dumps(asdict(snapshot), indent=2))
        return 0

    if command == "summarize-artifacts":
        report = summarize_artifacts(args.root)
        print(json.dumps(asdict(report), indent=2))
        return 0

    config = CalibrationConfig.from_file(args.config)
    manifest = load_manifest(config.manifests_path)
    document_text = _load_document_text(config)

    artifacts = run_cpu_calibration(config, document_text)

    print(f"variant={config.gemma_variant.value}")
    print(f"manifest_name={manifest.get('name', 'unknown')}")
    print(f"chunks={len(artifacts.chunks)}")
    print(f"scaffold_spans={len(artifacts.scaffold)}")
    print(f"shortlist={len(artifacts.shortlist)}")
    print(f"cache_spans={len(artifacts.cache_spans)}")
    print(f"allocation_steps={len(artifacts.allocation)}")
    print(f"write_units={artifacts.simulation.write_units}")
    print(f"cache_units={artifacts.simulation.cache_units}")
    print(f"simulated_expected_error={artifacts.simulation.expected_error:.4f}")
    print(f"route_overhead_ms={artifacts.accounting.route_overhead_ms:.3f}")
    print(f"prompt_tokens={artifacts.accounting.final_materialized_prompt_length}")

    if not args.dry_run:
        output_dir = config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        write_accounting(output_dir / "accounting.json", artifacts.accounting)
        (output_dir / "summary.json").write_text(
            json.dumps(
                {
                    "manifest": manifest.get("name", "unknown"),
                    "allocation_steps": [asdict(decision) for decision in artifacts.allocation],
                    "simulation": asdict(artifacts.simulation),
                },
                indent=2,
            )
            + "\n"
        )

    return 0


def _load_document_text(config: CalibrationConfig) -> str:
    if not config.sample_document_path:
        return SAMPLE_DOCUMENT

    path = Path(config.sample_document_path)
    if path.suffix == ".jsonl":
        records = load_jsonl_records(path)
        return "\n".join(record.document for record in records)
    return path.read_text()


if __name__ == "__main__":
    raise SystemExit(main())
