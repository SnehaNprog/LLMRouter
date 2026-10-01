"""The README makes claims about the code. These tests keep it from drifting away from it."""

import importlib
import re
from collections import Counter

import pytest
from fastapi.routing import APIRoute

from app.core.config import Settings
from app.core.tiers import Tier
from app.evaluation.dataset import load_benchmark
from app.main import app
from app.providers.fake import FakeProvider
from app.router.deployment import Deployment
from app.router.latency import LatencyTracker
from app.router.registry import ModelRegistry
from app.routing.selector import choose_model
from tests.conftest import model_config
from tests.test_db import ROOT

README = (ROOT / "README.md").read_text()
MAKEFILE = (ROOT / "Makefile").read_text()


def code_spans() -> list[str]:
    """Everything the README shows as code: fenced blocks and inline `spans`."""
    fenced = re.findall(r"```[a-z]*\n(.*?)```", README, re.DOTALL)
    inline = re.findall(r"`([^`\n]+)`", README)
    return fenced + inline


# ---------- commands ----------


def test_every_make_target_the_readme_mentions_exists() -> None:
    targets = set(re.findall(r"^([a-z][a-z-]*):", MAKEFILE, re.MULTILINE))
    mentioned = {
        match for span in code_spans() for match in re.findall(r"\bmake ([a-z][a-z-]*)", span)
    }
    assert mentioned, "the README should mention some make targets"
    assert mentioned <= targets, sorted(mentioned - targets)


def test_every_python_module_the_readme_runs_exists() -> None:
    modules = {match for span in code_spans() for match in re.findall(r"python -m ([\w.]+)", span)}
    assert {"app.cli", "app.evaluation"} <= modules
    for module in modules:
        importlib.import_module(module)  # a typo would raise here


# ---------- settings ----------


def test_every_setting_the_readme_names_exists() -> None:
    fields = {name.upper() for name in Settings.model_fields}
    named = set(re.findall(r"LLMROUTER_([A-Z0-9_]+)", README))
    assert named, "the README should name some settings"
    assert named <= fields, sorted(named - fields)


# ---------- endpoints ----------


def route_paths() -> set[str]:
    return {route.path for route in app.routes if isinstance(route, APIRoute)}


def normalise(path: str) -> str:
    return re.sub(r"\{[^}]+\}", "{}", path)


def test_every_endpoint_in_the_api_table_exists() -> None:
    section = README.split("## API", 1)[1].split("## How it works", 1)[0]
    known = {normalise(path) for path in route_paths()} | {"/metrics"}
    found = 0
    for line in section.splitlines():
        if not line.startswith("|"):
            continue
        first_column = line.split("|")[1]
        for path in re.findall(r"`(?:[A-Z/]+ )?(/[^`]*)`", first_column):
            for piece in re.split(r",\s*", path):
                if "…" in piece:
                    continue  # shorthand for a longer path shown in full on the same row
                found += 1
                assert normalise(piece.split()[-1]) in known, piece
    assert found >= 8


def test_the_error_codes_the_readme_names_are_ones_the_gateway_sends() -> None:
    source = "".join(p.read_text() for p in (ROOT / "app").rglob("*.py"))
    section = README.split("Notable codes:", 1)[1].split("\n\n", 1)[0]
    for code in re.findall(r"`([a-z_]+)`", section):
        assert f'"{code}"' in source, code


# ---------- links and files ----------


def test_every_relative_link_points_at_a_file_that_exists() -> None:
    links = re.findall(r"\]\(([^)#]+)(?:#[^)]*)?\)", README)
    local = [link for link in links if not link.startswith(("http://", "https://", "mailto:"))]
    assert local
    for link in local:
        assert (ROOT / link).exists(), link


def test_every_path_in_the_layout_table_exists() -> None:
    section = README.split("Project layout:", 1)[1].split("##", 1)[0]
    paths = re.findall(r"\| `([^`]+)`", section)
    assert len(paths) >= 8
    for entry in paths:
        for path in entry.split(", "):
            assert (ROOT / path.strip()).exists(), path


# ---------- the numbers it quotes ----------


def test_the_benchmark_description_matches_the_benchmark() -> None:
    benchmark = load_benchmark()
    counts = benchmark.counts()["difficulty"]
    assert f"**Benchmark:** {len(benchmark)} prompts in six categories" in README
    assert f"is small** ({len(benchmark)} prompts)" in README
    assert f"{counts['easy']} easy, {counts['medium']} medium and {counts['hard']} hard" in README
    mix = "{:.0%} easy, {:.0%} medium, {:.0%} hard".format(
        benchmark.share("easy"), benchmark.share("medium"), benchmark.share("hard")
    )
    assert mix in README
    kinds = benchmark.counts()["scoring"]
    assert f"{kinds['exact'] + kinds['number'] + kinds['contains']} prompts are checked" in README
    assert f"{kinds['rubric']} open-ended" in README


def test_the_routing_split_the_readme_quotes_is_what_the_default_thresholds_do() -> None:
    """It is a claim about this code, so if the scoring changes, the README must be updated."""
    registry = ModelRegistry(
        [
            Deployment.create(
                model_config(name=name, tier=tier, input_cost_per_mtok=i, output_cost_per_mtok=o),
                FakeProvider(name),
            )
            for name, tier, i, o in (
                ("fast", Tier.SLM, 0.05, 0.08),
                ("balanced", Tier.MID, 0.59, 0.79),
                ("smart", Tier.FRONTIER, 3.0, 15.0),
            )
        ]
    )
    overall: Counter[str] = Counter()
    hard: Counter[str] = Counter()
    for item in load_benchmark():
        tier = choose_model(item.to_request("auto"), registry, LatencyTracker()).tier
        assert tier is not None
        overall[tier.value] += 1
        if item.difficulty == "hard":
            hard[tier.value] += 1

    assert f"{overall['slm']} of the 90 benchmark prompts to the small tier" in README
    assert f"{overall['mid']} to the mid tier" in README
    assert overall["frontier"] == 0 and "none to\nthe frontier tier" in README
    assert f"{hard['slm']} of the 25 prompts labelled hard" in README


def test_the_evaluation_targets_in_the_readme_are_the_ones_the_code_checks() -> None:
    from app.evaluation.evaluate import Targets

    assert Targets().cost_saved == 0.70 and Targets().quality_retained == 0.93


# ---------- honesty ----------


def test_the_readme_claims_no_measured_result_unless_a_report_exists() -> None:
    reports = (
        list((ROOT / "eval_results").glob("report*.json"))
        if (ROOT / "eval_results").exists()
        else []
    )
    if reports:
        pytest.skip("a report exists, so the README may state measured results")
    assert "There are no measured results in this repository yet" in README
    assert README.count("not yet run") >= 2
    section = README.split("### Results", 1)[1].split("## Load testing", 1)[0]
    assert not re.search(r"\b\d+(\.\d+)?% (saved|cheaper|lower|retained)", section)


def test_the_resume_table_does_not_call_the_unmeasured_numbers_verified() -> None:
    table = README.split("## Résumé claims", 1)[1].split("## Limitations", 1)[0]
    for claim in ("70% lower inference cost", "93%+ quality retention"):
        row = next(line for line in table.splitlines() if claim in line)
        assert "Not reproduced" in row
    assert "cannot be verified from this\nrepository until the evaluation is run" in table
