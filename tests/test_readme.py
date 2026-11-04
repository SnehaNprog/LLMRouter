"""The README makes claims about the code. These tests keep those claims true.

They check what the README *says*, not how it is laid out: a section that is absent makes no
claims, so editing, reordering or deleting parts of the README never breaks them. They fail only
when the README states something the code contradicts (a make target that does not exist, a
setting that was renamed, a figure that is wrong) or claims a measured result nobody measured.
"""

import importlib
import re
from collections import Counter

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


def section(start: str, end: str | None = None) -> str | None:
    """The README text from ``start`` to ``end`` (or the next heading), or None if it has none."""
    if start not in README:
        return None
    text = README.split(start, 1)[1]
    if end is not None and end in text:
        return text.split(end, 1)[0]
    return re.split(r"\n#{1,3} ", text, maxsplit=1)[0]


# ---------- commands ----------


def test_every_make_target_the_readme_mentions_exists() -> None:
    targets = set(re.findall(r"^([a-z][a-z-]*):", MAKEFILE, re.MULTILINE))
    mentioned = {
        match for span in code_spans() for match in re.findall(r"\bmake ([a-z][a-z-]*)", span)
    }
    assert mentioned <= targets, sorted(mentioned - targets)


def test_every_python_module_the_readme_runs_exists() -> None:
    modules = {match for span in code_spans() for match in re.findall(r"python -m ([\w.]+)", span)}
    for module in modules:
        importlib.import_module(module)  # a typo would raise here


# ---------- settings ----------


def test_every_setting_the_readme_names_exists() -> None:
    fields = {name.upper() for name in Settings.model_fields}
    named = set(re.findall(r"LLMROUTER_([A-Z0-9_]+)", README))
    assert named <= fields, sorted(named - fields)


# ---------- endpoints ----------


def route_paths() -> set[str]:
    return {route.path for route in app.routes if isinstance(route, APIRoute)}


def normalise(path: str) -> str:
    return re.sub(r"\{[^}]+\}", "{}", path)


def test_every_endpoint_in_the_api_table_exists() -> None:
    api = section("## API", "## How it works")
    if api is None:
        return
    known = {normalise(path) for path in route_paths()} | {"/metrics"}
    for line in api.splitlines():
        if not line.startswith("|"):
            continue
        first_column = line.split("|")[1]
        for path in re.findall(r"`(?:[A-Z/]+ )?(/[^`]*)`", first_column):
            for piece in re.split(r",\s*", path):
                if "…" in piece:
                    continue  # shorthand for a longer path shown in full on the same row
                assert normalise(piece.split()[-1]) in known, piece


def test_the_error_codes_the_readme_names_are_ones_the_gateway_sends() -> None:
    codes = section("Notable codes:", "\n\n")
    if codes is None:
        return
    source = "".join(p.read_text() for p in (ROOT / "app").rglob("*.py"))
    for code in re.findall(r"`([a-z_]+)`", codes):
        assert f'"{code}"' in source, code


# ---------- links and files ----------


def test_every_relative_link_points_at_a_file_that_exists() -> None:
    links = re.findall(r"\]\(([^)#]+)(?:#[^)]*)?\)", README)
    for link in links:
        if not link.startswith(("http://", "https://", "mailto:")):
            assert (ROOT / link).exists(), link


def test_every_path_in_the_layout_table_exists() -> None:
    layout = section("Project layout:")
    if layout is None:
        return
    for entry in re.findall(r"\| `([^`]+)`", layout):
        for path in entry.split(", "):
            assert (ROOT / path.strip()).exists(), path


# ---------- the numbers it quotes ----------


def test_the_benchmark_figures_the_readme_quotes_are_the_benchmarks() -> None:
    benchmark = load_benchmark()
    difficulty = benchmark.counts()["difficulty"]
    kinds = benchmark.counts()["scoring"]
    mechanical = kinds["exact"] + kinds["number"] + kinds["contains"]

    for size in re.findall(r"\b(\d+) prompts in (?:six|\d+) categories", README):
        assert int(size) == len(benchmark)
    for size in re.findall(r"benchmark is small\*{0,2} \((\d+) prompts\)", README):
        assert int(size) == len(benchmark)
    for easy, medium, hard in re.findall(r"(\d+) easy, (\d+) medium and (\d+) hard", README):
        assert (int(easy), int(medium), int(hard)) == (
            difficulty["easy"],
            difficulty["medium"],
            difficulty["hard"],
        )
    for easy, medium, hard in re.findall(r"(\d+)% easy, (\d+)% medium, (\d+)% hard", README):
        assert (int(easy), int(medium), int(hard)) == (
            round(benchmark.share("easy") * 100),
            round(benchmark.share("medium") * 100),
            round(benchmark.share("hard") * 100),
        )
    for number in re.findall(r"(\d+) prompts are checked mechanically", README):
        assert int(number) == mechanical
    for number in re.findall(r"(\d+) open-ended", README):
        assert int(number) == kinds["rubric"]


def routing_split() -> tuple[Counter[str], Counter[str]]:
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
    return overall, hard


def test_the_routing_split_the_readme_quotes_is_what_the_default_thresholds_do() -> None:
    """A claim about this code: if the scoring changes, the README must be updated."""
    overall, hard = routing_split()
    flat = re.sub(r"\s+", " ", README)
    for number in re.findall(r"(\d+) of the 90 benchmark prompts to the small tier", flat):
        assert int(number) == overall["slm"]
    for number in re.findall(r"(\d+) to the mid tier", flat):
        assert int(number) == overall["mid"]
    if "none to the frontier tier" in flat:
        assert overall["frontier"] == 0
    for number in re.findall(r"(\d+) of the 25 prompts labelled hard", flat):
        assert int(number) == hard["slm"]


def test_the_evaluation_targets_in_the_readme_are_the_ones_the_code_checks() -> None:
    from app.evaluation.evaluate import Targets

    assert Targets().cost_saved == 0.70 and Targets().quality_retained == 0.93


# ---------- honesty ----------

# A measured-sounding result: a percentage said to be saved, cheaper, retained and so on.
RESULT_CLAIM = re.compile(
    r"\b\d+(?:\.\d+)?%\+?\s+"
    r"(?:lower|cheaper|saved|savings?|reduction|retention|retained|of the quality)"
    r"|\b(?:saves?|saved|cuts?|reduces?)\s+(?:inference\s+)?(?:costs?\s+)?(?:by\s+)?\d+(?:\.\d+)?%",
    re.IGNORECASE,
)


def test_the_readme_states_no_measured_result_unless_a_report_exists() -> None:
    """A figure on cost saved or quality retained must come from a real evaluation run."""
    reports = list((ROOT / "eval_results").glob("report*.json"))
    if reports:
        return  # a real report exists, so the README may quote what it shows
    claims = [m.group(0) for m in RESULT_CLAIM.finditer(README)]
    assert claims == [], f"the README claims results but there is no report: {claims}"


def test_the_resume_table_if_there_is_one_does_not_call_the_unmeasured_numbers_verified() -> None:
    table = section("## Résumé claims", "## Limitations")
    if table is None:
        return  # no such section, no such claims
    if (ROOT / "eval_results").exists() and list((ROOT / "eval_results").glob("report*.json")):
        return
    for claim in ("70% lower inference cost", "93%+ quality retention"):
        for line in table.splitlines():
            if claim in line:
                assert "Not reproduced" in line, line
