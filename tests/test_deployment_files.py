"""The Dockerfile, compose file and friends, checked without needing Docker.

They are configuration that nothing else exercises until someone deploys, so a typo here (a
setting name, a service name, a file the image needs but .dockerignore drops) would otherwise
surface only then.
"""

import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI

from app.core.config import Settings
from tests.test_db import ROOT

DOCKERFILE = (ROOT / "Dockerfile").read_text()
COMPOSE: dict[str, Any] = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
SERVICES: dict[str, dict[str, Any]] = COMPOSE["services"]
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text())
SETTING_NAMES = {name.upper() for name in Settings.model_fields}


def instructions() -> list[tuple[str, str]]:
    """The Dockerfile's instructions, with continued lines joined and comments dropped."""
    joined = re.sub(r"\\\n\s*", " ", DOCKERFILE)
    found = []
    for line in joined.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            keyword, _, rest = line.partition(" ")
            found.append((keyword.upper(), rest.strip()))
    return found


def of(keyword: str) -> list[str]:
    return [rest for kind, rest in instructions() if kind == keyword]


# ---------- the Dockerfile ----------


def test_images_are_pinned_to_a_version_not_latest() -> None:
    for rest in of("FROM"):
        image = rest.split()[0]
        assert ":" in image and not image.endswith(":latest"), image


def test_both_stages_use_the_same_python() -> None:
    """A virtualenv built by one Python must be run by the same one."""
    bases = {rest.split()[0] for rest in of("FROM")}
    assert len(bases) == 1
    assert "3.12" in bases.pop()  # the version pyproject.toml asks for
    assert PYPROJECT["project"]["requires-python"] == ">=3.12"


def test_it_is_a_multi_stage_build_that_ships_only_the_runtime() -> None:
    stages = [rest.lower() for rest in of("FROM")]
    assert len(stages) == 2 and stages[0].endswith("as build") and stages[1].endswith("as runtime")
    copies_from_build = [c for c in of("COPY") if "--from=build" in c]
    assert copies_from_build == ["--from=build /opt/venv /opt/venv"]


def test_the_gateway_does_not_run_as_root() -> None:
    kinds = [kind for kind, _ in instructions()]
    users = [rest for kind, rest in instructions() if kind == "USER"]
    assert users and users[-1] not in {"root", "0"}
    assert kinds.index("USER") < kinds.index("CMD")


def test_it_has_a_health_check_that_asks_the_health_endpoint() -> None:
    [check] = of("HEALTHCHECK")
    assert "/health" in check and "8000" in check


def test_the_command_is_in_exec_form_so_signals_reach_the_server() -> None:
    [command] = of("CMD")
    assert command.startswith("[") and "uvicorn" in command and "app.main:app" in command


def test_the_server_listens_on_every_interface_on_the_port_the_image_exposes() -> None:
    [command] = of("CMD")
    assert '"0.0.0.0"' in command and '"8000"' in command
    assert of("EXPOSE") == ["8000"]


def test_no_secret_is_baked_in() -> None:
    for kind, rest in instructions():
        if kind in {"ENV", "ARG"}:
            assert not re.search(r"KEY|SECRET|PASSWORD|TOKEN", rest, re.IGNORECASE), rest
    copied = " ".join(of("COPY"))
    for forbidden in (".env", "config.yaml", "tests", ".git"):
        assert not re.search(rf"(^|\s){re.escape(forbidden)}(\s|$)", copied), forbidden


def test_the_runtime_gets_the_code_the_migration_and_the_example_config() -> None:
    copied = " ".join(of("COPY"))
    for needed in ("app ./app", "migrations ./migrations", "alembic.ini", "config.example.yaml"):
        assert needed in copied


def test_dependencies_are_installed_from_pyproject_where_every_one_is_pinned() -> None:
    assert "pyproject.toml" in " ".join(of("COPY")) and "tomllib" in " ".join(of("RUN"))
    dependencies = PYPROJECT["project"]["dependencies"]
    unpinned = [d for d in dependencies if "==" not in d]
    assert unpinned == [], "an unpinned dependency makes two builds of one commit differ"


def test_the_runtime_dependencies_the_deployment_needs_are_declared() -> None:
    names = {re.split(r"[=\[<>]", d)[0] for d in PYPROJECT["project"]["dependencies"]}
    assert {"uvicorn", "asyncpg", "aiosqlite", "alembic", "redis", "prometheus-client"} <= names


def test_what_the_command_starts_is_the_application() -> None:
    from app.main import app

    assert isinstance(app, FastAPI)
    assert "/health" in {route.path for route in app.routes}  # type: ignore[attr-defined]


# ---------- .dockerignore ----------


def ignored() -> set[str]:
    lines = (ROOT / ".dockerignore").read_text().splitlines()
    return {line.strip() for line in lines if line.strip() and not line.startswith("#")}


def test_secrets_and_local_state_stay_out_of_the_build_context() -> None:
    assert {".env", ".git", ".venv", "tests", "*.db", "config.yaml", "eval_results"} <= ignored()


def test_the_example_env_file_is_the_one_exception() -> None:
    assert "!.env.example" in ignored() and ".env.*" in ignored()


def test_nothing_the_dockerfile_copies_is_ignored() -> None:
    for needed in ("app", "migrations", "alembic.ini", "config.example.yaml", "pyproject.toml"):
        assert needed not in ignored(), needed
        assert (ROOT / needed).exists(), needed


# ---------- docker compose ----------


def test_the_stack_has_the_four_services() -> None:
    assert set(SERVICES) == {"postgres", "redis", "migrate", "app"}


def test_every_image_is_pinned() -> None:
    for name, service in SERVICES.items():
        image = service["image"]
        assert ":" in image and not image.endswith(":latest"), (name, image)


def test_the_gateway_starts_only_after_the_database_is_migrated_and_redis_is_up() -> None:
    needs = SERVICES["app"]["depends_on"]
    assert needs["migrate"]["condition"] == "service_completed_successfully"
    assert needs["redis"]["condition"] == "service_healthy"
    assert SERVICES["migrate"]["depends_on"]["postgres"]["condition"] == "service_healthy"


def test_the_migration_runs_once_and_is_not_restarted() -> None:
    migrate = SERVICES["migrate"]
    assert migrate["command"] == ["alembic", "upgrade", "head"]
    assert migrate["restart"] == "no"
    assert migrate["image"] == SERVICES["app"]["image"]  # the same code that will run against it


def test_the_stores_have_health_checks_for_the_ordering_above_to_mean_anything() -> None:
    for name in ("postgres", "redis"):
        assert "healthcheck" in SERVICES[name], name
    assert "pg_isready" in " ".join(SERVICES["postgres"]["healthcheck"]["test"])


def test_the_gateway_talks_to_the_other_services_by_name_with_the_right_drivers() -> None:
    environment = SERVICES["app"]["environment"]
    assert environment["LLMROUTER_DATABASE_URL"].startswith("postgresql+asyncpg://")
    assert "@postgres:5432/" in environment["LLMROUTER_DATABASE_URL"]
    assert environment["LLMROUTER_REDIS_URL"] == "redis://redis:6379/0"
    assert (
        SERVICES["migrate"]["environment"]["LLMROUTER_DATABASE_URL"]
        == (environment["LLMROUTER_DATABASE_URL"])
    )


def test_every_setting_the_compose_file_sets_exists() -> None:
    """A mistyped LLMROUTER_* name is silently ignored by the app, so check them here."""
    for name, service in SERVICES.items():
        for key in service.get("environment", {}):
            if key.startswith("LLMROUTER_"):
                assert key.removeprefix("LLMROUTER_") in SETTING_NAMES, (name, key)


def test_the_data_stores_are_not_published_beyond_the_machine() -> None:
    assert "ports" not in SERVICES["postgres"]  # reached over the compose network only
    for port in SERVICES["redis"]["ports"]:
        assert str(port).startswith("127.0.0.1:"), port


def test_state_lives_in_named_volumes() -> None:
    assert set(COMPOSE["volumes"]) == {"pg-data", "redis-data"}
    assert any(v.startswith("pg-data:") for v in SERVICES["postgres"]["volumes"])
    assert any(v.startswith("redis-data:") for v in SERVICES["redis"]["volumes"])


def test_the_model_config_is_mounted_read_only() -> None:
    [mount] = SERVICES["app"]["volumes"]
    assert mount.endswith(":ro") and "/etc/llmrouter/config.yaml" in mount
    assert SERVICES["app"]["environment"]["LLMROUTER_CONFIG_PATH"] == "/etc/llmrouter/config.yaml"


def test_provider_keys_come_from_an_optional_env_file_not_the_compose_file() -> None:
    [env_file] = SERVICES["app"]["env_file"]
    assert env_file == {"path": ".env", "required": False}


def test_the_only_password_in_the_file_is_an_overridable_development_default() -> None:
    text = (ROOT / "docker-compose.yml").read_text()
    for match in re.finditer(r"PASSWORD[:=]\s*(\S+)|:(\$\{POSTGRES_PASSWORD[^}]*\})@", text):
        value = match.group(1) or match.group(2)
        assert value.startswith("${POSTGRES_PASSWORD:-") and "dev-only" in value, value
    assert "dev-only" in text and "production" in text  # and the file says to change it


def test_redis_still_persists_its_counters() -> None:
    assert "--appendonly" in SERVICES["redis"]["command"]


# ---------- the example environment and the Makefile ----------


def test_every_setting_in_the_example_env_file_exists() -> None:
    text = (ROOT / ".env.example").read_text()
    for match in re.finditer(r"^#?\s*(LLMROUTER_[A-Z0-9_]+)=", text, re.MULTILINE):
        assert match.group(1).removeprefix("LLMROUTER_") in SETTING_NAMES, match.group(1)


def test_the_example_env_file_holds_no_real_looking_secret() -> None:
    text = (ROOT / ".env.example").read_text()
    assert not re.search(r"(sk-[A-Za-z0-9]{10,}|gsk_[A-Za-z0-9]{10,}|llmr_[A-Za-z0-9]{10,})", text)


def test_the_compose_variables_are_documented() -> None:
    text = (ROOT / ".env.example").read_text()
    for variable in ("POSTGRES_PASSWORD", "APP_PORT", "REDIS_PORT"):
        assert variable in text


def test_the_make_targets_for_the_stack_exist() -> None:
    makefile = (ROOT / "Makefile").read_text()
    phony = re.search(r"^\.PHONY:(.*)$", makefile, re.MULTILINE)
    assert phony is not None
    for target in ("docker-build", "up", "down", "logs", "test-postgres", "redis-up"):
        assert re.search(rf"^{target}:", makefile, re.MULTILINE), target
        assert target in phony.group(1).split(), target


def test_config_yaml_is_not_committed_but_the_example_is() -> None:
    gitignore = (ROOT / ".gitignore").read_text().split()
    assert "config.yaml" in gitignore and ".env" in gitignore
    assert (ROOT / "config.example.yaml").exists()


# ---------- the command the migration service runs ----------


def test_alembic_upgrades_a_database_named_only_by_the_environment(tmp_path: Path) -> None:
    """The migrate service has no alembic.ini URL: LLMROUTER_DATABASE_URL must be enough."""
    database = tmp_path / "from-env.db"
    env = {**os.environ, "LLMROUTER_DATABASE_URL": f"sqlite+aiosqlite:///{database}"}

    result = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr
    assert database.exists()
