import os
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import yaml

from app.loadtest.stub import StubSettings

ROOT = Path(__file__).resolve().parents[2]


class HarnessError(Exception):
    """The system under test could not be started or driven."""


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@dataclass
class StackConfig:
    """What to start: a stub model server and a gateway in front of it."""

    stub: StubSettings = field(default_factory=StubSettings)
    redis_url: str | None = None  # None: no Redis, so quotas fail open (see the notes)
    database_url: str | None = None  # None: a SQLite file in a temporary directory
    gateway_env: dict[str, str] = field(default_factory=dict)  # extra LLMROUTER_* settings
    ready_timeout: float = 40.0


@dataclass
class Stack:
    stub_url: str
    gateway_url: str
    admin_key: str
    gateway_pid: int
    workdir: Path

    def make_key(
        self,
        *,
        rpm_limit: int | None = None,
        tpm_limit: int | None = None,
        monthly_budget_usd: float | None = None,
        name: str = "load test",
    ) -> str:
        """A new key for the platform tenant that may call models, optionally with limits."""
        headers = {"Authorization": f"Bearer {self.admin_key}"}
        with httpx.Client(base_url=self.gateway_url, headers=headers, timeout=30) as client:
            tenants = client.get("/admin/tenants").json()["data"]
            tenant_id = tenants[0]["id"]
            body = {
                "name": name,
                "scopes": ["chat:write", "models:read", "usage:read"],
                "rpm_limit": rpm_limit,
                "tpm_limit": tpm_limit,
                "monthly_budget_usd": monthly_budget_usd,
            }
            response = client.post(
                f"/admin/tenants/{tenant_id}/keys",
                json={k: v for k, v in body.items() if v is not None},
            )
        if response.status_code != 201:
            raise HarnessError(f"could not create a key: {response.status_code} {response.text}")
        return str(response.json()["secret"])

    def stub_stats(self) -> dict[str, int]:
        return dict(httpx.get(f"{self.stub_url}/__stub/stats", timeout=10).json())

    def stub_reset(self) -> None:
        httpx.post(f"{self.stub_url}/__stub/reset", timeout=10)

    def stub_configure(self, **changes: float) -> None:
        httpx.post(f"{self.stub_url}/__stub/config", json=changes, timeout=10).raise_for_status()

    def gateway_process_stats(self) -> dict[str, float] | None:
        """The gateway process's memory (MB) and CPU time (s) so far, if ``ps`` can tell us."""
        try:
            out = subprocess.run(
                ["ps", "-o", "rss=,time=", "-p", str(self.gateway_pid)],
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.split()
            rss_kb, cpu = float(out[0]), out[1]
        except (OSError, ValueError, IndexError, subprocess.SubprocessError):
            return None
        return {"rss_mb": round(rss_kb / 1024, 1), "cpu_seconds": round(_cpu_seconds(cpu), 2)}


def _cpu_seconds(text: str) -> float:
    """CPU time as ``ps`` prints it: ``[[days-]hours:]minutes:seconds``."""
    days, _, clock = text.rpartition("-")
    seconds = 0.0
    for part in clock.split(":"):
        seconds = seconds * 60 + float(part)
    return seconds + (float(days) * 86_400 if days else 0.0)


def model_config_yaml(stub_url: str) -> str:
    """Two models behind the stub, one per tier, so ``auto`` has something to choose between."""
    return yaml.safe_dump(
        {
            "models": [
                {
                    "name": "small",
                    "tier": "slm",
                    "provider": "openai_compatible",
                    "model": "stub-small",
                    "base_url": f"{stub_url}/v1",
                    "context_window": 32000,
                    "input_cost_per_mtok": 0.05,
                    "output_cost_per_mtok": 0.08,
                },
                {
                    "name": "big",
                    "tier": "frontier",
                    "provider": "openai_compatible",
                    "model": "stub-big",
                    "base_url": f"{stub_url}/v1",
                    "context_window": 128000,
                    "input_cost_per_mtok": 3.0,
                    "output_cost_per_mtok": 15.0,
                },
            ],
            "fallbacks": {"small": ["big"]},
        },
        sort_keys=False,
    )


def gateway_environment(
    config: StackConfig, config_path: Path, database_url: str
) -> dict[str, str]:
    env = {
        **os.environ,
        "LLMROUTER_CONFIG_PATH": str(config_path),
        "LLMROUTER_DATABASE_URL": database_url,
        "LLMROUTER_LOG_LEVEL": "WARNING",  # a log line per request would be measured too
    }
    if config.redis_url:
        env["LLMROUTER_REDIS_URL"] = config.redis_url
    env.update(config.gateway_env)
    return env


def _wait(url: str, process: subprocess.Popen[bytes], log: Path, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            tail = log.read_text(errors="replace")[-1500:]
            raise HarnessError(f"{url} exited early (code {process.returncode}):\n{tail}")
        try:
            if httpx.get(url, timeout=2).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.25)
    raise HarnessError(f"{url} was not ready after {timeout:.0f}s:\n{log.read_text()[-1500:]}")


def _run(command: list[str], env: dict[str, str]) -> str:
    result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise HarnessError(f"{' '.join(command)} failed:\n{result.stderr[-1500:]}")
    return result.stdout


@contextmanager
def running_stack(config: StackConfig | None = None) -> Iterator[Stack]:
    """Start the stub and the gateway as separate processes, yield them, and stop them after."""
    config = config or StackConfig()
    with tempfile.TemporaryDirectory(prefix="llmrouter-load-") as directory:
        workdir = Path(directory)
        stub_port, gateway_port = free_port(), free_port()
        stub_url, gateway_url = f"http://127.0.0.1:{stub_port}", f"http://127.0.0.1:{gateway_port}"
        database_url = config.database_url or f"sqlite+aiosqlite:///{workdir / 'load.db'}"
        config_path = workdir / "config.yaml"
        config_path.write_text(model_config_yaml(stub_url))
        env = gateway_environment(config, config_path, database_url)

        _run([sys.executable, "-m", "alembic", "upgrade", "head"], env)
        admin_key = (
            _run([sys.executable, "-m", "app.cli", "create-admin"], env).strip().splitlines()[-1]
        )

        processes: list[subprocess.Popen[bytes]] = []
        try:
            stub_log, gateway_log = workdir / "stub.log", workdir / "gateway.log"
            stub = subprocess.Popen(
                [
                    sys.executable, "-m", "app.loadtest", "stub", "--port", str(stub_port),
                    "--latency-ms", str(config.stub.latency_ms),
                    "--jitter-ms", str(config.stub.jitter_ms),
                    "--error-rate", str(config.stub.error_rate),
                    "--completion-tokens", str(config.stub.completion_tokens),
                ],
                cwd=ROOT, env=env, stdout=stub_log.open("wb"), stderr=subprocess.STDOUT,
            )  # fmt: skip
            processes.append(stub)
            _wait(f"{stub_url}/health", stub, stub_log, config.ready_timeout)
            gateway = subprocess.Popen(
                [
                    sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1",
                    "--port", str(gateway_port), "--log-level", "warning",
                ],
                cwd=ROOT, env=env, stdout=gateway_log.open("wb"), stderr=subprocess.STDOUT,
            )  # fmt: skip
            processes.append(gateway)
            _wait(f"{gateway_url}/ready", gateway, gateway_log, config.ready_timeout)
            yield Stack(stub_url, gateway_url, admin_key, gateway.pid, workdir)
        finally:
            for process in reversed(processes):
                process.terminate()
            for process in processes:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
