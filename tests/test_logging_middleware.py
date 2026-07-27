import json
import logging

from fastapi.testclient import TestClient

from app.core.logging import JsonFormatter, request_id_var


def test_response_has_generated_request_id(client: TestClient) -> None:
    response = client.get("/health")

    assert len(response.headers["X-Request-ID"]) == 32


def test_incoming_request_id_is_echoed(client: TestClient) -> None:
    response = client.get("/health", headers={"X-Request-ID": "abc-123"})

    assert response.headers["X-Request-ID"] == "abc-123"


def test_access_log_carries_request_id(client: TestClient) -> None:
    lines: list[str] = []

    class Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            lines.append(JsonFormatter().format(record))

    access_logger = logging.getLogger("app.access")
    handler = Capture()
    access_logger.addHandler(handler)
    try:
        client.get("/health", headers={"X-Request-ID": "trace-1"})
    finally:
        access_logger.removeHandler(handler)

    data = json.loads(lines[0])
    assert data["request_id"] == "trace-1"
    assert data["path"] == "/health"
    assert data["method"] == "GET"


def test_json_formatter_outputs_one_json_object() -> None:
    token = request_id_var.set("rid-9")
    try:
        record = logging.LogRecord("x", logging.INFO, __file__, 1, "hello %s", ("world",), None)
        record.user_field = "v"
        line = JsonFormatter().format(record)
    finally:
        request_id_var.reset(token)

    data = json.loads(line)
    assert "\n" not in line
    assert data["message"] == "hello world"
    assert data["level"] == "INFO"
    assert data["request_id"] == "rid-9"
    assert data["user_field"] == "v"


def test_configure_logging_is_idempotent_and_keeps_foreign_handlers() -> None:
    from app.core.logging import _JsonHandler, configure_logging

    root = logging.getLogger()
    foreign = logging.NullHandler()
    root.addHandler(foreign)
    try:
        configure_logging("INFO")
        configure_logging("INFO")

        assert foreign in root.handlers
        assert sum(isinstance(h, _JsonHandler) for h in root.handlers) == 1
    finally:
        root.removeHandler(foreign)
