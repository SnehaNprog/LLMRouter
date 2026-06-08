from app.schemas.error import ErrorResponse

OPENAI_ERROR = {
    "error": {
        "message": "Incorrect API key provided: sk-xxxx.",
        "type": "invalid_request_error",
        "param": None,
        "code": "invalid_api_key",
    }
}


def test_matches_openai_error_format() -> None:
    parsed = ErrorResponse.model_validate(OPENAI_ERROR)

    assert parsed.model_dump() == OPENAI_ERROR


def test_nullable_fields_are_serialized_as_null() -> None:
    body = ErrorResponse.of("boom", "api_error").model_dump()

    assert body == {"error": {"message": "boom", "type": "api_error", "param": None, "code": None}}


def test_of_sets_param_and_code() -> None:
    body = ErrorResponse.of("bad", "invalid_request_error", param="messages", code="missing")

    assert body.error.param == "messages"
    assert body.error.code == "missing"
