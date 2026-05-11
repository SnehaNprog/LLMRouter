from fastapi import FastAPI

app = FastAPI()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


class Message:
    role = str
    content = str


class ChatRequest:
    model: str
    messages: Message
