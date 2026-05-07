from fastapi import FastAPI

app = FastAPI()


@app.get("/health")
def health():
    return {"status": "ok"}

class message :
    role = str
    content = str

class ChatRequest :
    model : str
    messages : message