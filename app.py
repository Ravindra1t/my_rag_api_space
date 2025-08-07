from fastapi import FastAPI, Request
from pydantic import BaseModel
from huggingface_hub import hf_hub_download
import uvicorn

app = FastAPI()

# Dummy loader and model (replace with real RAG pipeline later)
def load_model():
    return lambda text: "This is a dummy RAG answer for: " + text

model = load_model()

class RAGInput(BaseModel):
    documents: str
    questions: list[str]

@app.post("/hackrx/run")
async def rag_api(input: RAGInput):
    answers = [model(q) for q in input.questions]
    return {"answers": answers}
