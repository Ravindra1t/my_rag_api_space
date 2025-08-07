from fastapi import FastAPI, Request
from pydantic import BaseModel
import requests
import os
import tempfile
import mimetypes
import fitz  # PyMuPDF for PDF
import docx2txt
import email
from email import policy
from sentence_transformers import SentenceTransformer
import faiss
from groq import Groq

# Global Setup (only once)
model = SentenceTransformer("all-MiniLM-L6-v2")
DIM = 384  # Dimension of MiniLM embeddings
index = faiss.IndexFlatL2(DIM)
doc_chunks = []
chunk_texts = []
groq_client = Groq(api_key="<your_groq_api_key>")

def download_document(url):
    response = requests.get(url)
    response.raise_for_status()
    suffix = mimetypes.guess_extension(response.headers.get("content-type", "application/octet-stream"))
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(response.content)
        return tmp.name

def extract_text(path):
    if path.endswith(".pdf"):
        doc = fitz.open(path)
        return "\n".join([page.get_text() for page in doc])
    elif path.endswith(".docx"):
        return docx2txt.process(path)
    elif path.endswith(".eml"):
        with open(path, "rb") as f:
            msg = email.message_from_binary_file(f, policy=policy.default)
        return msg.get_body(preferencelist=('plain')).get_content()
    else:
        raise ValueError("Unsupported file type")

def chunk_text(text, max_tokens=200):
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, current = [], ""
    for sent in sentences:
        if len((current + sent).split()) <= max_tokens:
            current += sent + " "
        else:
            chunks.append(current.strip())
            current = sent + " "
    if current:
        chunks.append(current.strip())
    return chunks

def build_faiss_index(text):
    global index, doc_chunks, chunk_texts
    doc_chunks = chunk_text(text)
    chunk_texts = doc_chunks.copy()
    vectors = model.encode(doc_chunks)
    index.reset()
    index.add(vectors)

def retrieve_context(question, k=3):
    question_vec = model.encode([question])
    D, I = index.search(question_vec, k)
    return "\n".join([chunk_texts[i] for i in I[0]])

def ask_llama3(context, question):
    completion = groq_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Use ONLY the following policy text to answer the question accurately and precisely."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}"}
        ]
    )
    return completion.choices[0].message.content

# FastAPI app
app = FastAPI()

class HackRxInput(BaseModel):
    documents: str
    questions: list[str]

class HackRxOutput(BaseModel):
    answers: list[str]

@app.post("/hackrx/run", response_model=HackRxOutput)
def run_hackrx(data: HackRxInput):
    path = download_document(data.documents)
    text = extract_text(path)
    os.remove(path)
    build_faiss_index(text)
    answers = []
    for q in data.questions:
        context = retrieve_context(q)
        answer = ask_llama3(context, q)
        answers.append(answer)
    return {"answers": answers}
