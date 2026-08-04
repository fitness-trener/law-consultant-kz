"""HTTP-слой: страницы, консультация, анализ документов."""
from __future__ import annotations

import sqlite3

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import db
import llm
import prompts
import retriever
from config import ROOT, load_settings

SETTINGS = load_settings()
app = FastAPI(title="Юридический консультант по НПА РК")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
templates = Jinja2Templates(directory=ROOT / "templates")


def get_connection() -> sqlite3.Connection:
    conn = db.connect()
    db.create_schema(conn)
    return conn


def answer_question(conn: sqlite3.Connection, question: str) -> dict:
    hits = retriever.search(conn, question, k=4)
    if not hits:
        return {"answer": prompts.REFUSAL, "sources": [], "refused": True}
    answer = llm.complete(prompts.build_consultation_messages(question, hits))
    return {"answer": answer, "sources": hits, "refused": False}


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    conn = get_connection()
    acts = db.corpus_summary(conn)
    conn.close()
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "acts": acts, "disclaimer": prompts.DISCLAIMER},
    )


@app.post("/chat", response_class=HTMLResponse)
def chat(request: Request, question: str = Form(...)):
    question = question.strip()
    if not question:
        return templates.TemplateResponse(
            "_answer.html",
            {"request": request, "error": "Введите вопрос.", "disclaimer": prompts.DISCLAIMER},
        )
    conn = get_connection()
    try:
        result = answer_question(conn, question)
    except llm.LLMError as exc:
        return templates.TemplateResponse(
            "_answer.html",
            {"request": request, "error": str(exc), "disclaimer": prompts.DISCLAIMER},
        )
    finally:
        conn.close()
    return templates.TemplateResponse(
        "_answer.html",
        {
            "request": request,
            "question": question,
            "result": result,
            "disclaimer": prompts.DISCLAIMER,
        },
    )
