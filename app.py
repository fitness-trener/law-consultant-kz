"""HTTP-слой: страницы, консультация, анализ документов."""
from __future__ import annotations

import json
import logging
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Iterator

from fastapi import Depends, FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import analyzer
import db
import llm
import prompts
import retriever
from config import ROOT, load_settings
from docparse import DocParseError, parse_document

SETTINGS = load_settings()
logger = logging.getLogger(__name__)
app = FastAPI(title="Юридический консультант по НПА РК")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
templates = Jinja2Templates(directory=ROOT / "templates")

# Shown for any unexpected failure on a user-facing page: never the actual
# exception text (that goes to the log only), always Russian, always paired
# with prompts.DISCLAIMER by the caller.
GENERIC_ERROR = "Произошла непредвиденная ошибка на сервере. Повторите запрос позже."


def get_connection() -> sqlite3.Connection:
    conn = db.connect()
    db.create_schema(conn)
    return conn


def db_conn() -> Iterator[sqlite3.Connection]:
    """Shared try/finally discipline for routes that need a DB connection:
    guarantees conn.close() runs even if the route raises."""
    conn = get_connection()
    try:
        yield conn
    finally:
        conn.close()


def answer_question(conn: sqlite3.Connection, question: str) -> dict:
    hits = retriever.search(conn, question, k=4)
    if not hits:
        return {"answer": prompts.REFUSAL, "sources": [], "refused": True}
    answer = llm.complete(prompts.build_consultation_messages(question, hits))
    return {"answer": answer, "sources": hits, "refused": False}


@app.get("/", response_class=HTMLResponse)
def index(request: Request, conn: sqlite3.Connection = Depends(db_conn)):
    try:
        acts = db.corpus_summary(conn)
    except Exception:
        logger.exception("Ошибка получения списка НПА")
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "acts": [], "error": GENERIC_ERROR, "disclaimer": prompts.DISCLAIMER},
        )
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "acts": acts, "disclaimer": prompts.DISCLAIMER},
    )


@app.post("/chat", response_class=HTMLResponse)
def chat(request: Request, question: str = Form(...), conn: sqlite3.Connection = Depends(db_conn)):
    question = question.strip()
    if not question:
        return templates.TemplateResponse(
            "_answer.html",
            {"request": request, "error": "Введите вопрос.", "disclaimer": prompts.DISCLAIMER},
        )
    try:
        result = answer_question(conn, question)
    except llm.LLMError as exc:
        return templates.TemplateResponse(
            "_answer.html",
            {"request": request, "error": str(exc), "disclaimer": prompts.DISCLAIMER},
        )
    except Exception:
        logger.exception("Ошибка при обработке вопроса")
        return templates.TemplateResponse(
            "_answer.html",
            {"request": request, "error": GENERIC_ERROR, "disclaimer": prompts.DISCLAIMER},
        )
    return templates.TemplateResponse(
        "_answer.html",
        {
            "request": request,
            "question": question,
            "result": result,
            "disclaimer": prompts.DISCLAIMER,
        },
    )


ALLOWED_SUFFIXES = {".docx", ".pdf"}
JOBS: dict[str, dict] = {}

# Uploaded contracts sit on disk until the events stream deletes them in its
# finally block. A client that uploads and never opens the stream would leave
# the file there forever, so every new upload sweeps anything older than this.
STALE_UPLOAD_RETENTION_SECONDS = 60 * 60  # 1 hour


def _sweep_stale_uploads() -> None:
    if not SETTINGS.upload_dir.exists():
        return
    # Never touch a path a tracked job still owns, no matter its age.
    active_paths = {job["path"] for job in JOBS.values()}
    cutoff = time.time() - STALE_UPLOAD_RETENTION_SECONDS
    for entry in SETTINGS.upload_dir.iterdir():
        if entry in active_paths or not entry.is_file():
            continue
        try:
            if entry.stat().st_mtime < cutoff:
                entry.unlink()
        except OSError:
            # ponytail: best-effort cleanup — one bad file must not fail the upload.
            logger.exception("Не удалось удалить устаревший файл %s", entry)


@app.post("/analyze")
async def analyze(file: UploadFile):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(400, "Поддерживаются DOCX и PDF.")

    payload = await file.read()
    limit = SETTINGS.max_upload_mb * 1024 * 1024
    if len(payload) > limit:
        raise HTTPException(
            400,
            f"Файл {len(payload) / 1024 / 1024:.1f} МБ превышает предел "
            f"{SETTINGS.max_upload_mb} МБ.",
        )

    SETTINGS.upload_dir.mkdir(parents=True, exist_ok=True)
    _sweep_stale_uploads()
    job_id = uuid.uuid4().hex
    path = SETTINGS.upload_dir / f"{job_id}{suffix}"
    path.write_bytes(payload)
    JOBS[job_id] = {
        "path": path,
        "filename": file.filename,
        "findings": None,
        "error": None,
        "clauses": 0,
    }
    return {"job_id": job_id}


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.get("/analyze/{job_id}/events")
def analyze_events(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Задача не найдена.")

    def stream() -> Iterator[str]:
        # conn is created *inside* the try (not before it) so that a failure
        # in get_connection() itself still hits the finally below — the
        # uploaded contract must never survive a failed job.
        conn: sqlite3.Connection | None = None
        try:
            conn = get_connection()
            clauses = parse_document(job["path"], SETTINGS.max_clauses)
            job["clauses"] = len(clauses)
            yield _sse("progress", {"done": 0, "total": len(clauses)})

            events: list[str] = []
            job["findings"] = analyzer.analyze_document(
                conn,
                clauses,
                progress=lambda done, total: events.append(
                    _sse("progress", {"done": done, "total": total})
                ),
            )
            yield from events
            yield _sse("done", {"findings": len(job["findings"])})
        except (DocParseError, llm.LLMError) as exc:
            job["error"] = str(exc)
            yield _sse("error", {"message": str(exc)})
        except Exception:
            # Any other failure (DB fault, analyzer bug, disk error, ...) must
            # never fall through silently: job["findings"] would stay None and
            # the report template would render the reassuring "no risks found"
            # message over a document that was never actually checked.
            logger.exception("Ошибка анализа документа job_id=%s", job_id)
            message = (
                "Не удалось завершить анализ документа из-за внутренней ошибки. "
                "Документ проверен не полностью, результатам доверять нельзя."
            )
            job["error"] = message
            yield _sse("error", {"message": message})
        finally:
            if conn is not None:
                conn.close()
            job["path"].unlink(missing_ok=True)

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/analyze/{job_id}/report", response_class=HTMLResponse)
def analyze_report(request: Request, job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Задача не найдена.")
    return templates.TemplateResponse(
        "_report.html",
        {
            "request": request,
            "filename": job["filename"],
            "clauses": job["clauses"],
            "findings": job["findings"],
            "error": job["error"],
            "disclaimer": prompts.DISCLAIMER,
        },
    )
