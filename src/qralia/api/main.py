"""Application FastAPI QraLia."""

from __future__ import annotations

import os
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Callable

from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from qralia import __version__
from qralia.api.models import (
    ConversationCreateRequest,
    ConversationMessageRequest,
    ConversationMessageResponse,
    ConversationResponse,
    HealthResponse,
    InvoiceResponse,
    QuestionRequest,
    QuestionResponse,
    ReadinessResponse,
)
from qralia.api.services import ConversationService, InvoiceApplicationService, QuestionAnswerService
from qralia.domain.extracted_invoice import ExtractedInvoice
from qralia.ocr.extractor import extract_invoice

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CORPUS = ROOT / "data/processed/onee-regler-facture-residentiel/chunks.jsonl"
ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg"}
ALLOWED_CONTENT_TYPES = {"application/pdf", "image/png", "image/jpeg"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
LOGGER = logging.getLogger(__name__)


def _looks_like_allowed_file(data: bytes, suffix: str) -> bool:
    if suffix == ".pdf":
        return data.startswith(b"%PDF-")
    if suffix == ".png":
        return data.startswith(b"\x89PNG\r\n\x1a\n")
    return data.startswith(b"\xff\xd8\xff")


def _cors_origins() -> list[str]:
    value = os.getenv("QRALIA_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
    return [item.strip() for item in value.split(",") if item.strip()]


def _tesseract_available() -> bool:
    configured = os.getenv("QRALIA_TESSERACT_CMD")
    standard_windows = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    return bool((configured and Path(configured).is_file()) or shutil.which("tesseract") or standard_windows.is_file())


def create_app(
    *,
    extractor: Callable[[Path], ExtractedInvoice] = extract_invoice,
    corpus_path: Path | None = None,
    ocr_ready_override: bool | None = None,
    ai_enabled_override: bool | None = None,
) -> FastAPI:
    app = FastAPI(
        title="QraLia API",
        version=__version__,
        description="Lecture prudente de factures synthétiques et réponses fondées sur des sources officielles.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins(),
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type"],
    )
    invoices = InvoiceApplicationService(extractor=extractor)
    configured_corpus = corpus_path or Path(os.getenv("QRALIA_ONEE_CHUNKS_PATH", str(DEFAULT_CORPUS)))
    ai_enabled = (
        ai_enabled_override
        if ai_enabled_override is not None
        else os.getenv("QRALIA_AI_ENABLED", "true").casefold() in {"1", "true", "yes", "on"}
    )
    questions = QuestionAnswerService(configured_corpus, ai_enabled=ai_enabled)
    conversations = ConversationService(questions)
    app.state.invoices = invoices
    app.state.questions = questions
    app.state.conversations = conversations

    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        # Ne journaliser ni le corps de la requête ni le contenu de la facture.
        LOGGER.exception("Unhandled API error on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Erreur interne du serveur. Consultez le terminal FastAPI."},
        )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(version=__version__)

    @app.get("/ready", response_model=ReadinessResponse)
    def ready(response: Response) -> ReadinessResponse:
        ocr_ready = ocr_ready_override if ocr_ready_override is not None else _tesseract_available()
        components = {
            "api": True,
            "ocr": ocr_ready,
            "official_corpus": questions.ready,
            "rag_llm": (not ai_enabled) or questions.ai_pipeline is not None,
        }
        is_ready = all(components.values())
        if not is_ready:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        details = {key: "available" if value else "unavailable" for key, value in components.items()}
        if not ai_enabled:
            details["rag_llm"] = "disabled by configuration"
        if ocr_ready and ocr_ready_override is None:
            try:
                import pytesseract
                from qralia.ocr.extractor import _configure_tesseract, _ocr_language

                _configure_tesseract(pytesseract)
                language = _ocr_language(pytesseract)
                details["ocr"] = "available (fra)" if language == "fra" else "available (English fallback; fra model missing)"
            except (OSError, RuntimeError):
                details["ocr"] = "available binary; language check failed"
        return ReadinessResponse(
            status="ready" if is_ready else "not_ready",
            components=components,
            details=details,
        )

    @app.post("/v1/invoices", response_model=InvoiceResponse, status_code=status.HTTP_201_CREATED)
    async def upload_invoice(
        file: UploadFile = File(...),
        confirm_synthetic: bool = Form(...),
    ) -> InvoiceResponse:
        if not confirm_synthetic:
            raise HTTPException(status_code=400, detail="Seules les factures synthétiques sont autorisées dans ce MVP.")
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS or file.content_type not in ALLOWED_CONTENT_TYPES:
            raise HTTPException(status_code=415, detail="Format autorisé : PDF, PNG ou JPEG.")
        content = await file.read(MAX_UPLOAD_BYTES + 1)
        await file.close()
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Le document dépasse la limite de 10 Mo.")
        if not _looks_like_allowed_file(content, suffix):
            raise HTTPException(status_code=415, detail="Le contenu ne correspond pas au format annoncé.")
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temporary:
                temporary.write(content)
                temporary_path = Path(temporary.name)
            record = await run_in_threadpool(invoices.ingest, temporary_path)
        except (RuntimeError, ValueError, OSError) as exc:
            raise HTTPException(status_code=422, detail=f"Extraction impossible : {exc}") from exc
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        return invoices.response(record)

    @app.get("/v1/invoices/{invoice_id}", response_model=InvoiceResponse)
    def get_invoice(invoice_id: str) -> InvoiceResponse:
        record = invoices.store.get(invoice_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Facture inconnue ou session expirée.")
        return invoices.response(record)

    @app.post("/v1/invoices/{invoice_id}/questions", response_model=QuestionResponse)
    async def ask_question(invoice_id: str, request: QuestionRequest) -> QuestionResponse:
        record = invoices.store.get(invoice_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Facture inconnue ou session expirée.")
        return await run_in_threadpool(questions.answer, request.question, request.language, record)

    @app.post(
        "/v1/invoices/{invoice_id}/conversations",
        response_model=ConversationResponse,
        status_code=status.HTTP_201_CREATED,
    )
    def create_conversation(invoice_id: str, request: ConversationCreateRequest) -> ConversationResponse:
        if invoices.store.get(invoice_id) is None:
            raise HTTPException(status_code=404, detail="Facture inconnue ou session expirée.")
        return conversations.response(conversations.store.create(invoice_id, request.language))

    @app.get("/v1/conversations/{conversation_id}", response_model=ConversationResponse)
    def get_conversation(conversation_id: str) -> ConversationResponse:
        conversation = conversations.store.get(conversation_id)
        if conversation is None or invoices.store.get(conversation.invoice_id) is None:
            raise HTTPException(status_code=404, detail="Conversation inconnue ou session expirée.")
        return conversations.response(conversation)

    @app.post(
        "/v1/conversations/{conversation_id}/messages",
        response_model=ConversationMessageResponse,
    )
    async def send_conversation_message(
        conversation_id: str,
        request: ConversationMessageRequest,
    ) -> ConversationMessageResponse:
        conversation = conversations.store.get(conversation_id)
        if conversation is None:
            raise HTTPException(status_code=404, detail="Conversation inconnue ou session expirée.")
        record = invoices.store.get(conversation.invoice_id)
        if record is None:
            conversations.store.delete(conversation_id)
            raise HTTPException(status_code=404, detail="Facture inconnue ou session expirée.")
        return await run_in_threadpool(
            conversations.message, conversation, record, request.message, request.language,
        )

    @app.delete("/v1/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
    def delete_conversation(conversation_id: str) -> Response:
        if not conversations.store.delete(conversation_id):
            raise HTTPException(status_code=404, detail="Conversation inconnue ou session expirée.")
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @app.delete("/v1/invoices/{invoice_id}", status_code=status.HTTP_204_NO_CONTENT)
    def delete_invoice(invoice_id: str) -> Response:
        if not invoices.store.delete(invoice_id):
            raise HTTPException(status_code=404, detail="Facture inconnue ou session expirée.")
        conversations.store.delete_for_invoice(invoice_id)
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    return app


app = create_app()

