
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from Scripts.generate_answer import (
    safety_check,
    retrieve_chunks,
    generate_grounded_answer,
    TemporaryServiceUnavailable,
)

# Keep detailed errors in the terminal, not in the browser response.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="PCOSakhi API",
    description="Educational PCOS information grounded in the 2023 evidence-based guideline.",
    version="1.0.0",
)

# Allow your local React/Vite frontend to call this API.
# We can add a deployed frontend origin later.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        # Add your Vercel frontend URL here after deployment
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class ChatRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The user's question",
    )


@app.get("/api/health")
def health_check():
    """Check whether the API is running."""
    return {
        "status": "ok",
        "service": "PCOSakhi API",
    }


@app.post("/api/chat")
def chat(request: ChatRequest):
    """Retrieve guideline passages and generate a grounded answer."""

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=422,
            detail="Please enter a question.",
        )

    # Apply the safety checks already implemented in generate_answer.py.
    safety_response = safety_check(question)

    if safety_response:
        return {
            "answer": safety_response,
            "citations": [],
            "insufficient_evidence": False,
        }

    try:
        # Retrieve relevant passages from the existing Firestore index.
        chunks = retrieve_chunks(question)

        # Generate the answer using the existing grounded-answer function.
        result = generate_grounded_answer(question, chunks)

        # Preserve the response contract expected by the frontend.
        return {
            "answer": result["answer"],
            "citations": result.get("citations", []),
            "insufficient_evidence": result.get(
                "insufficient_evidence", False
            ),
        }

    except TemporaryServiceUnavailable:
        logger.warning("Gemini service temporarily unavailable.")
        raise HTTPException(
            status_code=503,
            detail="The service is temporarily busy. Please try again shortly.",
        )

    except HTTPException:
        raise

    except Exception:
        logger.exception("Unexpected error while answering a question.")
        raise HTTPException(
            status_code=500,
            detail="Something went wrong while processing your question.",
        )
