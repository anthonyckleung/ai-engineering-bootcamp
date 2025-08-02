from fastapi import APIRouter, Request
import logging

from api.processors.submit_feedback import submit_feedback
# from api.rag.retrieval import rag_pipeline_wrapper
from api.rag.graph import run_agent_wrapper

from api.api.models import RAGRequest, RAGResponse, FeedBackRequest, FeedBackResponse


logger = logging.getLogger(__name__)

rag_router = APIRouter()
feedback_router = APIRouter()


@rag_router.post("/rag")
async def rag(
    request: Request,
    payload: RAGRequest
) -> RAGResponse:
    # logger.info(payload.query)
    result = run_agent_wrapper(payload.query, payload.thread_id)
    # used_image_urls = [RAGUsedImage(image_url=image["image_url"], price=image["price"], description=image["description"]) for image in result["retrieved_images"]]

    return RAGResponse(
        request_id=request.state.request_id,
        answer=result["answer"],
        trace_id=result["trace_id"]
        # used_image_urls=used_image_urls
    )

@feedback_router.post("/submit_feedback")
async def send_feedback(
    request: Request,
    payload: FeedBackRequest
) -> FeedBackResponse:


    ##### Implement feedback submission
    submit_feedback(payload.trace_id, payload.feedback_score, payload.feedback_text, payload.feedback_source_type)

    return FeedBackResponse(
        request_id=request.state.request_id,
        status="success"
    )


api_router = APIRouter()
api_router.include_router(rag_router, tags=["rag"])
api_router.include_router(feedback_router, tags=["feedback"])