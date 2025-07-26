from qdrant_client import QdrantClient
from qdrant_client.models import Prefetch, Filter, FieldCondition, MatchText, FusionQuery

import openai
import pickle
from minio import Minio
from langsmith import traceable, get_current_run_tree

from openai import OpenAI

from api.core.config import config
from api.ml.classifier import job_description_classifier


minio_client = Minio(
    endpoint = "minio:9000",
    access_key= "minioadmin",
    secret_key= "minioadmin",
    secure=False
)

response = minio_client.get_object('fraud-classifier', 'count_vectorizer.pkl')
obj_bytes = response.read()

COUNT_VECTORIZER = pickle.loads(obj_bytes)

response = minio_client.get_object('fraud-classifier', 'model.pkl')
obj_bytes = response.read()

MODEL = pickle.loads(obj_bytes)


@traceable(
    name="embed_query",
    run_type="embedding",
    metadata={"ls_provider": config.EMBEDDING_MODEL_PROVIDER, "ls_model_name": config.EMBEDDING_MODEL}
)
def get_embedding(text, model=config.EMBEDDING_MODEL):
    response = openai.embeddings.create(
        input=[text],
        model=model,
    )

    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": response.usage.prompt_tokens,
            "total_tokens": response.usage.total_tokens,
        }

    return response.data[0].embedding


@traceable(
    name="retrieve_top_n",
    run_type="retriever"
)
def retrieve_context(query, top_k=5):
    query_embedding = get_embedding(query)

    qdrant_client = QdrantClient(url=config.QDRANT_URL)

    results = qdrant_client.query_points(
        collection_name=config.QDRANT_COLLECTION_NAME,
        prefetch=[
            Prefetch(
                query=query_embedding,
                limit=20
            ),
            Prefetch(
                filter=Filter(
                    must=[
                        FieldCondition(
                            key="text",
                            match=MatchText(text=query)
                        )
                    ]
                ),
                limit=20
            )
        ],
        query=FusionQuery(fusion="rrf"),
        limit=top_k
    )

    retrieved_context_ids = []
    retrieved_context = []
    similarity_scores = []

    for result in results.points:
        retrieved_context_ids.append(result.id)
        retrieved_context.append(result.payload['text'])
        similarity_scores.append(result.score)

    top_context = retrieved_context[0]

    return {
        "retrieved_context_ids": retrieved_context_ids,
        "retrieved_context": retrieved_context,
        "similarity_scores": similarity_scores,
        "retrieved_job_posting": top_context
    }


@traceable(
    name="format_retrieved_context",
    run_type="prompt"
)
def process_context(context):

    formatted_context = ""

    for id, chunk in zip(context["retrieved_context_ids"], context["retrieved_context"]):
        formatted_context += f"- {id}: {chunk}\n"

    return formatted_context


def get_formatted_context(query: str, top_k: int = 5) -> dict:

    """Get the top k context, each representing an job posting for a given query.
    
    Args:
        query: The query to get the top k context for
        top_k: The number of context chunks to retrieve, works best with 5 or more
    
    Returns:
        A dictionary of the top k context chunks with IDs prepending each chunk, each representing job posting for a given query.
    """

    context = retrieve_context(query, top_k)
    # formatted_context = process_context(context)

    return context


def get_prediction(text:str) -> dict:
    """Classifies if a given job posting is real or fake.
    Uses the top retrieved context from get_formatted_context as input
    and return classification_result.

    Args:
        text: Input string of the job posting from retrieved_context from get_formatted_context
    
    Returns:
        A dictionary of the classification label and prediction probability score.
    """
    if not text:
        return {"classification_result": {"error": "Missing job description."}}

    prediction_result = job_description_classifier(MODEL, COUNT_VECTORIZER, text)
    return prediction_result