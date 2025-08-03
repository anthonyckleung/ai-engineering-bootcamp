from qdrant_client import QdrantClient
from qdrant_client.models import Prefetch, Filter, FieldCondition, MatchText, FusionQuery

import openai

from src.items_mcp_server.core.config import config



# @traceable(
#     name="embed_query",
#     run_type="embedding",
#     metadata={"ls_provider": config.EMBEDDING_MODEL_PROVIDER, "ls_model_name": config.EMBEDDING_MODEL}
# )
def get_embedding(text, model=config.EMBEDDING_MODEL):
    response = openai.embeddings.create(
        input=[text],
        model=model,
    )

    # current_run = get_current_run_tree()
    # if current_run:
    #     current_run.metadata["usage_metadata"] = {
    #         "input_tokens": response.usage.prompt_tokens,
    #         "total_tokens": response.usage.total_tokens,
    #     }

    return response.data[0].embedding


# @traceable(
#     name="retrieve_top_n",
#     run_type="retriever"
# )
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


# @traceable(
#     name="format_retrieved_context",
#     run_type="prompt"
# )
def process_context(context):

    formatted_context = ""

    for id, chunk in zip(context["retrieved_context_ids"], context["retrieved_context"]):
        formatted_context += f"- {id}: {chunk}\n"

    return formatted_context