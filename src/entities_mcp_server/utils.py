from qdrant_client import QdrantClient
from qdrant_client.models import Prefetch, Filter, FieldCondition, MatchText, FusionQuery

import openai
import spacy 
import re

from src.entities_mcp_server.core.config import config, ENTITY_LABELS, CUSTOM_PATTERNS
from collections import defaultdict

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")


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
        retrieved_context_ids.append(result.payload['job_id'])
        retrieved_context.append(result.payload['text'])
        similarity_scores.append(result.score)

    top_context = retrieved_context[0]

    return {
        "retrieved_context_ids": retrieved_context_ids,
        "retrieved_context": retrieved_context,
        "similarity_scores": similarity_scores,
        "retrieved_job_posting": top_context
    }

def extract_entities_job_posting(text: str) -> dict:
    doc = nlp(text.lower())
    group_entities = defaultdict(set)

    # include both spacy stanard relevant entities and custom ones
    for ent in doc.ents:
        label = ent.label_
        if label in ENTITY_LABELS or label in {p['label'] for p in CUSTOM_PATTERNS}:
            group_entities[label].add(ent.text)
        
    # Convert sets to sorted lists for json serialization
    grouped_entities = {label: sorted(list(ents)) for label, ents in group_entities.items()}
    total_entities = sum(len(ents) for ents in grouped_entities.values())

    return {
        "total_entities_detected": total_entities,
        "entities_by_label": grouped_entities
    }

# @traceable(
#     name="format_context_entities",
#     run_type="prompt"
# )
def format_context_entities(context):
    md = ""

    for id, chunk in zip(context["retrieved_context_ids"], context["retrieved_context"]):
        chunk_entities = extract_entities_job_posting(chunk)

        md += f" - {id} \n"
        md += f"total entities: {chunk_entities["total_entities_detected"]} \n"
        md += f"entities by label: \n"
        for label, entities in chunk_entities["entities_by_label"].items():
            md += f"- {label}: {", ".join(entities)}\n"
        md += "--- \n"
    return md



# @traceable(
#     name="format_retrieved_context",
#     run_type="prompt"
# )
# def process_context(context):

#     formatted_context = ""

#     for id, chunk in zip(context["retrieved_context_ids"], context["retrieved_context"]):
#         formatted_context += f"- {id}: {chunk}\n"

#     return formatted_context