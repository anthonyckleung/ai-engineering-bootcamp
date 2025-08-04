from fastmcp import FastMCP
from minio import Minio
import pickle
from src.classifier_mcp_server.utils import job_description_classifier, format_classification_output
from typing import Literal, Dict, Any, Annotated, List, Optional

mcp = FastMCP("items")

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


@mcp.tool()
def get_prediction(text:str) -> Dict[str, Any]:
    """Classifies if a given job posting is real or fake.
    Uses the top retrieved job posting from get_formatted_context as input
    and return the classification result.

    Args:
        text: Input string of the job posting
    
    Returns:
        A dictionary of the classification label and prediction probability score.
    """
    if not text:
        return "classification_result: error - Missing job description"

    prediction_result = job_description_classifier(MODEL, COUNT_VECTORIZER, text)
    # formatted_result = format_classification_output(prediction_result['is_fraud'], prediction_result['probability'])
    return prediction_result


if __name__ == "__main__":
    mcp.run(transport='http', host="0.0.0.0", port=8000)