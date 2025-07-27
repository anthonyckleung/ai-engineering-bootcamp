from pydantic import BaseModel, Field
from typing import List, Dict, Any, Annotated, Optional, Union
from operator import add

from api.rag.agent import ToolCall, RAGUsedContext, agent_node
from api.rag.utils.utils import get_tool_descriptions_from_node
from api.rag.tools import get_formatted_context, get_prediction
from api.core.config import config

from qdrant_client import QdrantClient

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.postgres import PostgresSaver

import logging

# Basic logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("app")


class State(BaseModel):
    messages: Annotated[List[Any], add] = []
    answer: str = ""
    iteration: int = Field(default=0)
    final_answer: bool = Field(default=False)
    available_tools: List[Dict[str, Any]] = []
    tool_calls: Optional[List[ToolCall]] = Field(default_factory=list)
    retrieved_context_ids: List[RAGUsedContext] = []
    # NEW fields for classifier integration
    retrieved_job_posting: Optional[str] = None            # stores the retrieved job posting text
    classification_result: Optional[Dict[str, Union[bool, float, str]]] = None  # store fraud classification result


def tool_router(state: State) -> str:
    """Decide whether to continue or end"""
    
    if state.final_answer or state.iteration > 3:
        return "end"

    if state.tool_calls:
        last_tool = state.tool_calls[-1].name
        if last_tool == "get_formatted_context":
            return "tools"
        elif last_tool == "get_prediction":
            return "classifier_node"

    return "end"


tools = [get_formatted_context, get_prediction]
tool_node = ToolNode(tools)
classifier_node = ToolNode([get_prediction])

tool_descriptions = get_tool_descriptions_from_node(tool_node)

workflow = StateGraph(State)
workflow.add_node("agent_node", agent_node)
workflow.add_node("tool_node", tool_node)
workflow.add_node("classifier_node", classifier_node)

workflow.add_edge(START, "agent_node")

workflow.add_conditional_edges(
    "agent_node",
    tool_router,
    {
        "tools": "tool_node",
        "classifier_node": "classifier_node",
        "end": END,
    }
)

workflow.add_edge("tool_node", "agent_node")
workflow.add_edge("classifier_node", "agent_node")


def run_agent(question: str, thread_id: str):

    initial_state = {
        "messages": [{"role": "user", "content": question}],
        "iteration": 0,
        "available_tools": tool_descriptions
    }

    graph_config = {"configurable": {"thread_id": thread_id}}

    graph = workflow.compile()
    result = graph.invoke(initial_state)

    logger.info(config.POSTGRES_CONN_STRING)

    with PostgresSaver.from_conn_string(config.POSTGRES_CONN_STRING) as checkpointer:
        try:
            checkpointer.setup()
        except Exception as e:
            pass

        graph = workflow.compile(checkpointer=checkpointer)

        result = graph.invoke(initial_state, config=graph_config)

    return result


def run_agent_wrapper(question: str, thread_id: str):

    qdrant_client = QdrantClient(url=config.QDRANT_URL)

    result = run_agent(question, thread_id)
    logger.info(result.get("answer"))

    # image_url_list = []
    # for id in result.get("retrieved_context_ids"):
    #     payload = qdrant_client.retrieve(
    #         collection_name=config.QDRANT_COLLECTION_NAME,
    #         ids=[id.id]
    #     )[0].payload
        # image_url = payload.get("first_large_image")
        # price = payload.get("price")
        # if image_url:
        #     image_url_list.append({"image_url": image_url, "price": price, "description": id.description})

    return {
        "answer": result.get("answer"),
        # "retrieved_images": image_url_list
    }