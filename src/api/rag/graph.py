from pydantic import BaseModel, Field
from typing import List, Dict, Any, Annotated, Optional, Union
from operator import add

from api.rag.agents import ToolCall, RAGUsedContext, Delegation, coordinator_agent_node, job_posting_qa_agent_node, classifier_agent_node
from api.rag.utils.utils import get_tool_descriptions_from_mcp_servers, qa_mcp_tool_node, classifier_mcp_tool_node
# from api.rag.tools import get_formatted_context, get_prediction
from api.core.config import config

from qdrant_client import QdrantClient

from langgraph.graph import StateGraph, START, END
# from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

import logging
import pprint

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
    classifier_iteration: int = Field(default=0)
    coordinator_iteration: int = Field(default=0)
    job_posting_qa_final_answer: bool = Field(default=False)
    classifier_final_answer: bool = Field(default=False)
    coordinator_final_answer: bool = Field(default=False)
    qa_available_tools: List[Dict[str, Any]] = []
    classifier_available_tools: List[Dict[str, Any]] = []
    qa_tool_calls: Optional[List[ToolCall]] = Field(default_factory=list)
    classifier_tool_calls: Optional[List[ToolCall]] = Field(default_factory=list)
    retrieved_context_ids: List[RAGUsedContext] = []
    # NEW fields for classifier integration
    # retrieved_job_posting: Optional[str] = ""            # stores the retrieved job posting text
    classification_result: str = ""  # store fraud classification result
    user_intent: str = ""
    plan: list[Delegation] = Field(default_factory=list)
    next_agent: str = ""
    trace_id: str = ""


#### ROUTERS
def tool_router(state: State) -> str:
    """Decide whether to continue or end"""
    print("[DEBUG] State:", state)
    print("State, iteration: ", state.iteration)
    if state.job_posting_qa_final_answer:
        return "end"
    elif state.iteration > 3:
        return "end"
    elif len(state.qa_tool_calls) > 0:
        return "tools"
    else:
        return "end"
    
def coordinator_router(state) -> str:
    """Decide whether to continue or end"""
    
    if state.coordinator_final_answer:
        return "end"
    elif state.coordinator_iteration > 6:
        return "end"
    elif state.next_agent == "job_posting_qa_agent":
        return "job_posting_qa_agent"
    elif state.next_agent == "classifier_agent":
        return "classifier_agent"
    else:
        return "end"
    
def classifier_router(state: State) -> str:
    print("[DEBUG] State:", state)
    print("State, iteration: ", state.classifier_iteration)
    if state.classifier_final_answer:
        return "end"
    elif state.classifier_iteration > 4:
        return "end"
    elif len(state.classifier_tool_calls) > 0:
        return "tools"
    else:
        return "end"


workflow = StateGraph(State)


workflow.add_edge(START, "coordinator_agent_node")

workflow.add_node("coordinator_agent_node", coordinator_agent_node)
workflow.add_node("classifier_agent_node", classifier_agent_node)
workflow.add_node("agent_node", job_posting_qa_agent_node)

workflow.add_node("qa_mcp_tool_node", qa_mcp_tool_node)
workflow.add_node("classifier_mcp_tool_node", classifier_mcp_tool_node)


workflow.add_conditional_edges(
    "coordinator_agent_node",
    coordinator_router,
    {
        "job_posting_qa_agent": "agent_node",
        "classifier_agent": "classifier_agent_node",
        "end": END
    }
)

workflow.add_conditional_edges(
    "agent_node",
    tool_router,
    {
        "tools": "qa_mcp_tool_node",
        "end": "coordinator_agent_node"
    }
)

workflow.add_conditional_edges(
    "classifier_agent_node",
    classifier_router,
    {
        "tools": "classifier_mcp_tool_node",
        "end": "coordinator_agent_node"
    }
)

workflow.add_edge("qa_mcp_tool_node", "agent_node")
workflow.add_edge("classifier_mcp_tool_node", "classifier_agent_node")


async def run_agent(question: str, thread_id: str):

    # logger.info("Tool descriptions:\n%s", pprint.pformat(tool_descriptions))
    qa_mcp_servers = ["http://items_mcp_server:8000/mcp", "http://entities_mcp_server:8000/mcp"]
    classifier_mcp_servers = ["http://classifier_mcp_server:8000/mcp", "http://items_mcp_server:8000/mcp"]

    qa_tool_descriptions = await get_tool_descriptions_from_mcp_servers(qa_mcp_servers)
    classifier_tool_descriptions = await get_tool_descriptions_from_mcp_servers(classifier_mcp_servers)

    initial_state = {
        "messages": [{"role": "user", "content": question}],
        "iteration": 0,
        "qa_available_tools": qa_tool_descriptions,
        "classifier_available_tools": classifier_tool_descriptions
    }

    graph_config = {"configurable": {"thread_id": thread_id}}

    # logger.info(config.POSTGRES_CONN_STRING)

    async with AsyncPostgresSaver.from_conn_string(config.POSTGRES_CONN_STRING) as checkpointer:
        try:
            checkpointer.setup()
        except Exception as e:
            pass

        graph = workflow.compile(checkpointer=checkpointer)

        result = await graph.ainvoke(initial_state, config=graph_config)

    return result


async def run_agent_wrapper(question: str, thread_id: str):

    qdrant_client = QdrantClient(url=config.QDRANT_URL)

    result = await run_agent(question, thread_id)
    # logger.info(result.get("answer"))

    # image_url_list = []
    # for id in result.get("retrieved_context_ids", []):
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
        "trace_id": result.get("trace_id")
        # "retrieved_images": image_url_list
    }