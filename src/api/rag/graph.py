from pydantic import BaseModel, Field
from typing import List, Dict, Any, Annotated, Optional, Union
from operator import add

from api.rag.agent import ToolCall, RAGUsedContext, agent_node
from api.rag.utils.utils import get_tool_descriptions_from_node, mcp_tool_node, get_tool_descriptions_from_mcp_servers
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
    final_answer: bool = Field(default=False)
    available_tools: List[Dict[str, Any]] = []
    tool_calls: Optional[List[ToolCall]] = Field(default_factory=list)
    retrieved_context_ids: List[RAGUsedContext] = []
    # NEW fields for classifier integration
    retrieved_job_posting: Optional[str] = ""            # stores the retrieved job posting text
    classification_result: Optional[Dict[str, Union[bool, float, str]]] = None  # store fraud classification result
    trace_id: str = ""


def tool_router(state: State) -> str:
    """Decide whether to continue or end"""
    
    if state.final_answer:
        return "end"
    elif state.iteration > 2:
        return "end"
    elif len(state.tool_calls) > 0:
        return "tools"
    else:
        return "end"


# tools = [get_formatted_context, get_prediction]
# tool_node = ToolNode(tools)
# classifier_node = ToolNode([get_prediction])
# tool_descriptions = get_tool_descriptions_from_node(tool_node)

# tool_descriptions = await get_tool_descriptions_from_mcp_servers(mcp_servers)



workflow = StateGraph(State)
workflow.add_node("agent_node", agent_node)
workflow.add_node("mcp_tool_node", mcp_tool_node)
# workflow.add_node("tool_node", tool_node)
# workflow.add_node("classifier_node", classifier_node)

workflow.add_edge(START, "agent_node")

# workflow.add_conditional_edges(
#     "agent_node",
#     tool_router,
#     {
#         "tools": "tool_node",
#         "classifier_node": "classifier_node",
#         "end": END,
#     }
# )

workflow.add_conditional_edges(
    "agent_node",
    tool_router,
    {
        "tools": "mcp_tool_node",
        "end": END
    }
)
workflow.add_edge("mcp_tool_node", "agent_node")
# workflow.add_edge("tool_node", "agent_node")
# workflow.add_edge("classifier_node", "agent_node")


async def run_agent(question: str, thread_id: str):

    # logger.info("Tool descriptions:\n%s", pprint.pformat(tool_descriptions))
    mcp_servers = ["http://items_mcp_server:8000/mcp", "http://classifier_mcp_server:8000/mcp"]

    tool_descriptions = await get_tool_descriptions_from_mcp_servers(mcp_servers)

    initial_state = {
        "messages": [{"role": "user", "content": question}],
        "iteration": 0,
        "available_tools": tool_descriptions
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
        "trace_id": result.get("trace_id")
        # "retrieved_images": image_url_list
    }