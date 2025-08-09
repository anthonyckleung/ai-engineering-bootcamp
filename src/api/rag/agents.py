from pydantic import BaseModel, Field
from typing import List, Optional
import instructor
from instructor.exceptions import InstructorRetryException
from openai import OpenAI
from langsmith import traceable, get_current_run_tree
from langchain_core.messages import AIMessage

from api.rag.utils.utils import lc_messages_to_regular_messages
from api.rag.utils.utils import prompt_template_config
from api.core.config import config
import json

from jinja2 import Template

import logging
import pprint

# Basic logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("app")

client = instructor.from_openai(OpenAI(api_key=config.OPENAI_API_KEY))

class ToolCall(BaseModel):
    name: str
    arguments: dict = Field(alias="parameters")
    server: Optional[str] = ""

class RAGUsedContext(BaseModel):
    id: int
    description: str

class JobPostQAResponse(BaseModel):
    answer: str
    tool_calls: List[ToolCall] = Field(default_factory=list)
    final_answer: bool = Field(default=False)
    retrieved_context_ids: List[RAGUsedContext]

class IntentRouterAgentResponse(BaseModel):
    user_intent: str
    answer: str 


### User Intent Router Node


@traceable(
    name="intent_router_agent",
    run_type="llm",
    metadata={"ls_provider": "openai", "ls_model_name": "gpt-4.1"}
)
def intent_router_agent_node(state) -> dict:

    prompt_template = prompt_template_config(config.RAG_PROMPT_TEMPLATE_PATH, "intent_router_agent")

    prompt = prompt_template.render()
    #    print("[DEBUG] Agent State: ", state.model_dump_json)
    messages = state.messages

    conversation = []   # Previous messages + tool messages

    for msg in messages:
        conversation.append(lc_messages_to_regular_messages(msg))

    client = instructor.from_openai(OpenAI())

    try:
        response, raw_response = client.chat.completions.create_with_completion(
            model="gpt-4.1",
            response_model=IntentRouterAgentResponse,
            messages=[{"role": "system", "content": prompt}, *conversation],
            temperature=0.,
        )
    except InstructorRetryException as ire:
        print("Raw LLM output that caused error:", ire.last_completion)
        raise  # or handle as needed

    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": raw_response.usage.prompt_tokens,
            "output_tokens": raw_response.usage.completion_tokens,
            "total_tokens": raw_response.usage.total_tokens,
        }
        trace_id = str(getattr(current_run, "trace_id", current_run.id))


    if response.user_intent == "job_posting_qa":
        ai_message = []
    else:
        ai_message = [AIMessage(
        content=response.answer,
    )]

    return {
        "messages": [ai_message],
        "answer": response.answer,
        "user_intent": response.user_intent,
        "trace_id": trace_id
    }


@traceable(
    name="job_posting_qa_agent",
    run_type="llm",
    metadata={"ls_provider": config.GENERATION_MODEL_PROVIDER, "ls_model_name": config.GENERATION_MODEL}
)
def job_posting_qa_agent_node(state):

    prompt_template = prompt_template_config(config.RAG_PROMPT_TEMPLATE_PATH, "job_posting_qa_agent")

    prompt = prompt_template.render(
        available_tools=state.available_tools
    )

    messages = state.messages

    conversation = []

    for msg in messages:
        conversation.append(lc_messages_to_regular_messages(msg))

    print("[DEBUG] Conversation:", conversation)
    client = instructor.from_openai(OpenAI())

    # print("[DEBUG] Prompt: ", prompt)

    response, raw_response = client.chat.completions.create_with_completion(
        model="gpt-4.1-mini",
        response_model=JobPostQAResponse,
        messages=[{"role": "system", "content": prompt}, *conversation],
        temperature=0.5,
    )

    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": raw_response.usage.prompt_tokens,
            "output_tokens": raw_response.usage.completion_tokens,
            "total_tokens": raw_response.usage.total_tokens,
        }
        trace_id = str(getattr(current_run, "trace_id", current_run.id))

    # Patch missing 'server' field in each tool_call by matching known tools list
    for tool_call in response.tool_calls:
        if not getattr(tool_call, "server", None):
            matched_tool = next((t for t in state.available_tools if t["name"] == tool_call.name), None)
            if matched_tool:
                tool_call.server = matched_tool["server"]

    if response.tool_calls and not response.final_answer:
       tool_calls = []
       for i, tc in enumerate(response.tool_calls):
          tool_calls.append({
                "id": f"call_{i}",
                "name": tc.name,
                "args": tc.arguments
          })

       ai_message = AIMessage(
          content=response.answer,
          tool_calls=tool_calls
          )
    else:
       ai_message = AIMessage(
          content=response.answer,
       )
    # Prepare tool calls for AIMessage

    # print("[DEBUG] tool_calls so far:",  response.tool_calls)

    return {
        "messages": [ai_message],
        "tool_calls": response.tool_calls,
        "iteration": state.iteration + 1,
        "answer": response.answer,
        "final_answer": response.final_answer,
        "retrieved_context_ids": response.retrieved_context_ids,
        "trace_id": trace_id
    }