from pydantic import BaseModel, Field
from typing import List
import instructor
from litellm import completion
from openai import OpenAI
from langsmith import traceable, get_current_run_tree
from langchain_core.messages import AIMessage

from api.rag.utils.utils import lc_messages_to_regular_messages, format_ai_message
from api.rag.utils.utils import prompt_template_config
from api.core.config import config
import json

from jinja2 import Template

import logging

# client = instructor.from_openai(OpenAI(api_key=config.OPENAI_API_KEY))
client = instructor.from_litellm(completion)

# Basic logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("app")



class ToolCall(BaseModel):
    name: str
    arguments: dict = Field(alias="parameters")
    server: str

class RAGUsedContext(BaseModel):
    id: int
    description: str

class JobPostQAResponse(BaseModel):
    answer: str
    tool_calls: List[ToolCall] = Field(default_factory=list)
    final_answer: bool = Field(default=False)
    retrieved_context_ids: List[RAGUsedContext]

class ClassifierAgentResponse(BaseModel):
    answer: str
    tool_calls: List[ToolCall] = Field(default_factory=list)
    final_answer: bool = Field(default=False)
    retrieved_context_ids: List[RAGUsedContext]

class Delegation(BaseModel):
    agent: str
    task: str = Field(default="")

class CoordinatorAgentResponse(BaseModel):
    next_agent: str
    plan: list[Delegation]
    final_answer: bool = Field(default=False)
    answer: str


### Coordinator Router Node


@traceable(
    name="coordinator_agent",
    run_type="llm",
    metadata={"ls_provider": "openai", "ls_model_name": "gpt-4.1"}
)
def coordinator_agent_node(state, models = ["gpt-4.1", "gpt-4.1-mini"]) :

    prompts = {}

    for model in models:
        prompt_template = prompt_template_config(config.COORDINATOR_AGENT_PROMPT_TEMPLATE_PATH, model)
        prompt = prompt_template.render()
        prompts[model] = prompt

    #    print("[DEBUG] Agent State: ", state.model_dump_json)
    messages = state.messages

    conversation = []   # Previous messages + tool messages

    for msg in messages:
        conversation.append(lc_messages_to_regular_messages(msg))


    for model in models:
        try:
            print(f"[DEBUG] model: {model}")
            response, raw_response = client.chat.completions.create_with_completion(
                model=model,
                response_model=CoordinatorAgentResponse,
                messages=[{"role": "system", "content": prompts[model]}, *conversation],
                temperature=0,
            )
            break
        except Exception as e:
            print(f"Error with model {model}: {e}")
            continue

    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": raw_response.usage.prompt_tokens,
            "output_tokens": raw_response.usage.completion_tokens,
            "total_tokens": raw_response.usage.total_tokens,
        }
        trace_id = str(getattr(current_run, "trace_id", current_run.id))

    # logger.info(f"Trace id from Intent Agent: {trace_id}")


    if response.final_answer:
        ai_message = [AIMessage(
            content=response.answer,
        )]
    else:
        ai_message = []

    return {
        "messages": ai_message,
        "answer": response.answer,
        "next_agent": response.next_agent,
        "plan": response.plan,
        "coordinator_final_answer": response.final_answer,
        "coordinator_iteration": state.coordinator_iteration + 1,
        "trace_id": trace_id
    }


@traceable(
    name="job_posting_qa_agent",
    run_type="llm",
    metadata={"ls_provider": config.GENERATION_MODEL_PROVIDER, "ls_model_name": config.GENERATION_MODEL}
)
def job_posting_qa_agent_node(state, models = ["gpt-4.1", "gpt-4.1-mini"]):

    prompts = {}

    for model in models:
        prompt_template = prompt_template_config(config.JOB_POSTING_QA_AGENT_PROMPT_TEMPLATE_PATH, model)
        prompt = prompt_template.render(available_tools=state.qa_available_tools)
        prompts[model] = prompt

    messages = state.messages
    conversation = []

    for msg in messages:
        conversation.append(lc_messages_to_regular_messages(msg))

    for model in models:
        try:
            response, raw_response = client.chat.completions.create_with_completion(
                model=model,
                response_model=JobPostQAResponse,
                messages=[{"role": "system", "content": prompts[model]}, *conversation],
                temperature=0,
            )
            break
        except Exception as e:
            print(f"Error with model {model}: {e}")
            continue

    logger.info(f"[DEBUG] Raw response from QA Agent: {raw_response}")

    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": raw_response.usage.prompt_tokens,
            "output_tokens": raw_response.usage.completion_tokens,
            "total_tokens": raw_response.usage.total_tokens,
        }
        # trace_id = str(getattr(current_run, "trace_id", current_run.id))

    # Patch missing 'server' field in each tool_call by matching known tools list
    for tool_call in response.tool_calls:
        if not getattr(tool_call, "server", None):
            matched_tool = next((t for t in state.qa_available_tools if t["name"] == tool_call.name), None)
            if matched_tool:
                tool_call.server = matched_tool["server"]

    ai_message = format_ai_message(response)
    # Prepare tool calls for AIMessage
    logger.info(f"[DEBUG]: Tool calls so far: {response.tool_calls}")

    return {
        "messages": [ai_message],
        "qa_tool_calls": response.tool_calls,
        "iteration": state.iteration + 1,
        "answer": response.answer,
        "job_posting_qa_final_answer": response.final_answer,
        "retrieved_context_ids": response.retrieved_context_ids,
        # "trace_id": trace_id
    }


@traceable(
    name="classifier_agent",
    run_type="llm",
    metadata={"ls_provider": "openai", "ls_model_name": "gpt-4.1"}
)
def classifier_agent_node(state, models = ["gpt-4.1", "gpt-4.1-mini"]) -> dict:

    prompts = {}

    for model in models:
        prompt_template = prompt_template_config(config.CLASSIFIER_AGENT_PROMPT_TEMPLATE_PATH, model)
        prompt = prompt_template.render(available_tools=state.classifier_available_tools)
        prompts[model] = prompt
    messages = state.messages

    conversation = []   # Previous messages + tool messages

    for msg in messages:
        conversation.append(lc_messages_to_regular_messages(msg))

    from instructor.exceptions import InstructorRetryException

    for model in models:
        try:
            response, raw_response = client.chat.completions.create_with_completion(
                model=model,
                response_model=ClassifierAgentResponse,
                messages=[{"role": "system", "content": prompts[model]}, *conversation],
                temperature=0,
            )
            break
        except Exception as e:
            print(f"Error with model {model}: {e}")
            continue


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
            matched_tool = next((t for t in state.classifier_available_tools if t["name"] == tool_call.name), None)
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

    return {
        "messages": [ai_message],
        "classifier_tool_calls": response.tool_calls,
        "classifier_iteration": state.classifier_iteration + 1,
        "answer": response.answer,
        "classifier_final_answer": response.final_answer,
        "retrieved_context_ids": response.retrieved_context_ids,
        "trace_id": trace_id
    }