from pydantic import BaseModel, Field
from typing import List
import instructor
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
    arguments: dict

class RAGUsedContext(BaseModel):
    id: int
    description: str

class AgentResponse(BaseModel):
    answer: str
    tool_calls: List[ToolCall] = Field(default_factory=list)
    final_answer: bool = Field(default=False)
    retrieved_context_ids: List[RAGUsedContext]


@traceable(
    name="agent_node",
    run_type="llm",
    metadata={"ls_provider": config.GENERATION_MODEL_PROVIDER, "ls_model_name": config.GENERATION_MODEL}
)
def agent_node(state):

    # prompt_template = prompt_template_config(config.RAG_PROMPT_TEMPLATE_PATH, "rag_generation")

    prompt_template =  """You are a Fraud Analyst Assistant. The user is a Fraud Analyst and your job is to determine whether a job posting is real or fraudulent. 
        Use your knowledge of common fraud indicators, best practices for job verification, and any relevant information you can retrieve to support your answer.
        Always explain your reasoning and provide actionable advice. 

        User may provide either the full job posting text, job title, or a job ID. 

        If a job ID is provided, retrieve the corresponding job posting details before analysis.

        You will be given a question and a list of tools you can use to answer that question.

        If the user specifically requests classification on a job posting to tell if it is real or fake, 
        you should first retrieve the posting using the `get_formatted_context` tool if you haven't already, 
        then call the `get_prediction` to classify it.

        <Available tools>
        {{ available_tools | tojson }}
        </Available tools>

        After the tools are used you will get the outputs from the tools.

        When calling tools, format your response as:

        <tool_call>
        {"name": "tool_name", "arguments": {...}}
        </tool_call>

        Use names specifically provided in the available tools. Don't add any additional text to the names.

        You should tend to use tools when additional information is needed to answer the question.

        If you set final_answer to True, you should not use any tools.

        If you have already retrieved a job posting and need to assess if it's fraudulent, you should use the `get_prediction` tool.


        Instructions:
        - Carefully analyze the provided job details and user input above.
        - Do not return the same tool call more than once.
        - Use up-to-date information about job scams, legitimate job ad characteristics, and known fraud patterns.
        - Provide a clear verdict: "Likely Real", "Likely Fraudulent", or "Uncertain".
        - Explain your reasoning with specific evidence from the posting, user input, and all retrieved information.
        - List any red flags or positive signs you identified.
        - Offer actionable advice to the user.
"""

    prompt_template = Template(prompt_template)

    prompt = prompt_template.render(
        available_tools=state.available_tools
    )

    messages = state.messages

    conversation = []

    for msg in messages:
        conversation.append(lc_messages_to_regular_messages(msg))

    response, raw_response = client.chat.completions.create_with_completion(
        model="gpt-4.1-mini",
        response_model=AgentResponse,
        messages=[{"role": "system", "content": prompt}, *conversation],
        temperature=0.5,
    )

    # Extract the tool call (assuming single tool call here for simplicity)
    try:
        tool_calls = raw_response.choices[0].message.tool_calls
        if not tool_calls:
            raise ValueError("No tool calls in response")

        tool_call = tool_calls[0]
        arguments_str = tool_call.function.arguments  # This is a JSON string
        logger.info(f"Raw tool call arguments (string): {arguments_str}")

        # Parse JSON string to a dict
        arguments_dict = json.loads(arguments_str)
        logger.info(f"Parsed tool call arguments (dict): {arguments_dict}")

        # Validate and parse into your Pydantic model
        response = AgentResponse.parse_obj(arguments_dict)

    except Exception as e:
        logger.error(f"Failed to parse tool call arguments: {e}")
        # Optionally, handle or re-raise error

    # Your existing logic to construct AIMessage etc. from parsed response
    if response.tool_calls and not response.final_answer:
        tool_calls_list = []
        for i, tc in enumerate(response.tool_calls):
            tool_calls_list.append({
                "id": f"call_{i}",
                "name": tc.name,
                "args": tc.arguments,
            })

    # logger.info(f"Raw assistant response (tool calls): {raw_response}")

    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": raw_response.usage.prompt_tokens,
            "output_tokens": raw_response.usage.completion_tokens,
            "total_tokens": raw_response.usage.total_tokens,
        }
        trace_id = str(getattr(current_run, "trace_id", current_run.id))

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
        "tool_calls": response.tool_calls,
        "iteration": state.iteration + 1,
        "answer": response.answer,
        "final_answer": response.final_answer,
        "retrieved_context_ids": response.retrieved_context_ids,
        "trace_id": trace_id
    }