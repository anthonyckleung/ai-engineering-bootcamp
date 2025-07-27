from pydantic import BaseModel, Field
from typing import List
import instructor
from openai import OpenAI
from langsmith import traceable, get_current_run_tree
from langchain_core.messages import AIMessage

from api.rag.utils.utils import lc_messages_to_regular_messages
from api.rag.utils.utils import prompt_template_config
from api.core.config import config


from jinja2 import Template

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

When calling tools, always use:

<tool_call>
{"name": "tool_name", "arguments": {...}}
</tool_call>

Use names specifically provided in the available tools. Don't add any additional text to the names.

You should tend to use tools when additional information is needed to answer the question.

If you set final_answer to True, you should not use any tools.

If you have already retrieved a job posting and need to assess if it's fraudulent, you should use the `get_prediction` tool.


Instructions:
   1. Carefully analyze the provided job details and user input above.
   2. Use up-to-date information about job scams, legitimate job ad characteristics, and known fraud patterns.
   3. Provide a clear verdict: "Likely Real", "Likely Fraudulent", or "Uncertain".
   4. Explain your reasoning with specific evidence from the posting, user input, and all retrieved information.
   5. List any red flags or positive signs you identified.
   6. Offer actionable advice to the user.
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

    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": raw_response.usage.prompt_tokens,
            "output_tokens": raw_response.usage.completion_tokens,
            "total_tokens": raw_response.usage.total_tokens,
        }

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
        "retrieved_context_ids": response.retrieved_context_ids
    }