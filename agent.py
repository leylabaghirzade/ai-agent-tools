import json
import os
import time
from typing import Any, Dict
from dotenv import load_dotenv
from groq import Groq
from tools import GROQ_TOOLS_SCHEMA, TOOL_MAP

load_dotenv()

MODEL_COST_RATES = {
    "llama-3.1-8b-instant": {"input": 0.00005, "output": 0.00008},
    "openai/gpt-oss-120b": {"input": 0.0006, "output": 0.0012},
}

FEW_SHOT_SYSTEM_PROMPT = """
You are a tool-calling assistant. Choose tools carefully and avoid extra tool calls.
Examples:
User: "Convert 100 C to F"
Assistant: Call convert_temperature(value=100, from_unit='C', to_unit='F')

User: "What's the weather in Baku in Fahrenheit?"
Step 1: Call get_current_weather(location='Baku')
Step 2: Take temperature from result and call convert_temperature.
"""


class AgentExecutionEngine:

    def __init__(
        self,
        model: str = "openai/gpt-oss-120b",
        max_iterations: int = 5,
        enable_few_shot: bool = False,
    ):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = model
        self.max_iterations = max_iterations
        self.enable_few_shot = enable_few_shot

    def run(self, user_query: str) -> Dict[str, Any]:
        messages = []
        if self.enable_few_shot:
            messages.append(
                {"role": "system", "content": FEW_SHOT_SYSTEM_PROMPT.strip()}
            )
        messages.append({"role": "user", "content": user_query})

        total_prompt_tokens = 0
        total_completion_tokens = 0
        execution_trace = []
        start_time = time.time()
        iteration = 0

        while iteration < self.max_iterations:
            iteration += 1

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=GROQ_TOOLS_SCHEMA,
                tool_choice="auto",
                temperature=0.0,
            )

            if response.usage:
                total_prompt_tokens += response.usage.prompt_tokens
                total_completion_tokens += response.usage.completion_tokens

            response_msg = response.choices[0].message
            tool_calls = response_msg.tool_calls

            msg_dict = response_msg.model_dump()
            messages.append(
                {k: v for k, v in msg_dict.items() if v is not None}
            )

            if not tool_calls:
                total_latency = round(time.time() - start_time, 3)
                rates = MODEL_COST_RATES.get(
                    self.model, {"input": 0.0001, "output": 0.0002}
                )
                estimated_cost = (
                    (total_prompt_tokens / 1000) * rates["input"]
                ) + ((total_completion_tokens / 1000) * rates["output"])

                return {
                    "output": response_msg.content or "",
                    "latency": total_latency,
                    "prompt_tokens": total_prompt_tokens,
                    "completion_tokens": total_completion_tokens,
                    "total_tokens": total_prompt_tokens + total_completion_tokens,
                    "estimated_cost_usd": round(estimated_cost, 6),
                    "iterations": iteration,
                    "trace": execution_trace,
                }

            for call in tool_calls:
                fn_name = call.function.name
                fn_args = json.loads(call.function.arguments)

                try:
                    if fn_name in TOOL_MAP:
                        result = TOOL_MAP[fn_name](**fn_args)
                    else:
                        result = f"Error: Tool '{fn_name}' does not exist."
                except Exception as err:
                    result = f"Error executing tool '{fn_name}': {str(err)}"

                execution_trace.append(
                    {
                        "step": iteration,
                        "tool": fn_name,
                        "args": fn_args,
                        "result": result,
                    }
                )

                messages.append(
                    {
                        "tool_call_id": call.id,
                        "role": "tool",
                        "name": fn_name,
                        "content": str(result),
                    }
                )

        return {
            "output": "Execution stopped: Max iteration limit reached.",
            "latency": round(time.time() - start_time, 3),
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_prompt_tokens + total_completion_tokens,
            "estimated_cost_usd": 0.0,
            "iterations": iteration,
            "trace": execution_trace,
        }


# Alias for backwards compatibility
AdaptedAgentEngine = AgentExecutionEngine