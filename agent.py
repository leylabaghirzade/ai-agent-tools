import json
import os
import time
from dotenv import load_dotenv
from groq import Groq, RateLimitError
from tools import GROQ_TOOLS_SCHEMA, TOOL_MAP

load_dotenv()


class AgentExecutionEngine:

    def __init__(self, max_iterations: int = 5):
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY not found in environment!")

        self.client = Groq(api_key=groq_api_key)
        self.max_iterations = max_iterations
        self.model = self._get_best_available_model()
        print(f"🤖 Agent initialized using model: {self.model}")

    def _get_best_available_model(self) -> str:
        """Selects the best available model for general tool calling."""
        try:
            available = [m.id for m in self.client.models.list().data]
        except Exception:
            available = []

        preferred_models = [
            "openai/gpt-oss-120b",
            "qwen/qwen3.6-27b",
            "openai/gpt-oss-20b",
            "groq/compound",
        ]

        for model in preferred_models:
            if model in available:
                return model

        return available[0] if available else "openai/gpt-oss-120b"

    def run(self, user_query: str) -> str:
        print(f"\n==========================================")
        print(f"🤖 AGENT USER QUERY: '{user_query}'")
        print(f"==========================================")

        system_instruction = (
            "You are an AI assistant equipped with specialized tools for weather, temperature conversion, and arithmetic calculations.\n"
            "Guidelines:\n"
            "1. Always use available tools for supported queries.\n"
            "2. Tool calls must use valid JSON with double quotes.\n"
            "3. Execute multi-step operations sequentially (one tool call per iteration).\n"
            "4. Temperature conversion strictly supports 'C' and 'F'. Do not invoke tools for unsupported units (e.g., Kelvin).\n"
            "5. Refuse calculation directly for division by zero or non-mathematical code/imports."
        )

        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": "Convert 50 Kelvin to Celsius"},
            {
                "role": "assistant",
                "content": "I can only convert temperatures between Celsius ('C') and Fahrenheit ('F'). Kelvin is not currently supported.",
            },
            {"role": "user", "content": "Calculate 10 / 0"},
            {
                "role": "assistant",
                "content": "Division by zero is mathematically undefined.",
            },
            {"role": "user", "content": "Calculate import os; os.system('ls')"},
            {
                "role": "assistant",
                "content": "Invalid expression. Only standard mathematical arithmetic is supported.",
            },
            {"role": "user", "content": user_query},
        ]

        iteration = 0

        while iteration < self.max_iterations:
            iteration += 1
            print(f"\n--- [Iteration {iteration}/{self.max_iterations}] ---")

            response = None
            for attempt in range(3):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        tools=GROQ_TOOLS_SCHEMA,
                        tool_choice="auto",
                        temperature=0.0,
                    )
                    break
                except RateLimitError:
                    print("⏳ Agent hit rate limit. Retrying in 10 seconds...")
                    time.sleep(10)

            if not response:
                return "Error: Unable to complete request due to rate limit."

            response_message = response.choices[0].message
            tool_calls = response_message.tool_calls

            msg_dict = response_message.model_dump()
            cleaned_msg = {k: v for k, v in msg_dict.items() if v is not None}
            messages.append(cleaned_msg)

            if not tool_calls:
                print("🟢 [Agent Reasoning]: No further tools needed.")
                final_content = response_message.content or ""
                print(f"\n✨ Final Answer:\n{final_content}")
                return final_content

            for tool_call in tool_calls:
                tool_name = tool_call.function.name
                tool_id = tool_call.id

                try:
                    tool_args = json.loads(tool_call.function.arguments)
                except Exception:
                    tool_args = {}

                print(f"🛠️  [Tool Call Selected]: {tool_name}(args={tool_args})")

                if tool_name in TOOL_MAP:
                    try:
                        result = TOOL_MAP[tool_name](**tool_args)
                        print(f"📥 [Tool Result]: {result}")
                    except Exception as e:
                        result = f"Error executing tool: {str(e)}"
                        print(f"❌ [Tool Error]: {result}")
                else:
                    result = f"Error: Tool '{tool_name}' not available."

                messages.append(
                    {
                        "tool_call_id": tool_id,
                        "role": "tool",
                        "name": tool_name,
                        "content": str(result),
                    }
                )

        print(f"\n⚠️ [GUARDRAIL TRIGGERED]: Max iteration limit ({self.max_iterations}) reached.")
        return "Max step limit reached."