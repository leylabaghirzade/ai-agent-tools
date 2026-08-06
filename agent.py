import os
import json
from dotenv import load_dotenv
from groq import Groq
from tools import GROQ_TOOLS_SCHEMA, TOOL_MAP

load_dotenv()


class AgentExecutionEngine:

    def __init__(self, max_iterations: int = 5):
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY not found in environment!")

        self.client = Groq(api_key=groq_api_key)
        self.model = "llama-3.3-70b-versatile"
        self.max_iterations = max_iterations

    def run(self, user_query: str):
        print(f"\n==========================================")
        print(f"🤖 AGENT USER QUERY: '{user_query}'")
        print(f"==========================================")

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an AI agent with access to tools. "
                    "When asked to perform a complex request (e.g., weather lookup followed by temperature conversion), "
                    "call ONE tool first, receive its output, and then decide if another tool is required."
                ),
            },
            {"role": "user", "content": user_query},
        ]

        iteration = 0

        while iteration < self.max_iterations:
            iteration += 1
            print(f"\n--- [Iteration {iteration}/{self.max_iterations}] ---")

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=GROQ_TOOLS_SCHEMA,
                tool_choice="auto",
                temperature=0.0,
            )

            response_message = response.choices[0].message
            tool_calls = response_message.tool_calls

            # Append the assistant's turn to message history
            messages.append(response_message)

            if not tool_calls:
                print("🟢 [Agent Reasoning]: No further tools needed.")
                print(f"\n✨ Final Answer:\n{response_message.content}")
                return response_message.content

            # Process tool calls
            for tool_call in tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)
                tool_id = tool_call.id

                print(
                    f"🛠️  [Tool Call Selected]: {tool_name}(args={tool_args})"
                )

                if tool_name in TOOL_MAP:
                    try:
                        result = TOOL_MAP[tool_name](**tool_args)
                        print(f"📥 [Tool Result]: {result}")
                    except Exception as e:
                        result = f"Error executing tool: {str(e)}"
                        print(f"❌ [Tool Error]: {result}")
                else:
                    result = f"Error: Tool '{tool_name}' not available."

                # Append tool result matching native OpenAI/Groq spec
                messages.append(
                    {
                        "tool_call_id": tool_id,
                        "role": "tool",
                        "name": tool_name,
                        "content": str(result),
                    }
                )

        print(
            f"\n⚠️ [GUARDRAIL TRIGGERED]: Max iteration limit ({self.max_iterations}) reached to prevent infinite loops."
        )
        return "Max step limit reached."