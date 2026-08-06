# AI Agent Tool Execution Engine

A modular Python AI agent built using native Groq LLM function calling, capable of dynamically selecting tools, chaining tool executions across iterations, and preventing infinite loops.

## Features
- **Native Function Calling:** Uses `groq` SDK with OpenAI-compatible tool definitions.
- **Sequential Tool Chaining:** Evaluates tool outputs and dynamically passes intermediate outputs to dependent tools.
- **Infinite Loop Protection:** Enforces a maximum iteration limit (`max_iterations = 5`).
- **Trace Logging:** Full visibility into tool selection, arguments, and intermediate results.

## Requirements
- Python 3.10+
- Groq API Key

## Setup
1. Clone the repository:
   ```bash
   git clone <YOUR_REPOSITORY_URL>
   cd ai-agent-tools