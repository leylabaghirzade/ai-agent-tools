from agent import AgentExecutionEngine

if __name__ == "__main__":
    agent = AgentExecutionEngine(max_iterations=5)

    # Test 1: Direct Tool Selection (Weather)
    agent.run("What is the weather like in Baku right now?")

    # Test 2: Chained Tool Usage (Weather -> Temperature Conversion)
    agent.run(
        "What's the weather in Baku right now, and convert its temperature to Fahrenheit?"
    )

    # Test 3: Math Evaluation Tool
    agent.run("Calculate (145 * 12) / 4 + 89")

    # Test 4: Direct Answer (No Tool Needed)
    agent.run("What is the capital of Azerbaijan?")