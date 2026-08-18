from agent import AgentExecutionEngine

BENCHMARK_SUITE = [
    {
        "id": "Q1",
        "query": "What is 25 multiplied by 14 divided by 2?",
        "expected_tool": "calculate_math",
    },
    {
        "id": "Q2",
        "query": "What's the current weather in Baku right now?",
        "expected_tool": "get_current_weather",
    },
    {
        "id": "Q3",
        "query": "Convert 30 Celsius to Fahrenheit.",
        "expected_tool": "convert_temperature",
    },
    {
        "id": "Q4",
        "query": "What is the weather in London, and what is that temperature in Fahrenheit?",
        "expected_tool": "get_current_weather",
    },
    {
        "id": "Q5",
        "query": "Calculate (100 - 45) * 3 / 5.",
        "expected_tool": "calculate_math",
    },
    {
        "id": "Q6",
        "query": "What is 0 divided by 0?",
        "expected_tool": "calculate_math",
    },
    {
        "id": "Q7",
        "query": "Get weather for Sumgayit.",
        "expected_tool": "get_current_weather",
    },
    {
        "id": "Q8",
        "query": "Convert 98.6 F to C.",
        "expected_tool": "convert_temperature",
    },
    {
        "id": "Q9",
        "query": "What is 2 to the power of 10?",
        "expected_tool": "calculate_math",
    },
    {
        "id": "Q10",
        "query": "Convert 0 C to F.",
        "expected_tool": "convert_temperature",
    },
    {
        "id": "Q11",
        "query": "What is the weather in Sumgayit in C?",
        "expected_tool": "get_current_weather",
    },
    {
        "id": "Q12",
        "query": "Check if temperature 25C is warm and convert it to F.",
        "expected_tool": "convert_temperature",
    },
    {
        "id": "Q13",
        "query": "Calculate 15 * 15.",
        "expected_tool": "calculate_math",
    },
    {
        "id": "Q14",
        "query": "What is the temperature in London in C?",
        "expected_tool": "get_current_weather",
    },
    {
        "id": "Q15",
        "query": "Convert 100 C to F.",
        "expected_tool": "convert_temperature",
    },
    {
        "id": "Q16",
        "query": "What is 500 divided by 25?",
        "expected_tool": "calculate_math",
    },
]


def evaluate_run(enable_few_shot: bool):
    engine = AgentExecutionEngine(enable_few_shot=enable_few_shot)
    passed = 0
    total_cost = 0.0
    total_latency = 0.0

    mode = "Few-Shot Adapted" if enable_few_shot else "Zero-Shot Baseline"
    print(f"\n--- Running Benchmark Evaluation [{mode}] ---")

    for item in BENCHMARK_SUITE:
        res = engine.run(item["query"])
        tools_used = [t["tool"] for t in res["trace"]]
        is_pass = item["expected_tool"] in tools_used
        if is_pass:
            passed += 1

        total_cost += res["estimated_cost_usd"]
        total_latency += res["latency"]

    accuracy = (passed / len(BENCHMARK_SUITE)) * 100
    avg_latency = total_latency / len(BENCHMARK_SUITE)

    print(f"Accuracy: {accuracy:.1f}% ({passed}/{len(BENCHMARK_SUITE)})")
    print(f"Avg Latency: {avg_latency:.3f}s")
    print(f"Total USD Cost: ${total_cost:.6f}")
    return accuracy, avg_latency, total_cost


if __name__ == "__main__":
    evaluate_run(enable_few_shot=False)
    evaluate_run(enable_few_shot=True)