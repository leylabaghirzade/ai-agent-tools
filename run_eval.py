import json
import os
import time
from dotenv import load_dotenv
from groq import Groq, RateLimitError
from agent import AgentExecutionEngine

load_dotenv()

judge_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def get_judge_model() -> str:
    """Selects an available model for LLM-as-a-Judge."""
    try:
        available = [m.id for m in judge_client.models.list().data]
    except Exception:
        available = []

    preferred_models = [
        "openai/gpt-oss-120b",
        "qwen/qwen3.6-27b",
        "openai/gpt-oss-20b",
    ]

    for model in preferred_models:
        if model in available:
            return model

    return available[0] if available else "openai/gpt-oss-120b"


JUDGE_MODEL = get_judge_model()


def llm_as_a_judge(query: str, expected: str, actual: str) -> dict:
    """Evaluates agent response correctness on a scale of 0.0 to 1.0 using LLM-as-a-Judge."""
    prompt = f"""
    You are an impartial AI judge evaluating the accuracy of an AI Agent response.

    User Query: "{query}"
    Expected Reference Answer / Behavior: "{expected}"
    Actual Agent Output: "{actual}"

    Evaluate the actual response against the expected answer.
    Return ONLY a JSON object with this structure:
    {{
      "score": <float between 0.0 and 1.0>,
      "passed": <boolean true if score >= 0.8 else false>,
      "reasoning": "<brief explanation of score>"
    }}
    """
    for attempt in range(3):
        try:
            response = judge_client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            content = response.choices[0].message.content or "{}"
            return json.loads(content)
        except RateLimitError:
            print("⏳ Judge hit rate limit. Waiting 10 seconds...")
            time.sleep(10)
        except Exception as e:
            return {
                "score": 0.0,
                "passed": False,
                "reasoning": f"Judge evaluation failed: {str(e)}",
            }

    return {"score": 0.0, "passed": False, "reasoning": "Judge rate limit exceeded"}


def run_evaluation_suite():
    dataset_path = "dataset/test_set.json"
    if not os.path.exists(dataset_path):
        print(f"❌ Error: Could not find dataset file at {dataset_path}")
        return

    with open(dataset_path, "r") as f:
        test_cases = json.load(f)

    agent = AgentExecutionEngine(max_iterations=5)
    results = []

    total_latency = 0
    passed_count = 0

    print(f"🚀 Starting Automated Test Evaluation Suite using Judge Model: {JUDGE_MODEL}...\n")

    for item in test_cases:
        qid = item.get("id", "UNKNOWN")
        query = item.get("query", "")
        expected = item.get("expected_answer") or item.get("expected", "")

        print(f"[{qid}] Running: '{query}'")

        time.sleep(1)

        start_time = time.time()
        actual_output = agent.run(query)
        latency = round(time.time() - start_time, 2)
        total_latency += latency

        judge_res = llm_as_a_judge(query, expected, actual_output)
        is_passed = judge_res.get("passed", False)
        score = judge_res.get("score", 0.0)
        reasoning = judge_res.get("reasoning", "")

        if is_passed:
            passed_count += 1
            status = "PASSED"
        else:
            status = "FAILED"

        print(f"   ↳ Result: {status} | Latency: {latency}s | Score: {score}")
        print(f"   ↳ Reason: {reasoning}\n")

        results.append(
            {
                "id": qid,
                "type": item.get("type", "normal"),
                "query": query,
                "expected": expected,
                "actual": actual_output,
                "latency_sec": latency,
                "score": score,
                "passed": is_passed,
                "reasoning": reasoning,
            }
        )

    total_tests = len(test_cases)
    pass_rate = round((passed_count / total_tests) * 100, 2) if total_tests > 0 else 0
    avg_latency = round(total_latency / total_tests, 2) if total_tests > 0 else 0

    summary = {
        "total_tests": total_tests,
        "passed": passed_count,
        "failed": total_tests - passed_count,
        "pass_rate_pct": pass_rate,
        "average_latency_sec": avg_latency,
    }

    os.makedirs("results", exist_ok=True)
    report_path = "results/baseline_eval.json"
    with open(report_path, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)

    print("==========================================")
    print("📊 EVALUATION SUMMARY")
    print(f"Pass Rate: {pass_rate}% ({passed_count}/{total_tests})")
    print(f"Avg Latency: {avg_latency} seconds")
    print(f"Saved evaluation report to {report_path}")
    print("==========================================")


if __name__ == "__main__":
    run_evaluation_suite()